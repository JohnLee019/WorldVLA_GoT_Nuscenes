#!/usr/bin/env python
"""How much of what we need is recoverable from ONE frame, in the representation
the model actually sees? Gate for the decomposition (Graph-of-Thoughts as
divide-and-conquer) track. Encoding needs 1 GPU for minutes; the probe is CPU.

Why this exists
---------------
sec.1.10 measured that the residual signal a planner would have to exploit is the
3-second acceleration profile: the oracle beats greedy by 0.55 m, but selecting on
a PERFECT first-step signal gains nothing (+0.0205), because the pool's first steps
span 0.2703 m while its 3 s outcomes span 6.9353 m. So the remaining question is not
about selection at all -- it is whether the model could PREDICT the profile better,
i.e. whether the information is in the frame and simply unused.

Decomposition (V2V-GoT-style: ask the model intermediate questions and feed the
answers forward) is the one untried way to use more of a single frame, and it
costs weeks (QA supervision + retraining). Before that, this script asks the only
question that can make it pointless:

  ★ THE MODEL NEVER SEES PIXELS. The frame becomes a deterministic centre crop,
    then ~N VQGAN codebook tokens. No amount of prompting, decomposition or
    capacity can recover information that the encoder already threw away.

So we train a small probe on the FROZEN features and compare three numbers:

  A  probe on VQGAN codebook vectors  -- what the 7B actually sees
  B  probe on a reference encoder     -- what is in the image at all
  C  the 7B's own achieved error      -- already measured, sec.1.10a

  C >> A   the 7B is not using what its own representation contains -> the
           decomposition track has something to find (GO)
  B >> A   the compression destroyed it -> resolution/encoder, not prompting
  A ~ B ~ mean-predictor   one frame does not contain it -> sec.1.10 confirmed,
           and the limitation is about the INPUT, not about our model

⚠ THE TRAP THIS SCRIPT IS BUILT AROUND
Driving is mostly "keep going", so a probe that has learned nothing but the
training mean still posts a small MAE. Absolute error is therefore NOT a verdict
input; SKILL OVER THE MEAN PREDICTOR is. The self-test's "empty" world has a large
target mean for exactly this reason: it must read EMPTY despite a flattering MAE.
This is sec.11.5 ("do not move the decision rule after seeing the number") in a
new costume -- the same shape as reading lik_full's top1 rise as identification.

Usage
-----
    # 1) encode (1 GPU, minutes). Once per feature type and split.
    python analyze_input_ceiling.py encode \
        --records ./data/nuscenes_records/nuscenes_v1.0-trainval_train.json \
        --tokenizer_path ../ckpts/Lumina-mGPT-7B-768 \
        --features vqgan --stride 4 --out feats/train_vqgan.npz
    python analyze_input_ceiling.py encode \
        --records ./data/nuscenes_records/nuscenes_val_scenespread.json \
        --tokenizer_path ../ckpts/Lumina-mGPT-7B-768 \
        --features vqgan --out feats/val_vqgan.npz

    # 2) probe + verdict (CPU, seconds)
    python analyze_input_ceiling.py probe \
        --train feats/train_vqgan.npz --val feats/val_vqgan.npz \
        --ref_train feats/train_pixels.npz --ref_val feats/val_pixels.npz

    python analyze_input_ceiling.py --selftest        # no data, no torch

Run it from the repo root (data/ and got_drive/ must be importable).
"""

from __future__ import annotations
# --- 리포 루트를 import 경로에 넣는다 -------------------------------------
# 이 파일은 2026-08-21에 루트에서 이 폴더로 옮겨졌다. 파이썬은 sys.path[0]에
# *스크립트가 있는 폴더*를 넣으므로, 이 두 줄이 없으면 `python analysis/x.py`가
# got_drive / model / xllmx 를 못 찾고 ModuleNotFoundError로 죽는다.
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
# -------------------------------------------------------------------------

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

from analyze_got_csv import cluster_bootstrap_ci, wilcoxon_p

# ---- preregistered decision constants -------------------------------------- #
# fixed here before the run so the verdict cannot be argued into existence (11.5)
T_SKILL = 0.10          # skill over the mean predictor below this = no information
T_GAP = 0.10            # metres. Differences smaller than this are not results.
                        # The scene-cluster CI half-width on avgL2 diffs is ~0.031;
                        # these are first-step errors around 1 m, so 0.10 keeps the
                        # bar at ~10% of the quantity rather than at the floor.

# the 7B's OWN achieved error on the same 600-record scene-spread set, from
# sec.1.10a block [2] (|pool median - GT|, longitudinal). These are the numbers a
# probe has to beat for "the model is leaving information unused" to be true.
C_WP0_LON = 1.0015
C_WP5_LON = 5.9757

TARGETS = ("wp0", "wp5", "resid")
HZ = 5                  # waypoint index of the 3 s horizon
DT_STEPS = 6            # wp5 is 6 steps of 0.5 s after t0, so constant velocity
                        # from wp0 predicts 6 * wp0


# --------------------------------------------------------------------------- #
# targets
# --------------------------------------------------------------------------- #

def build_targets(wp):
    """(T, 2) GT waypoints -> {name: (2,)}.

    wp0    the first step. Its magnitude IS the initial speed (0.5 s apart), so
           this is the quantity past frames / ego status would hand over.
    wp5    the 3 s endpoint. What the headline metric is about.
    resid  wp5 - 6*wp0: what a CONSTANT-VELOCITY extrapolation of the first step
           fails to explain. ★ This is the target that matters. sec.1.9(c) found
           R^2(e5 ~ 6*e0) = 0.77 ACROSS RECORDS, and sec.1.10(d) found that the
           relation does not hold WITHIN a pool -- so `resid` is exactly the part
           neither ego status nor past frames can supply, and the only part a
           better reading of the current frame could win.
    """
    wp = np.asarray(wp, dtype=np.float64)
    return {"wp0": wp[0].copy(),
            "wp5": wp[HZ].copy(),
            "resid": wp[HZ] - DT_STEPS * wp[0]}


# --------------------------------------------------------------------------- #
# encode  (the only part that needs torch / a GPU)
# --------------------------------------------------------------------------- #

def token_grid(n, wh):
    """(grid_h, grid_w) of an n-token image, or None if it cannot be determined.

    ⚠ nuScenes frames are 16:9 and var_center_crop keeps the aspect ratio, so the
    token grid is NOT square. An earlier version assumed it was and silently fell
    back to a global mean, which threw away every positional cue and would have
    biased the whole measurement toward EMPTY. Derive the grid from the cropped
    image instead of guessing, and say so out loud when it cannot be derived.
    """
    if n <= 0:
        return None
    if wh is not None:
        w, h = wh
        f = np.sqrt((w * h) / float(n))         # VQGAN downsample factor
        gw, gh = int(round(w / f)), int(round(h / f))
        if gw > 0 and gh > 0 and gw * gh == n:
            return gh, gw
    side = int(round(np.sqrt(n)))
    if side * side == n:
        return side, side
    return None


def _pool_grid(vecs, pool, grid=None):
    """(N, e) codebook vectors -> pooled feature vector.

    Average-pools the token grid to pool x pool cells and appends the global mean.
    Keeping SOME spatial layout matters: the cues this script is looking for --
    where the lead vehicle is, how far the road runs before it turns, whether the
    brake lights ahead are lit -- are positional, and a single global average
    erases all of them. Keeping ALL of them (N * e is tens of thousands of dims)
    would just overfit.

    `grid` is (h, w) from token_grid(); with grid=None this returns the global
    mean alone, which is a MUCH weaker feature set -- the caller must not let that
    happen silently.
    """
    n, e = vecs.shape
    if grid is None:
        return vecs.mean(axis=0)
    gh, gw = grid
    g = vecs.reshape(gh, gw, e)
    out = []
    for i in range(pool):
        for j in range(pool):
            r0, r1 = i * gh // pool, max((i + 1) * gh // pool, i * gh // pool + 1)
            c0, c1 = j * gw // pool, max((j + 1) * gw // pool, j * gw // pool + 1)
            out.append(g[r0:r1, c0:c1].reshape(-1, e).mean(axis=0))
    out.append(vecs.mean(axis=0))
    return np.concatenate(out, axis=0)


def cmd_encode(a):
    # imported lazily so that `probe` and `--selftest` run on a machine with no
    # torch at all -- the local box is edit-only (sec.4)
    import torch
    from PIL import Image

    from data.item_processor import FlexARItemProcessor_Action_NuScenes
    from got_drive.eval_crop import crop_for_eval

    with open(a.records) as f:
        records = json.load(f)
    records = records[::a.stride]
    if a.limit:
        records = records[:a.limit]
    print(f"[encode] {len(records)} records (stride {a.stride}) from {a.records}")

    ip = None
    tok = None
    if a.features == "vqgan":
        ip = FlexARItemProcessor_Action_NuScenes(tokenizer=a.tokenizer_path,
                                                 target_size=a.resolution)
        tok = ip.chameleon_ori_image_tokenizer
        from got_drive.wm_image_metric import _codebook_vectors
    else:
        # the crop still has to match the eval path, so we need the crop_size_list
        ip = FlexARItemProcessor_Action_NuScenes(tokenizer=a.tokenizer_path,
                                                 target_size=a.resolution)

    X, Y, toks, scenes, n_bad = [], {t: [] for t in TARGETS}, [], [], 0
    with torch.no_grad():
        for i, r in enumerate(records):
            wp = np.asarray(r.get("waypoints") or [], dtype=np.float64)
            if wp.ndim != 2 or wp.shape[0] <= HZ:
                n_bad += 1
                continue
            try:
                img = Image.open(r["images"][0]).convert("RGB")
            except Exception:
                n_bad += 1
                continue
            # SAME deterministic centre crop the eval uses; a probe fed a randomly
            # cropped frame would measure a different pipeline (sec.1.4)
            img = crop_for_eval(img, ip, legacy=False)

            if a.features == "vqgan":
                v = _codebook_vectors(tok, img).detach().float().cpu().numpy()
                grid = token_grid(v.shape[0], img.size)
                if grid is None:
                    # loud, and fatal on the first frame: a silent global-mean
                    # fallback would strip every positional cue and quietly bias
                    # the whole measurement toward EMPTY
                    sys.exit(f"[fatal] cannot derive the token grid: {v.shape[0]} "
                             f"tokens for a {img.size[0]}x{img.size[1]} crop. Fix "
                             f"token_grid() rather than pooling globally -- the "
                             f"spatial layout is the point.")
                if i == 0 or not X:
                    print(f"  token grid {grid[0]}x{grid[1]} = {v.shape[0]} tokens, "
                          f"e_dim {v.shape[1]} -> feature dim "
                          f"{(a.pool * a.pool + 1) * v.shape[1]}", flush=True)
                feat = _pool_grid(v, a.pool, grid)
            else:
                # deliberately WEAK reference: raw greyscale pixels, downsampled.
                # It understates what a good encoder would find, which makes the
                # BOTTLENECK verdict conservative -- if even this beats the VQGAN
                # features, the compression really did throw the signal away.
                small = img.convert("L").resize((a.pixel_size, a.pixel_size))
                feat = (np.asarray(small, dtype=np.float32) / 255.0).reshape(-1)

            X.append(feat.astype(np.float32))
            for t, y in build_targets(wp).items():
                Y[t].append(y)
            toks.append(r["sample_token"])
            scenes.append(r.get("scene", "?"))
            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(records)} encoded", flush=True)

    if not X:
        sys.exit("[fatal] nothing encoded")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    np.savez_compressed(
        a.out,
        X=np.stack(X).astype(np.float16),        # halves the file; refit in f32
        tokens=np.asarray(toks), scenes=np.asarray(scenes),
        features=a.features, pool=a.pool,
        **{f"y_{t}": np.stack(v) for t, v in Y.items()})
    print(f"[encode] wrote {a.out}: X {np.stack(X).shape}, skipped {n_bad}")


# --------------------------------------------------------------------------- #
# probe  (numpy only)
# --------------------------------------------------------------------------- #

def _standardise(Xtr, Xva):
    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0)
    sd[sd < 1e-8] = 1.0
    return (Xtr - mu) / sd, (Xva - mu) / sd


def _ridge(X, Y, lam):
    d = X.shape[1]
    return np.linalg.solve(X.T @ X + lam * np.eye(d), X.T @ Y)


def _fit_predict(Xtr, Ytr, Xva, scenes_tr, lams=(1e1, 1e2, 1e3, 1e4, 1e5), seed=0):
    """Ridge with lambda chosen on a SCENE-held-out slice of the training split.

    Scenes, not rows: nuScenes keyframes inside a scene are repeated looks at one
    manoeuvre, so a row-wise validation split leaks the answer and picks a lambda
    that is far too small (sec.9's largest trap, in model-selection form).
    """
    uniq = sorted(set(scenes_tr))
    rng = np.random.RandomState(seed)
    held = set(rng.permutation(uniq)[: max(1, len(uniq) // 5)])
    m = np.array([s in held for s in scenes_tr])
    if m.all() or not m.any():
        m = np.zeros(len(scenes_tr), bool)
        m[: max(1, len(m) // 5)] = True

    Xa, Xb = Xtr[~m], Xtr[m]
    Ya, Yb = Ytr[~m], Ytr[m]
    Za, Zb = _standardise(Xa, Xb)
    mu_y = Ya.mean(axis=0)
    best, best_lam = None, lams[0]
    for lam in lams:
        W = _ridge(Za, Ya - mu_y, lam)
        err = np.abs((Zb @ W + mu_y) - Yb).mean()
        if best is None or err < best:
            best, best_lam = err, lam

    Ztr, Zva = _standardise(Xtr, Xva)
    mu_y = Ytr.mean(axis=0)
    W = _ridge(Ztr, Ytr - mu_y, best_lam)
    return Zva @ W + mu_y, mu_y, best_lam


def _errs(pred, Y):
    """Per-record errors: longitudinal |dx|, lateral |dy|, and the 2-D norm."""
    d = pred - Y
    return {"lon": np.abs(d[:, 0]), "lat": np.abs(d[:, 1]),
            "l2": np.linalg.norm(d, axis=-1)}


def _clusters(scenes, values):
    out = defaultdict(list)
    for s, v in zip(scenes, values):
        out[s].append(float(v))
    return out


def _ci(scenes, values, n_boot=5000):
    return cluster_bootstrap_ci(_clusters(scenes, values), n_boot=n_boot)


def run_probe(train, val, ref_train=None, ref_val=None, n_boot=5000, seed=0):
    """Fit the probe on each target and print the three-number comparison."""
    print(f"\n{'=' * 78}\n[1] WIRING\n{'=' * 78}")
    s_tr, s_va = set(train["scenes"]), set(val["scenes"])
    overlap = s_tr & s_va
    print(f"  train {len(train['X'])} rows / {len(s_tr)} scenes   "
          f"val {len(val['X'])} rows / {len(s_va)} scenes")
    print(f"  feature dim {train['X'].shape[1]}  ({train.get('features', '?')})")
    if overlap:
        sys.exit(f"[fatal] {len(overlap)} scenes appear in BOTH splits, e.g. "
                 f"{sorted(overlap)[:3]}. Every number below would be leakage: "
                 f"nuScenes keyframes inside one scene are repeated looks at the "
                 f"same manoeuvre (sec.9).")
    print("  ok  train and val scenes are disjoint")
    if ref_train is not None and ref_train["X"].shape[0] != train["X"].shape[0]:
        print(f"  note reference split has {ref_train['X'].shape[0]} train rows vs "
              f"{train['X'].shape[0]}; A/B compare their own splits, not per-record")

    out = {}
    for tgt in TARGETS:
        print(f"\n{'=' * 78}\n[2] TARGET  {tgt}\n{'=' * 78}")
        Ytr, Yva = train[f"y_{tgt}"], val[f"y_{tgt}"]
        row = {}
        for name, tr, va in (("A vqgan", train, val),
                             ("B reference", ref_train, ref_val)):
            if tr is None:
                continue
            pred, mu_y, lam = _fit_predict(tr["X"], tr[f"y_{tgt}"], va["X"],
                                           tr["scenes"], seed=seed)
            e = _errs(pred, va[f"y_{tgt}"])
            base = _errs(np.repeat(mu_y[None, :], len(va["X"]), axis=0),
                         va[f"y_{tgt}"])
            skill = {k: 1.0 - e[k].mean() / max(base[k].mean(), 1e-9) for k in e}
            lo, hi = _ci(va["scenes"], base["lon"] - e["lon"], n_boot)
            row[name] = {"e": e, "base": base, "skill": skill, "lam": lam,
                         "gain_lon_ci": (lo, hi)}

        print(f"\n  {'probe':<14} {'lam':>7} {'MAE lon':>8} {'MAE lat':>8} "
              f"{'MAE l2':>8} {'skill lon':>10} {'skill l2':>9} {'gain ci_sc':>21}")
        for name, r in row.items():
            lo, hi = r["gain_lon_ci"]
            print(f"  {name:<14} {r['lam']:>7.0f} {r['e']['lon'].mean():>8.4f} "
                  f"{r['e']['lat'].mean():>8.4f} {r['e']['l2'].mean():>8.4f} "
                  f"{r['skill']['lon']:>10.1%} {r['skill']['l2']:>9.1%} "
                  f"[{lo:>+9.4f},{hi:>+9.4f}]")
        any_r = next(iter(row.values()))
        print(f"  {'mean predictor':<14} {'--':>7} "
              f"{any_r['base']['lon'].mean():>8.4f} "
              f"{any_r['base']['lat'].mean():>8.4f} "
              f"{any_r['base']['l2'].mean():>8.4f} {0.0:>10.1%} {0.0:>9.1%}")
        ref_c = {"wp0": C_WP0_LON, "wp5": C_WP5_LON}.get(tgt)
        if ref_c is not None:
            print(f"  {'C: the 7B':<14} {'--':>7} {ref_c:>8.4f} "
                  f"{'--':>8} {'--':>8}    <- sec.1.10a, same 600 records")
        else:
            print(f"  {'C: the 7B':<14} {'--':>7} {'n/a':>8}    "
                  f"<- no logged reference for this target")
        print("\n  Read SKILL, not MAE. Driving is mostly 'keep going', so a probe")
        print("  that learned only the training mean still posts a small MAE.")
        out[tgt] = row
    return out


# --------------------------------------------------------------------------- #
# verdict
# --------------------------------------------------------------------------- #

def verdict(res, c_wp0=C_WP0_LON):
    """-> 'GO' | 'BOTTLENECK' | 'SATURATED' | 'EMPTY'.

    The decision is about the DECOMPOSITION track (weeks of QA supervision and
    retraining), so the question it answers is narrow: is there information in
    what the model already sees that the model is not using?

    `c_wp0` is the 7B's own logged error (sec.1.10a). It is a parameter only so
    the self-test can supply a reference matched to its synthetic scale -- on real
    data it must stay at the logged constant.
    """
    print(f"\n{'=' * 78}\nVERDICT -- thresholds fixed in the source before the run\n{'=' * 78}")
    a0 = res["wp0"].get("A vqgan")
    b0 = res["wp0"].get("B reference")
    ar = res["resid"].get("A vqgan")
    if a0 is None:
        sys.exit("[fatal] no VQGAN probe: --train/--val are required")

    a_skill = a0["skill"]["lon"]
    b_skill = b0["skill"]["lon"] if b0 else float("nan")
    a_lon = a0["e"]["lon"].mean()
    b_lon = b0["e"]["lon"].mean() if b0 else float("nan")
    r_skill = ar["skill"]["l2"] if ar else float("nan")
    lo, hi = a0["gain_lon_ci"]

    # gain is (mean predictor - probe), so the probe helps when the LOWER bound is
    # above zero. Getting this backwards makes every world read EMPTY, which is
    # why the self-test asserts four distinct verdicts rather than one expected one.
    c_has_info = a_skill > T_SKILL and lo > 0
    c_beats_7b = a_lon < c_wp0 - T_GAP
    c_ref_better = bool(b0) and (b_lon < a_lon - T_GAP)
    c_resid = bool(ar) and r_skill > T_SKILL

    for name, v, ok in (
            (f"A has skill over the mean   > {T_SKILL:.0%}, CI excludes 0", a_skill, c_has_info),
            (f"A beats the 7B (wp0 lon)    < {c_wp0 - T_GAP:.3f} m", a_lon, c_beats_7b),
            (f"reference beats A           by > {T_GAP:.2f} m", b_lon - a_lon if b0 else float("nan"), c_ref_better),
            (f"resid is predictable        skill > {T_SKILL:.0%}", r_skill, c_resid)):
        print(f"  [{'PASS' if ok else 'FAIL'}]  {name:<52} {v:+.4f}")

    print()
    if not c_has_info and not c_ref_better:
        label = "EMPTY"
        print("  EMPTY. Neither the representation the model sees nor the reference")
        print("  encoder recovers the first step better than predicting the training")
        print("  mean. One frame does not carry it, so decomposition has nothing to")
        print("  decompose: asking the model 'how fast are we going?' cannot invent")
        print("  an answer the pixels never had.")
        print("  This is a RESULT, not a dead end -- it upgrades sec.1.10's limitation")
        print("  from 'our model did not extract it' to 'the input does not contain")
        print("  it', which is a much stronger sentence and is what the paper should")
        print("  say. Do NOT spend the QA-supervision + retraining weeks.")
    elif c_beats_7b and c_has_info:
        # both, deliberately: a probe that beat the 7B while having NO skill would
        # be a statement about the 7B, not about information in the frame, and
        # would be misread as the latter
        label = "GO"
        print("  GO. A small probe on the 7B's OWN features predicts the first step")
        print("  better than the 7B does. The information is inside the")
        print("  representation and the model is not using it, which is exactly the")
        print("  premise the decomposition track needs.")
        print("  Cheapest next step first: this is also consistent with a decoding or")
        print("  training-budget problem, so run E2 (continuous head, 10 min) and")
        print("  finish E1 before committing to QA supervision.")
        if c_resid:
            print("  AND `resid` is predictable, so the part neither ego status nor")
            print("  past frames could supply is visible in the frame. That is the")
            print("  strongest form of this result.")
    elif c_ref_better:
        label = "BOTTLENECK"
        print("  BOTTLENECK. The reference encoder recovers what the VQGAN features")
        print("  do not, so the compression -- not the prompt, not the reasoning")
        print("  structure -- is where the signal is lost. The model never sees")
        print("  pixels, so no amount of decomposition can reach it.")
        print("  The actionable change is resolution or the visual encoder, and note")
        print("  that the reference here is deliberately weak (grey pixels), so this")
        print("  verdict is conservative: a real encoder would show a larger gap.")
    else:
        label = "SATURATED"
        print("  SATURATED. The frame carries some signal, the model already")
        print("  extracts about as much as the probe can, and the reference encoder")
        print("  does not find more. Decomposition would be re-asking for what the")
        print("  model already computes. Nothing here justifies the retraining.")

    print("\n  Note what this does NOT decide: sec.1.10's GENERATOR verdict still")
    print("  holds either way. A better first-step prediction improves ABSOLUTE L2;")
    print("  it does not let the deliberation layer win, because a speed level is")
    print("  something you hit, not something you pick out of a candidate pool.")
    return label


# --------------------------------------------------------------------------- #
# self-test
# --------------------------------------------------------------------------- #

SYNTH_MEAN = np.array([5.0, 0.0])       # a large target mean, on purpose
SYNTH_SPREAD = 0.35                      # so the mean predictor's MAE is ~0.28
SYNTH_C = 0.20                           # stands in for the logged 7B error

# (noise on A's features, noise on B's features). None = pure noise, no signal.
SYNTH_WORLDS = {
    "informative": (0.02, 0.02),         # both recover it; A beats the reference C
    "compressed":  (None, 0.02),         # only the reference encoder recovers it
    "empty":       (None, None),         # nothing anywhere
    "saturated":   (1.60, 1.60),         # real skill, but not enough to beat C
}


def _world(kind, d=24, n_tr_scene=60, n_va_scene=20, per_scene=8, seed=0):
    """Four worlds that share the SAME large target mean.

    ★ The shared mean is the point: the MEAN PREDICTOR posts a small MAE in every
    one of them, so a rule that reads absolute error instead of SKILL passes all
    four. If the verdict ever starts depending on MAE, the worlds stop separating
    and the self-test fails loudly. That is sec.11.5 as an assertion.

    The target -> feature map P is drawn ONCE and shared by the train and val
    splits. (An earlier version drew it per split, which made every world read
    EMPTY: the probe cannot generalise across two different generating processes,
    and that failure looks exactly like "the frame carries nothing".)
    """
    rng = np.random.RandomState(seed)
    P = rng.randn(2, d)
    noise_a, noise_b = SYNTH_WORLDS[kind]

    def gen(n_scene, offset):
        n = n_scene * per_scene
        Y = SYNTH_MEAN + rng.randn(n, 2) * SYNTH_SPREAD
        scenes = np.asarray([f"sc{offset + i // per_scene}" for i in range(n)])
        return Y, scenes

    def feats(Y, noise):
        if noise is None:
            return rng.randn(len(Y), d).astype(np.float32)
        return ((Y - SYNTH_MEAN) @ P + rng.randn(len(Y), d) * noise).astype(np.float32)

    def pack(X, Y, S):
        wp5 = DT_STEPS * Y + rng.randn(len(Y), 2) * 0.1
        return {"X": X, "scenes": S, "features": "synthetic",
                "y_wp0": Y, "y_wp5": wp5, "y_resid": wp5 - DT_STEPS * Y}

    Ytr, Str = gen(n_tr_scene, 0)
    Yva, Sva = gen(n_va_scene, 1000)
    tr = pack(feats(Ytr, noise_a), Ytr, Str)
    va = pack(feats(Yva, noise_a), Yva, Sva)
    # the reference shares A's TARGETS record for record; only the features differ
    rtr = dict(tr, X=feats(Ytr, noise_b))
    rva = dict(va, X=feats(Yva, noise_b))
    return tr, va, rtr, rva


def _selftest():
    print("[selftest] targets")
    wp = np.stack([np.arange(1, 8) * 4.0, np.zeros(7)], 1)
    t = build_targets(wp)
    assert np.allclose(t["wp0"], [4.0, 0.0])
    assert np.allclose(t["wp5"], [24.0, 0.0])
    # a perfectly constant-velocity trajectory has zero residual by construction
    assert np.allclose(t["resid"], [0.0, 0.0]), t["resid"]
    print("  ok  resid is zero for constant velocity (it isolates the accel profile)")

    print("\n[selftest] token grid derivation")
    # nuScenes is 16:9 and var_center_crop keeps the aspect ratio, so the real
    # grid is NOT square. This is the case the first version got wrong.
    assert token_grid(240, (320, 192)) == (12, 20), token_grid(240, (320, 192))
    assert token_grid(64, (128, 128)) == (8, 8)
    assert token_grid(60, None) is None          # non-square with no size to use
    assert token_grid(64, None) == (8, 8)        # square still works without it
    assert token_grid(37, (320, 192)) is None    # nothing consistent -> say so
    print("  ok  non-square grids are derived from the crop, not assumed square")

    print("\n[selftest] pooling keeps spatial layout")
    for grid in ((4, 4), (3, 5)):                # square AND non-square
        v = np.zeros((grid[0] * grid[1], 3), dtype=np.float64)
        v[0] = [1.0, 0.0, 0.0]                   # top-left token only
        f = _pool_grid(v, 2, grid)
        assert len(f) == (4 + 1) * 3, (grid, len(f))
        assert f[0] > 0 and f[3] == 0 and f[6] == 0, (grid, f)
    assert len(_pool_grid(np.zeros((60, 3)), 2, None)) == 3
    print("  ok  a corner signal does not smear, on square and non-square grids")

    labels = {}
    for kind in SYNTH_WORLDS:
        print(f"\n{'#' * 78}\n[selftest] {kind.upper()} world\n{'#' * 78}")
        tr, va, rtr, rva = _world(kind, seed=1)
        res = run_probe(tr, va, rtr, rva, n_boot=400)
        labels[kind] = verdict(res, c_wp0=SYNTH_C)

    print(f"\n{'#' * 78}\n[selftest] assertions\n{'#' * 78}")
    assert labels["informative"] == "GO", labels
    print(f"  ok  informative -> GO")
    assert labels["compressed"] == "BOTTLENECK", labels
    print(f"  ok  compressed  -> BOTTLENECK (A is noise, reference recovers it)")
    assert labels["empty"] == "EMPTY", labels
    print(f"  ok  empty       -> EMPTY")
    assert labels["saturated"] == "SATURATED", labels
    print(f"  ok  saturated   -> SATURATED (skill exists, but does not beat the 7B)")
    assert len(set(labels.values())) == 4, labels
    print("  ok  the four worlds get four DIFFERENT verdicts")

    # the trap: the mean predictor's MAE is small in EVERY world because the target
    # mean is large. If a future edit reads MAE instead of skill, this fails.
    tr, va, rtr, rva = _world("empty", seed=1)
    res = run_probe(tr, va, rtr, rva, n_boot=200)
    mae = res["wp0"]["A vqgan"]["e"]["lon"].mean()
    base = res["wp0"]["A vqgan"]["base"]["lon"].mean()
    assert mae < 0.15 * SYNTH_MEAN[0], mae
    assert abs(mae - base) < 0.25 * base, (mae, base)
    print(f"  ok  the EMPTY world posts a flattering MAE ({mae:.3f}, only "
          f"{mae / SYNTH_MEAN[0]:.0%} of the\n      target) that sits on the mean "
          f"predictor ({base:.3f}) -- the verdict must not,\n      and does not, "
          f"read MAE")

    print("\n[selftest] scene leakage guard")
    tr, va, _, _ = _world("informative", seed=1)
    va_leaky = dict(va, scenes=tr["scenes"][: len(va["scenes"])])
    try:
        run_probe(tr, va_leaky, None, None, n_boot=100)
    except SystemExit as e:
        assert "BOTH splits" in str(e), e
        print("  ok  overlapping scenes abort the run instead of reporting leakage")
    else:
        raise AssertionError("the scene overlap check did not fire")

    print("\nself-test PASS")


# --------------------------------------------------------------------------- #

def _load(path):
    if path is None:
        return None
    if not os.path.exists(path):
        sys.exit(f"[fatal] {path} does not exist -- run the `encode` subcommand first")
    z = np.load(path, allow_pickle=True)
    d = {k: z[k] for k in z.files}
    d["X"] = d["X"].astype(np.float32)
    d["features"] = str(d.get("features", "?"))
    return d


def main():
    ap = argparse.ArgumentParser(
        "single-frame information ceiling: gate for the decomposition track")
    sub = ap.add_subparsers(dest="cmd")

    e = sub.add_parser("encode", help="frames -> frozen features (needs 1 GPU)")
    e.add_argument("--records", required=True)
    e.add_argument("--tokenizer_path", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--features", choices=["vqgan", "pixels"], default="vqgan",
                   help="vqgan = what the 7B actually sees. pixels = a deliberately "
                        "weak reference; it understates what a real encoder finds, "
                        "so a BOTTLENECK verdict built on it is conservative.")
    e.add_argument("--resolution", type=int, default=256)
    e.add_argument("--pool", type=int, default=2,
                   help="pool x pool spatial cells of the token grid, plus the "
                        "global mean")
    e.add_argument("--pixel_size", type=int, default=32)
    e.add_argument("--stride", type=int, default=1,
                   help="take every Nth record (the train split does not need to "
                        "be encoded whole for a probe)")
    e.add_argument("--limit", type=int, default=0)

    p = sub.add_parser("probe", help="fit the probe and print the verdict (CPU)")
    p.add_argument("--train", required=True)
    p.add_argument("--val", required=True)
    p.add_argument("--ref_train", default=None)
    p.add_argument("--ref_val", default=None)
    p.add_argument("--n_boot", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)

    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        _selftest()
        return
    if a.cmd == "encode":
        cmd_encode(a)
        return
    if a.cmd == "probe":
        res = run_probe(_load(a.train), _load(a.val),
                        _load(a.ref_train), _load(a.ref_val), a.n_boot, a.seed)
        verdict(res)
        return
    ap.error("give a subcommand (encode / probe) or --selftest")


if __name__ == "__main__":
    main()
