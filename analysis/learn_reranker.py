#!/usr/bin/env python
"""Stage 1 of the deep-reranker plan: does RAW TRAJECTORY GEOMETRY beat the
hand-made scalar? No GPU, no image, no training data beyond one eval run.

The premise being tested
------------------------
learn_selector.py fitted a nonlinear model to the logged score COMPONENTS --
kinematic, command, likelihood, pool statistics -- and got within-pool rho
0.524 against 0.519 for the best single component, still selecting 0.029 m
worse than not deliberating. The natural objection is that those components are
already-digested 1-D summaries: the model never saw the six waypoints' actual
shape. That objection is correct as far as it goes, and this script is its
direct test.

It matters because the objection is also the premise of the expensive plan
(image + geometry deep reranker, which needs GoT run over the train split --
roughly 7-20 GPU-hours before a single epoch of training). If raw geometry does
not beat the kinematic scalar, the premise "information was lost in the 1-D
compression" is false, and that investment loses its rationale. If it does beat
it, the investment is justified and this script says by how much.

Two traps this is built to avoid
--------------------------------
★ 1. PREDICTING ABSOLUTE ERROR LEARNS SCENE DIFFICULTY, NOT CANDIDATE QUALITY.
   A hard scene has high L2 for EVERY candidate, so a model fitted to absolute
   L2 scores well and contributes nothing to picking within a record. This
   project already fell into exactly this hole once: spearman_d_vs_L2 = 0.4217
   looked like the world model could rank plans, and the confound-free version
   was rho(delta, L2) = -0.009. The target here is therefore the WITHIN-POOL
   z-score of the true error, and the main feature block is each candidate's
   deviation from its own pool's mean trajectory.

★ 2. BETTER RANKING IS NOT BETTER SELECTION. In learn_selector the gradient
   booster had HIGHER within-pool rho than ridge (+0.5260 vs +0.5239) and
   picked WORSE trajectories (3.6024 vs 3.5845) with lower top1. The pool's top
   candidates are near-ties (pos0 3.00 vs pos1 3.08), so average ordering
   quality and argmax quality are decoupled. The verdict below is therefore
   read off the L2 YOU GET BY SELECTING WITH IT, with rho reported alongside
   as diagnosis, never as the decision.

Requires `got_cand_wps` in the csv (added 2026-08-03). Older runs do not have
it; re-run the eval rather than trying to reconstruct it.

Usage
-----
    python learn_reranker.py results/fusion/seg_all/per_sample.csv
    python learn_reranker.py results/headline/*/per_sample.csv
    python learn_reranker.py --selftest
"""
# --- 리포 루트를 import 경로에 넣는다 -------------------------------------
# 이 파일은 2026-08-21에 루트에서 이 폴더로 옮겨졌다. 파이썬은 sys.path[0]에
# *스크립트가 있는 폴더*를 넣으므로, 이 두 줄이 없으면 `python analysis/x.py`가
# got_drive / model / xllmx 를 못 찾고 ModuleNotFoundError로 죽는다.
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
# -------------------------------------------------------------------------

import argparse
import ast
import csv
import os
import sys
from collections import defaultdict

import numpy as np

from analyze_got_csv import _f, _list, spearman
from got_drive.scoring_driving import DEFAULT_DT
from learn_selector import (_mean_rho, _rho_pool, _z, baseline_rhos, fit_gbr,
                            fit_ridge, scene_folds)

KEY = "avgL2@3s"


def _nested(row, key):
    """A list-of-trajectories cell, or None. analyze_got_csv._list flattens to
    floats and cannot parse this shape."""
    v = row.get(key, "")
    if not v:
        return None
    try:
        out = ast.literal_eval(v)
    except (ValueError, SyntaxError):
        return None
    if not isinstance(out, list) or not out:
        return None
    try:
        arrs = [np.asarray(t, dtype=np.float64) for t in out]
    except (TypeError, ValueError):
        return None
    if any(a.ndim != 2 or a.shape != arrs[0].shape for a in arrs):
        return None
    return arrs


# --------------------------------------------------------------------------- #
# geometry features
# --------------------------------------------------------------------------- #

def _traj_geometry(wp, dt=DEFAULT_DT):
    """Per-trajectory descriptors, kept RAW rather than collapsed to one number.

    score_kinematic compresses all of this into a single scalar. Handing the
    parts over separately is the whole point: if the compression is what lost
    the signal, a model given the parts should beat the scalar.
    """
    p = np.vstack([np.zeros((1, 2)), wp])          # ego starts at the origin
    v = np.diff(p, axis=0) / dt                    # (T, 2)
    sp = np.linalg.norm(v, axis=1)                 # (T,)
    a = np.diff(v, axis=0) / dt if len(v) > 1 else np.zeros((1, 2))
    am = np.linalg.norm(a, axis=1)
    head = np.arctan2(v[:, 1], v[:, 0])
    # wrap into (-pi, pi]: without it a heading crossing the +-pi branch cut
    # reads as a ~2pi turn and swamps every other feature
    dhead = ((np.diff(head) + np.pi) % (2 * np.pi) - np.pi
             if len(head) > 1 else np.zeros(1))
    arc = float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)))
    return {
        "speeds": sp, "accel": am, "dheading": dhead,
        "final_y": float(wp[-1, 1]), "final_x": float(wp[-1, 0]),
        "arc": arc, "max_sp": float(sp.max()), "max_a": float(am.max()),
        "max_dh": float(np.abs(dhead).max()),
        "lat_dev": float(np.abs(wp[:, 1]).max()),
    }


FEATURE_MODES = ("geom", "geom+scalar", "scalar")


def pool_features(p, mode="geom"):
    """(n_candidates, n_features) for one record. `p` is a pool dict.

    The largest block is each candidate MINUS the pool's mean trajectory: the
    task is to order candidates inside one record, so the representation is
    built to carry exactly that and to be blind to how hard the scene is
    (trap 1 in the header).

    ★ `mode` decides WHICH QUESTION is being asked, and they are not the same:

      "geom"         geometry only, no hand-made score anywhere in the input.
                     Answers "does raw shape BEAT the scalar?", i.e. was
                     information lost in the 1-D compression. This is the one
                     that decides whether the image reranker's premise holds.
      "geom+scalar"  geometry plus the pool-z of kinematic/command/path_score.
                     Answers "does raw shape ADD to the scalar?". A model given
                     the scalar can always reproduce scalar-only selection, so
                     it can hardly lose -- which is exactly why this mode cannot
                     be used to test the premise, only to size the increment.
      "scalar"       the three pool-z scores alone: the control, reproducing
                     what learn_selector already measured.
    """
    if mode not in FEATURE_MODES:
        raise ValueError(f"mode must be one of {FEATURE_MODES}, got {mode!r}")
    wps = p["wps"]
    stack = np.stack(wps, axis=0)                  # (n, T, 2)
    n = stack.shape[0]
    mean_traj = stack.mean(axis=0)

    scal = None
    if mode in ("geom+scalar", "scalar"):
        cols = []
        for k in ("kin", "cmd", "tot"):
            v = p.get(k)
            # a missing component becomes a constant column rather than
            # dropping the record: the width must not depend on the data
            cols.append(_z(np.asarray(v, float)) if v is not None else np.zeros(n))
        scal = np.column_stack(cols)
    if mode == "scalar":
        return scal

    finals = np.array([w[-1, 1] for w in wps])
    arcs = np.array([_traj_geometry(w)["arc"] for w in wps])
    rows = []
    for i, wp in enumerate(wps):
        g = _traj_geometry(wp)
        d = wp - mean_traj                          # (T, 2) deviation from siblings
        rows.append(np.concatenate([
            d.reshape(-1),                          # 2T  the discriminative block
            np.linalg.norm(d, axis=1),              # T   deviation magnitude/step
            g["speeds"], g["accel"], g["dheading"],
            # shape descriptors. NOTE final_x / arc are near-constant inside a
            # pool, so they cannot help predict a within-pool target directly;
            # they are kept only as modulators, and they are the features most
            # able to identify a scene, so watch them if a flexible model starts
            # overfitting with ~100 training scenes.
            [g["final_y"], g["final_x"], g["arc"], g["max_sp"], g["max_a"],
             g["max_dh"], g["lat_dev"]],
            [_z(finals)[i], _z(arcs)[i], float(n)],
        ]))
    X = np.asarray(rows, dtype=np.float64)
    return X if scal is None else np.column_stack([X, scal])


# --------------------------------------------------------------------------- #
# listwise ranker: optimise the ARGMAX, not the average ordering
# --------------------------------------------------------------------------- #

def fit_listwise(train_pools, feats, epochs=200, hidden=16, lr=2e-3,
                 weight_decay=1e-3, dropout=0.1, seed=0, patience=20):
    """Softmax cross-entropy over each pool, label = the truly best candidate.

    Why listwise rather than the pointwise regression above, or than a uniform
    pairwise margin loss: this project measured that average ordering quality
    and argmax quality are DECOUPLED. In learn_selector the gradient booster had
    the higher within-pool rho (+0.5260 vs +0.5239) and picked worse
    trajectories (3.6024 vs 3.5845), because the pool's top candidates are
    near-ties (pos0 3.00 vs pos1 3.08). A loss over uniformly sampled pairs
    spends most of its capacity separating rank 5 from rank 7, which changes
    nothing. Softmax over the pool puts all of it on "who is first", which is
    the only thing the deliverable depends on.

    Returns a scorer f(X) -> (n,) with HIGHER = BETTER, or None if torch is
    absent. Early-stopped on an inner scene-disjoint split, so `epochs` is a
    ceiling and not a tuned quantity -- tuning it against the reported fold
    would silently turn the held-out number into a fitted one.
    """
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        return None
    torch.manual_seed(seed)
    # CPU only: the GPUs may be busy with an eval, and this model is ~700
    # parameters. Never let this script contend for a card.
    dev = torch.device("cpu")

    # inner split by SCENE, same reason as the outer one: the three seeds are
    # the same records replanned, and records within a scene are consecutive
    # keyframes, so any other split leaks.
    scenes = sorted({p["scene"] for p in train_pools})
    rng = np.random.RandomState(seed)
    rng.shuffle(scenes)
    n_val = max(1, len(scenes) // 5)
    val_sc = set(scenes[:n_val])
    tr = [p for p in train_pools if p["scene"] not in val_sc]
    va = [p for p in train_pools if p["scene"] in val_sc]
    if not tr or not va:
        tr, va = train_pools, train_pools

    # precompute features once instead of once per pool per epoch
    def prep(pool_list):
        out = []
        for p in pool_list:
            X = feats(p)
            if X is None or X.shape[0] < 2:
                continue
            out.append((torch.tensor(X, dtype=torch.float32, device=dev),
                        int(np.argmin(p["vals"])), p["vals"]))
        return out

    tr_d, va_d = prep(tr), prep(va)
    if not tr_d or not va_d:
        return None
    dim = tr_d[0][0].shape[1]

    # standardise on the training fold only
    allX = torch.cat([x for x, _, _ in tr_d], 0)
    mu, sd = allX.mean(0), allX.std(0)
    sd = torch.where(sd < 1e-8, torch.ones_like(sd), sd)

    model = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(),
                          nn.Dropout(dropout), nn.Linear(hidden, 1)).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    def scorer_from(state):
        m = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(),
                          nn.Dropout(dropout), nn.Linear(hidden, 1))
        m.load_state_dict(state)
        m.eval()

        def f(X):
            with torch.no_grad():
                t = (torch.tensor(np.asarray(X), dtype=torch.float32) - mu) / sd
                return m(t).squeeze(-1).numpy()
        return f

    best = (float("inf"), None, 0)
    order = list(range(len(tr_d)))
    for ep in range(epochs):
        model.train()
        rng.shuffle(order)
        for i in order:
            X, tgt, _ = tr_d[i]
            opt.zero_grad()
            s = model((X - mu) / sd).squeeze(-1)
            loss = -torch.log_softmax(s, dim=0)[tgt]
            loss.backward()
            opt.step()
        # model selection on the metric we actually care about -- the L2 you
        # get by selecting with it -- not on the loss
        model.eval()
        with torch.no_grad():
            picks = [float(vals[int(torch.argmax(
                model((X - mu) / sd).squeeze(-1)).item())])
                for X, _, vals in va_d]
        v = float(np.mean(picks))
        if v < best[0] - 1e-6:
            best = (v, {k: t.clone() for k, t in model.state_dict().items()}, ep)
        elif ep - best[2] >= patience:
            break
    return scorer_from(best[1]) if best[1] is not None else None


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #

def load_pools(paths, key=KEY):
    pools, skipped = [], defaultdict(int)
    for path in paths:
        if not os.path.exists(path):
            # eval_got_nuscenes writes per_sample.csv ONCE, after the whole
            # record loop -- there is no partial file to read mid-run.
            run_dir = os.path.dirname(path)
            log = run_dir + ".log"
            hint = ""
            if os.path.exists(log):
                try:
                    with open(log, errors="replace") as lf:
                        prog = [l for l in lf if "plans ok" in l]
                    hint = (f"\n  that run is still going: {prog[-1].strip()}"
                            if prog else "\n  that run has not reached its "
                                         "first progress line yet")
                except OSError:
                    pass
            sys.exit(f"[fatal] {path} does not exist.{hint}\n"
                     f"  per_sample.csv is written only when the eval finishes, "
                     f"so wait for the run to complete.")
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("got_status") != "ok" or r.get("base_status") != "ok":
                    skipped["arm_failed"] += 1
                    continue
                wps = _nested(r, "got_cand_wps")
                vals = _list(r, "got_cand_vals")
                if wps is None or not vals or len(wps) != len(vals) or len(vals) < 3:
                    skipped["no_wps_or_short_pool"] += 1
                    continue
                base = _f(r, f"base_{key}")
                if base is None:
                    skipped["no_base"] += 1
                    continue
                kin = _list(r, "got_cand_kin") or []
                cmd = _list(r, "got_cand_cmd") or []
                tot = _list(r, "got_cand_total") or []
                pools.append({
                    "scene": r.get("scene", "?"), "seed": r.get("seed", ""),
                    "command": r.get("command", "?"),
                    "wps": wps, "vals": np.asarray(vals, float),
                    "base": base,
                    "kin": np.asarray(kin, float) if len(kin) == len(vals) else None,
                    "cmd": np.asarray(cmd, float) if len(cmd) == len(vals) else None,
                    "tot": np.asarray(tot, float) if len(tot) == len(vals) else None,
                    "lik": None,
                })
    return pools, dict(skipped)


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #

def run(pools, k=5, modes=FEATURE_MODES):
    n = len(pools)
    ceil = float(np.mean([p["vals"].min() for p in pools]))
    rand = float(np.mean([p["vals"].mean() for p in pools]))
    greedy = float(np.mean([p["base"] for p in pools]))
    have_scalar = all(p["kin"] is not None for p in pools)
    kin_only = (float(np.mean([p["vals"][int(np.argmax(p["kin"]))] for p in pools]))
                if have_scalar else float("nan"))
    cur = (float(np.mean([p["vals"][int(np.argmax(p["tot"]))] for p in pools]))
           if all(p["tot"] is not None for p in pools) else float("nan"))

    print(f"\n{'=' * 78}\nSTAGE 1: RAW GEOMETRY vs THE HAND-MADE SCALAR\n{'=' * 78}")
    print(f"\n  {n} record-plans / {len({p['scene'] for p in pools})} scenes")
    print(f"    random candidate      {rand:.4f}")
    print(f"    kinematic scalar      {kin_only:.4f}")
    print(f"    current score         {cur:.4f}")
    print(f"    greedy (no deliberation) {greedy:.4f}")
    print(f"    oracle (minADE_C)     {ceil:.4f}")

    if have_scalar:
        print("\n  hand-made components on these records (score vs -true error)")
        for name, (rho, nu, nd) in baseline_rhos(pools).items():
            if rho == rho:
                print(f"    {name:<12} rho {rho:+.4f}   (n={nu}"
                      + (f", {nd} dropped)" if nd else ")"))

    tr_folds, te_folds, n_scenes = scene_folds(pools, k)
    print(f"\n  {k}-fold CV split by SCENE ({n_scenes} scenes)")

    def evaluate(fit_fold, predict_is_error):
        """Run one learner over the folds -> (selected L2, rho, top1) or None.

        predict_is_error: True when the model predicts the error (lower better,
        pointwise regression), False when it predicts a preference score
        (higher better, listwise). Getting this backwards silently reports the
        WORST candidate as the pick, so it is a parameter rather than a guess.
        """
        picks, rhos, top1 = [], [], []
        for tr, te in zip(tr_folds, te_folds):
            out = fit_fold(tr)
            if out is None:
                return None
            for p in te:
                pred = np.asarray(out(p))
                j = int(np.argmin(pred) if predict_is_error else np.argmax(pred))
                picks.append(p["vals"][j])
                top1.append(1.0 if j == int(np.argmin(p["vals"])) else 0.0)
                rhos.append(_rho_pool(-pred if predict_is_error else pred,
                                      p["vals"]))
        rho, nu, nd = _mean_rho(rhos)
        return float(np.mean(picks)), rho, nu, nd, float(np.mean(top1))

    def sk_fold(fitter, mode):
        def go(tr):
            Xtr = np.vstack([pool_features(p, mode) for p in tr])
            ytr = np.concatenate([_z(p["vals"]) for p in tr])
            fitted = fitter(Xtr, ytr)
            f = fitted[0] if isinstance(fitted, tuple) else fitted
            return None if f is None else (lambda p: f(pool_features(p, mode)))
        return go

    def lw_fold(mode):
        def go(tr):
            f = fit_listwise(tr, lambda p: pool_features(p, mode))
            return None if f is None else (lambda p: f(pool_features(p, mode)))
        return go

    results = {}
    for mode in modes:
        width = pool_features(pools[0], mode).shape[1]
        note = {"geom": "GEOMETRY ONLY -- does shape BEAT the scalar?",
                "geom+scalar": "geometry + scalar -- does shape ADD to it?",
                "scalar": "scalar only -- control (learn_selector's question)"}[mode]
        print(f"\n{'=' * 78}\n  [{mode}]  {width} features  --  {note}\n{'=' * 78}")
        for name, fit_fold, is_err in (
                ("ridge (pointwise)", sk_fold(fit_ridge, mode), True),
                ("gradient boosting (pointwise)", sk_fold(fit_gbr, mode), True),
                ("listwise softmax MLP", lw_fold(mode), False)):
            r = evaluate(fit_fold, is_err)
            if r is None:
                print(f"\n  -- {name}: dependency missing (sklearn/torch); skipped")
                continue
            m, rho, nu, nd, t1 = r
            results[(mode, name)] = m
            print(f"\n  -- {name} (held-out)")
            print(f"     mean {KEY} if it selected   {m:.4f}   "
                  f"({m - kin_only:+.4f} vs kinematic scalar, "
                  f"{m - greedy:+.4f} vs greedy)")
            print(f"     within-pool Spearman         {rho:+.4f}   (n={nu}"
                  + (f", {nd} dropped)" if nd else ")"))
            print(f"     top1                         {t1:.4f}")
            print(f"     headroom recovered           "
                  f"{((cur - m) / (cur - ceil) * 100 if cur == cur and cur > ceil else float('nan')):6.1f}%")

    print(f"\n{'-' * 78}\nVERDICT (read the L2, not the rho -- see trap 2 in the header)")
    print("-" * 78)
    geom = {k: v for k, v in results.items() if k[0] == "geom"}
    both = {k: v for k, v in results.items() if k[0] == "geom+scalar"}
    if geom:
        bg = min(geom.values())
        print(f"  best GEOMETRY-ONLY   {bg:.4f}  vs kinematic scalar {kin_only:.4f}"
              f"   ({bg - kin_only:+.4f})")
    if both:
        bb = min(both.values())
        print(f"  best geometry+scalar {bb:.4f}  vs kinematic scalar {kin_only:.4f}"
              f"   ({bb - kin_only:+.4f})")
    print(f"  greedy (the bar that actually matters) {greedy:.4f}")
    best = min(geom.values()) if geom else float("nan")
    if best != best:
        print("  no model ran.")
    elif best < kin_only - 0.02:
        print(f"  Raw geometry BEATS the hand-made scalar by {kin_only - best:.4f} m.")
        print("  -> the '1-D compression lost information' premise holds, and the")
        print("     image+geometry reranker is justified. Note it still has to")
        print(f"     clear greedy ({greedy:.4f}) to matter at all.")
    else:
        print(f"  Raw geometry does NOT beat the hand-made scalar "
              f"({best:.4f} vs {kin_only:.4f}).")
        print("  -> the premise behind the deep reranker is FALSE on this data:")
        print("     nothing was lost in the 1-D compression, so handing a model")
        print("     the full trajectory shape buys nothing. Spending 7-20 GPU-h")
        print("     generating train-split pools for an image model needs a")
        print("     different argument than 'the features were too digested'.")
    print("\n  Either way this is a result: it is the first measurement of whether")
    print("  trajectory SHAPE carries selection signal beyond its scalar summary.")


# --------------------------------------------------------------------------- #

def _selftest():
    rng = np.random.RandomState(0)

    # a pool whose true error depends on a SHAPE property no scalar summary in
    # the feature set captures directly (the sign pattern of lateral wobble):
    # the learner must find it, otherwise this harness cannot detect signal.
    pools = []
    for i in range(500):
        wobble = rng.randn(8)
        wps, vals = [], []
        for w in wobble:
            base = np.stack([np.arange(1, 7) * 4.0, np.zeros(6)], 1)
            wp = base.copy()
            wp[:, 1] += w * np.array([1, -1, 1, -1, 1, -1])   # alternating
            wps.append(wp)
            vals.append(3.0 + abs(w))
        pools.append({"scene": f"s{i // 5}", "seed": "42", "command": "straight",
                      "wps": wps, "vals": np.asarray(vals),
                      "base": 3.5, "kin": None, "cmd": None, "tot": None,
                      "lik": None})
    tr, te, _ = scene_folds(pools, 5)
    X = np.vstack([pool_features(p) for p in tr[0]])
    y = np.concatenate([_z(p["vals"]) for p in tr[0]])
    pred, _w = fit_ridge(X, y)
    picks = [p["vals"][int(np.argmin(pred(pool_features(p))))] for p in te[0]]
    rand = float(np.mean([p["vals"].mean() for p in te[0]]))
    assert np.mean(picks) < rand - 0.1, (np.mean(picks), rand)
    print(f"  ok  recovers a pure shape signal ({np.mean(picks):.3f} < "
          f"random {rand:.3f})")

    # pure noise: must land near random, i.e. no false positive
    npools = []
    for i in range(500):
        wps = [np.stack([np.arange(1, 7) * 4.0, rng.randn(6)], 1) for _ in range(8)]
        npools.append({"scene": f"s{i // 5}", "seed": "42", "command": "straight",
                       "wps": wps, "vals": 3.0 + rng.randn(8) * 0.5,
                       "base": 3.5, "kin": None, "cmd": None, "tot": None,
                       "lik": None})
    tr2, te2, _ = scene_folds(npools, 5)
    X = np.vstack([pool_features(p) for p in tr2[0]])
    y = np.concatenate([_z(p["vals"]) for p in tr2[0]])
    pred2, _ = fit_ridge(X, y)
    picks2 = [p["vals"][int(np.argmin(pred2(pool_features(p))))] for p in te2[0]]
    rand2 = float(np.mean([p["vals"].mean() for p in te2[0]]))
    assert abs(np.mean(picks2) - rand2) < 0.25, (np.mean(picks2), rand2)
    print("  ok  pure noise lands near random (no false positive)")

    # the feature block must be blind to scene difficulty: shifting a whole pool
    # downstream (a harder/faster scene) must not change the deviation features
    wps = [np.stack([np.arange(1, 7) * 4.0, rng.randn(6)], 1) for _ in range(6)]
    mk = lambda w: {"wps": w, "kin": None, "cmd": None, "tot": None}
    f1 = pool_features(mk(wps))
    shifted = [w + np.array([50.0, 0.0]) for w in wps]
    f2 = pool_features(mk(shifted))
    n_dev = 2 * wps[0].shape[0]
    assert np.allclose(f1[:, :n_dev], f2[:, :n_dev]), "deviation block must be shift-invariant"
    print("  ok  the deviation block is invariant to a whole-pool shift "
          "(trap 1: scene difficulty)")

    assert _nested({"got_cand_wps": ""}, "got_cand_wps") is None
    assert _nested({"got_cand_wps": "[[[1,2],[3,4]], [[1,2]]]"}, "got_cand_wps") is None
    got = _nested({"got_cand_wps": "[[[1,2],[3,4]], [[5,6],[7,8]]]"}, "got_cand_wps")
    assert got is not None and len(got) == 2 and got[0].shape == (2, 2)
    print("  ok  got_cand_wps parsing: empty and ragged rejected")

    # the three feature modes must have distinct widths, and 'scalar' must not
    # smuggle any geometry in -- otherwise the geom-vs-scalar comparison, which
    # is the entire point of this script, would be comparing two of the same
    q = dict(pools[0])
    q["kin"] = q["cmd"] = q["tot"] = np.arange(len(q["vals"]), dtype=float)
    w = {m: pool_features(q, m).shape[1] for m in FEATURE_MODES}
    assert w["scalar"] == 3 and w["geom+scalar"] == w["geom"] + 3, w
    print(f"  ok  feature modes {w} (scalar carries no geometry)")

    # listwise: the SIGN CONVENTION is the dangerous part. The listwise scorer
    # returns HIGHER = BETTER while the pointwise models return the predicted
    # ERROR (lower = better). Swapping them silently selects the WORST candidate
    # on every record and still produces a plausible-looking table.
    f = fit_listwise(pools[:300], lambda z: pool_features(z, "geom"),
                     epochs=25, patience=8)
    if f is None:
        print("  --  listwise skipped (torch not importable here)")
    else:
        te = pools[300:]
        picks = [p["vals"][int(np.argmax(f(pool_features(p, "geom"))))] for p in te]
        anti = [p["vals"][int(np.argmin(f(pool_features(p, "geom"))))] for p in te]
        rand = float(np.mean([p["vals"].mean() for p in te]))
        assert np.mean(picks) < np.mean(anti), (
            "argmax must beat argmin -- the sign convention is inverted "
            f"({np.mean(picks):.3f} vs {np.mean(anti):.3f})")
        assert np.mean(picks) < rand, (np.mean(picks), rand)
        print(f"  ok  listwise: argmax {np.mean(picks):.3f} < random "
              f"{rand:.3f} < argmin {np.mean(anti):.3f} (higher = better)")

    print("\nself-test PASS")


def main():
    p = argparse.ArgumentParser(
        "stage 1: does raw trajectory geometry beat the hand-made scalar? (0 GPU)")
    p.add_argument("csv", nargs="*")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--modes", nargs="+", default=list(FEATURE_MODES),
                   choices=list(FEATURE_MODES),
                   help="which feature sets to run. 'geom' is the one that "
                        "decides the premise; the others size and control it.")
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    if a.selftest:
        _selftest()
        return
    if not a.csv:
        p.error("give per_sample.csv files (or --selftest)")

    pools, skipped = load_pools(a.csv)
    if not pools:
        sys.exit("[fatal] no usable pools. This needs `got_cand_wps`, added to "
                 "eval_got_nuscenes on 2026-08-03 -- runs from before that do "
                 "not have it and the column cannot be reconstructed offline. "
                 f"skipped: {skipped}")
    print(f"[load] {len(pools)} record-plans / "
          f"{len({p['scene'] for p in pools})} scenes"
          + (f"   skipped {skipped}" if skipped else ""))
    run(pools, a.folds, a.modes)


if __name__ == "__main__":
    main()
