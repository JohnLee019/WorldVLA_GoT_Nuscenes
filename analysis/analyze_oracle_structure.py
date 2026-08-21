#!/usr/bin/env python
"""Gate (a) for best-of-n distillation: is the 16.3% pool headroom a STABLE
TARGET, or the order statistics of min-of-8? No GPU, no model, no new eval.

Why this exists
---------------
Inference-time selection is closed: nine interventions, a learned selector, a
learned fallback detector, raw-geometry rerankers and an oracle collision signal
all failed, and the fusion (Aggregate) wave was null. The one untried idea is to
move selection from inference time to TRAINING time -- draw candidates on the
train split, keep the one closest to GT, fine-tune the base model to emit it.
The advertised prize is oracle 2.9770 vs greedy 3.5557 (0.579 m, 16.3%).

That prize is not real until this script says so, because min-of-8 is BIASED
LOW BY CONSTRUCTION: with unbiased sampling noise around the model's mean
trajectory, the closest of eight draws lands nearer GT than any of them
individually, and the amount it gains is a property of the noise, not of
anything a network could learn. Buying 10-12 GPU-h of train-split GoT on that
number without measuring it first would repeat the mistake §11.7 says the
project got right three times: measure the ceiling before building.

★ THE TRAP THIS SCRIPT IS BUILT AROUND
--------------------------------------
The obvious test -- average each record's oracle across the three seeds and see
whether it stays good -- DOES NOT MEASURE LEARNABILITY, and reading it that way
would be exactly the spearman_d_vs_L2 = 0.4217 mistake in a new costume. Each
seed's oracle is picked USING GT, so the cross-seed average keeps pointing at
GT no matter what: even in a world where the pool is pure isotropic noise and
nothing whatsoever is learnable, the consensus of m oracles sits about 1.4 sigma
toward GT and looks like a large prize. It is an IN-SAMPLE CEILING. It can kill
the idea (a small ceiling is a real stop) but it can never bless it.

So the blocks below are ordered by how much they can actually decide:

  [2] DECODING GAP (decisive, and free if it fires). The base model was already
      trained on the GT waypoints with the same closs that best-of-n distillation
      would use. A distillation target drawn from the model's own samples is a
      strictly WEAKER version of the target it already had, so it cannot inject
      new supervision. The one mechanism that survives that argument is a
      mode-vs-mean mismatch: greedy token decoding need not be the L2-optimal
      point estimate of the model's own distribution. That mismatch is visible
      offline as "is the pool's centroid better than greedy?" -- and if it is,
      the first thing to try is re-decoding, which costs no training at all.
  [3] TARGET STABILITY (necessary). Does seed 42's oracle point the same way as
      seed 43's? Transfer is measured on an independently sampled pool (pick the
      candidate nearest the other seed's oracle), so a lucky draw cannot pass it.
      If the target is noise, the labels a distiller would fit are noise.
  [4] ★ DISPLACEMENT (decisive, and null-free). Are the good candidates the
      NEIGHBOURS of the trajectory the model already emits, or do they sit
      somewhere else? Compare the best of the half nearest the model's own pick
      against the best of a random half of equal size -- equal sizes, so order
      statistics cancel and only "does nearness to the decode cost accuracy"
      survives. If it costs nothing, min-of-n is order statistics around the
      existing output and a distilled target is just that output with less
      noise, which is what training on GT already asked for.
      ⚠ This block also runs three geometric nulls (reflected / matched donor /
      rotated, all about the pool mean) and they are DIAGNOSTIC ONLY. They
      cannot decide, because "a better mode sits displaced from the decode" and
      "the good candidates ARE the decode's neighbours, with a tail elsewhere
      dragging the mean" have identical pool geometry and score identically
      under every mean-centred null -- and no null can be centred on the decode,
      since candidate 0 is itself a draw (scrambling about it widens the cloud
      and invents alignment). The self-test's "offset" world is exactly that
      shape: the nulls report +0.43 m of alignment where nothing is learnable.
  [5] Direction of the shift (descriptive): is the oracle systematically slower,
      or does it just mirror wherever GT happens to be?

Reference points (600 records / 150 scenes, deterministic crop, seed 42):
greedy 3.5557, current score 3.5954, random candidate 4.6560, oracle 2.9770.
Scene-cluster CI half-width is ~0.031, so anything under ~0.05 m is not a result.

Usage
-----
    python analyze_oracle_structure.py results/headline/ref/per_sample.csv \
        --records_json ./data/nuscenes_records/nuscenes_val_scenespread.json

    # the stability block needs >= 2 seeds
    python analyze_oracle_structure.py \
        results/headline/ref/per_sample.csv \
        results/headline/ref_s43/per_sample.csv \
        results/headline/ref_s44/per_sample.csv \
        --records_json ./data/nuscenes_records/nuscenes_val_scenespread.json

    python analyze_oracle_structure.py --selftest        # no data needed

Run it from the repo root (got_drive/ must be importable as a package).
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
import csv
import itertools
import json
import os
import sys
from collections import defaultdict

import numpy as np

from analyze_got_csv import _f, _list, cluster_bootstrap_ci
from learn_reranker import _nested

# Reuse the eval's own feasibility predicate rather than reimplementing it: a
# private copy that drifted would answer a different question than summary.json.
try:
    from got_drive.scoring_driving import _feasible
except ImportError as e:                                    # pragma: no cover
    sys.exit(f"[fatal] run this from the repo root, not from got_drive/: {e}")

KEY = "avgL2@3s"
HZ = 5                      # avgL2@3s averages per-step distance over wp 0..5

# preregistered decision thresholds -- fixed here so the verdict cannot be
# talked into existence after seeing the numbers (§11.5)
T_CEILING = 0.10            # in-sample ceiling below this -> stop outright
T_PRIZE = 0.05              # smaller than this is inside the noise floor
T_RECOVERY = 0.25           # transfer must recover this share of random->oracle
T_AGREE = 0.70              # cross-seed oracle distance / pool scale


# --------------------------------------------------------------------------- #
# metric
# --------------------------------------------------------------------------- #

def avg_l2(traj, gt, hz=HZ):
    """avgL2@3s: mean over waypoints 0..hz of the per-waypoint distance.

    Identical functional to eval_nuscenes.l2_metrics()['avgL2@3s'] (ST-P3
    convention). NOT the Euclidean norm of the flattened residual -- keep them
    apart or every number here drifts from the ones in the paper table.
    """
    return float(np.linalg.norm(traj[:hz + 1] - gt[:hz + 1], axis=-1).mean())


def _cluster(pools, values):
    """{scene: [value, ...]} for cluster_bootstrap_ci."""
    out = defaultdict(list)
    for p, v in zip(pools, values):
        out[p["scene"]].append(float(v))
    return out


def _ci(pools, diffs, n_boot=5000):
    return cluster_bootstrap_ci(_cluster(pools, diffs), n_boot=n_boot)


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #

def load_gt(records_json, hz=HZ):
    with open(records_json) as f:
        records = json.load(f)
    gt = {}
    for r in records:
        wp = np.asarray(r.get("waypoints") or [], dtype=np.float64)
        if wp.ndim == 2 and wp.shape[0] > hz:
            gt[r["sample_token"]] = wp
    return gt


def load_pools(paths, gt_by_token, key=KEY):
    """One entry per (sample_token, seed) with its candidate pool and GT.

    Every arm-level filter learn_reranker uses applies here too; on top of that
    the record must have GT, because this script recomputes candidate errors
    from the waypoints rather than trusting the rounded logged values alone.
    """
    pools, skipped, seen = [], defaultdict(int), {}
    for path in paths:
        if not os.path.exists(path):
            sys.exit(f"[fatal] {path} does not exist. per_sample.csv is written "
                     f"only when the eval finishes (sec.9), so a running eval has "
                     f"no partial file to read.")
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("got_status") != "ok" or r.get("base_status") != "ok":
                    skipped["arm_failed"] += 1
                    continue
                tok, seed = r.get("sample_token"), str(r.get("seed", ""))
                gt = gt_by_token.get(tok)
                if gt is None:
                    skipped["no_gt"] += 1
                    continue
                # split the causes: "no usable pools" has four very different
                # fixes (re-run the eval / wrong arm / ragged cell / tiny pool)
                # and one lumped counter cannot tell you which one you have
                if not r.get("got_cand_wps"):
                    skipped["no_got_cand_wps_column"] += 1
                    continue
                wps = _nested(r, "got_cand_wps")
                if wps is None:
                    skipped["got_cand_wps_unparsable_or_ragged"] += 1
                    continue
                vals = _list(r, "got_cand_vals")
                if not vals:
                    skipped["no_got_cand_vals"] += 1
                    continue
                if len(wps) != len(vals):
                    skipped["wps_vals_length_mismatch"] += 1
                    continue
                if len(vals) < 3:
                    skipped["pool_smaller_than_3"] += 1
                    continue
                if wps[0].shape[0] <= HZ:
                    skipped["short_horizon"] += 1
                    continue
                base = _f(r, f"base_{key}")
                if base is None:
                    skipped["no_base"] += 1
                    continue
                if (tok, seed) in seen:
                    skipped["duplicate_token_seed"] += 1
                    continue
                seen[(tok, seed)] = path
                pools.append({
                    "token": tok, "seed": seed, "scene": r.get("scene", "?"),
                    "command": r.get("command", "?"),
                    "wps": np.stack(wps, axis=0),          # (n, T, 2)
                    "logged_vals": np.asarray(vals, float),
                    "gt": gt,
                    "base": base,
                    "got": _f(r, f"got_{key}"),
                    "fused": bool(r.get("got_fuse_n")),
                })
    return pools, dict(skipped)


def column_report(paths):
    """Which csvs actually carry the candidate waypoints, and for which seeds.

    got_cand_wps was added to eval_got_nuscenes on 2026-08-03; runs from before
    that log got_cand_vals but not the trajectories, and the column cannot be
    reconstructed offline. Printing this per file turns "no usable pools" into
    "these two arms predate the column, re-run them".
    """
    print(f"\n  {'file':<52} {'got_cand_wps':>12} {'rows':>6}  seeds")
    for path in paths:
        if not os.path.exists(path):
            print(f"  {path:<52} {'MISSING FILE':>12}")
            continue
        with open(path, newline="", encoding="utf-8") as f:
            rd = csv.DictReader(f)
            has = "got_cand_wps" in (rd.fieldnames or [])
            n, seeds, nonempty = 0, set(), 0
            for r in rd:
                n += 1
                seeds.add(str(r.get("seed", "")))
                if r.get("got_cand_wps"):
                    nonempty += 1
        tag = f"{nonempty}/{n}" if has else "ABSENT"
        print(f"  {path:<52} {tag:>12} {n:>6}  {sorted(seeds)}")


# --------------------------------------------------------------------------- #
# [1] wiring
# --------------------------------------------------------------------------- #

def block_wiring(pools, tol=0.01):
    """Recompute every candidate's error from GT and check it against the log.

    This is the join test. got_cand_wps is rounded to 4 decimals and
    got_cand_vals to 3, so agreement to ~1e-3 proves the GT lined up with the
    right record; disagreement means the records_json is not the set the run
    used, and every later block would be quietly meaningless.
    """
    print(f"\n{'=' * 78}\n[1] WIRING\n{'=' * 78}")
    worst, worst_tok, sel_bad, n_fused = 0.0, None, 0, 0
    for p in pools:
        vals = np.array([avg_l2(w, p["gt"]) for w in p["wps"]])
        p["vals"] = vals
        d = float(np.max(np.abs(vals - p["logged_vals"])))
        if d > worst:
            worst, worst_tok = d, p["token"]
        # candidate 0 is the score's pick (the pool is sorted best-first), so it
        # must equal the row's own got_avgL2@3s -- unless the arm fused, where
        # the output is a trajectory no candidate proposed.
        if p["fused"]:
            n_fused += 1
        elif p["got"] is not None and abs(vals[0] - p["got"]) > tol:
            sel_bad += 1

    scenes = len({p["scene"] for p in pools})
    seeds = sorted({p["seed"] for p in pools})
    print(f"  {len(pools)} record-plans / {scenes} scenes / seeds {seeds}")
    print(f"  recomputed vs logged candidate error: max |diff| {worst:.5f}"
          f"  (token {worst_tok})")
    if worst > tol:
        sys.exit(f"[fatal] recomputed candidate errors disagree with the csv by "
                 f"{worst:.4f} > {tol}. The --records_json is almost certainly "
                 f"not the evaluation set this run used; fix that before reading "
                 f"anything below.")
    print(f"  ok  GT join verified (rounding is 1e-3 on vals, 1e-4 on waypoints)")
    if n_fused:
        print(f"  note {n_fused} rows come from a FUSION arm; their output is not "
              f"a pool member, so the 'score pick' row below is skipped for them")
    n_checked = len(pools) - n_fused
    if not n_checked:
        # do NOT print the ok line here: every row was fused, so the check never
        # ran, and an unconditional "ok" would claim a verification that did not
        # happen -- the same shape of mistake as the sanity check that passed a
        # random crop by handing the same object in twice (sec.9)
        print(f"  --  candidate-0 check NOT RUN: all {len(pools)} rows are fused, "
              f"so the arm's output is not a pool member. The pool itself is "
              f"still the reference arm's (verify: greedy 3.5557, minADE_C "
              f"3.0053 for results/fusion/final_top3).")
    elif sel_bad:
        print(f"  WARN {sel_bad}/{n_checked} rows: candidate 0 != the row's own "
              f"got_{KEY}. The pool is supposed to be sorted best-first by the "
              f"selection score, so this many rows disagree with that assumption "
              f"-- treat the 'score pick' line as unreliable.")
    else:
        print(f"  ok  candidate 0 == the arm's own output on all {n_checked} "
              f"checked rows (pool is score-sorted, so 'score pick' is exact)")
    return pools


# --------------------------------------------------------------------------- #
# [2] the decoding gap -- the one block that can justify spending GPU
# --------------------------------------------------------------------------- #

def block_decoding(pools, n_boot=5000):
    """Is the pool's CENTROID better than greedy?

    The argument this measures against: the base model was trained on the GT
    waypoints with the same discrete-token loss a distillation run would use, so
    a target selected from its OWN samples is a strictly attenuated version of
    the supervision it already had. Best-of-n distillation cannot add
    information under that framing. What it CAN fix is a decoding mismatch --
    greedy token decoding returning a mode of the model's distribution when the
    L2-optimal point estimate is a central one. That mismatch, if it exists,
    shows up as the pool's centroid beating greedy, and it can then be captured
    by re-decoding (sample n, aggregate) with no training whatsoever.

    ⚠ The centroid is a fused trajectory: no candidate proposed it, so it never
    passed the feasibility veto (§1.5). The infeasible fraction is printed for
    the same reason run_fusion.sh prints it -- a "win" made of impossible paths
    is a bug, and the reference rate for non-fused arms is 2.5%.
    """
    print(f"\n{'=' * 78}\n[2] DECODING GAP -- is greedy the wrong point estimate?\n{'=' * 78}")
    greedy = np.array([p["base"] for p in pools])
    rows = []

    def add(name, trajs, note=""):
        v = np.array([avg_l2(t, p["gt"]) for t, p in zip(trajs, pools)])
        d = v - greedy
        lo, hi = _ci(pools, d, n_boot)
        infeas = float(np.mean([not _feasible(t) for t in trajs]))
        rows.append((name, float(v.mean()), float(d.mean()), lo, hi, infeas, note))

    add("pool centroid (mean)", [p["wps"].mean(axis=0) for p in pools],
        "all candidates")
    add("pool median", [np.median(p["wps"], axis=0) for p in pools],
        "componentwise")
    add("top-3 by score, mean", [p["wps"][:3].mean(axis=0) for p in pools],
        "cf. sec.1.5 final_top3")

    print(f"\n  {'estimator':<24} {'avgL2@3s':>9} {'vs greedy':>10} "
          f"{'ci_sc':>19} {'infeas':>7}  note")
    print(f"  {'greedy free-run':<24} {greedy.mean():>9.4f} {'--':>10} "
          f"{'--':>19} {'--':>7}  (2.5% reference rate)")
    for name, v, d, lo, hi, inf, note in rows:
        print(f"  {name:<24} {v:>9.4f} {d:>+10.4f} "
              f"[{lo:>+8.4f},{hi:>+8.4f}] {inf:>6.1%}  {note}")

    keep = [i for i, p in enumerate(pools) if not p["fused"]]
    if keep:
        sp = np.array([pools[i]["vals"][0] for i in keep])
        print(f"  {'score pick (the arm)':<24} {sp.mean():>9.4f} "
              f"{sp.mean() - greedy[keep].mean():>+10.4f}")
    orc = np.array([p["vals"].min() for p in pools])
    rnd = np.array([p["vals"].mean() for p in pools])
    print(f"  {'random candidate':<24} {rnd.mean():>9.4f} {rnd.mean() - greedy.mean():>+10.4f}")
    print(f"  {'oracle (minADE_C)':<24} {orc.mean():>9.4f} {orc.mean() - greedy.mean():>+10.4f}"
          f"   <- the advertised prize, in-sample by construction")

    best = min(r[2] for r in rows)
    print(f"\n  -> best aggregate estimator: {best:+.4f} m vs greedy "
          f"(negative = better than greedy).")
    if best < -T_PRIZE:
        print("     A MODE-vs-MEAN GAP EXISTS. Try the free fix first: aggregate")
        print("     at decode time. No training run is needed to collect it.")
    else:
        print("     No mode-vs-mean gap. Greedy already sits where the pool's")
        print("     central estimate sits, so the only mechanism by which")
        print("     best-of-n distillation could beat ordinary GT training on")
        print("     this model is absent.")
    return {"best_aggregate_gain": -best, "greedy": float(greedy.mean()),
            "oracle": float(orc.mean()), "random": float(rnd.mean())}


# --------------------------------------------------------------------------- #
# [3] target stability across independent samplings
# --------------------------------------------------------------------------- #

def _by_token(pools):
    g = defaultdict(dict)
    for p in pools:
        g[p["token"]][p["seed"]] = p
    return g


def block_stability(pools, n_boot=5000, rng_seed=0):
    """Does seed 42's oracle point where seed 43's oracle points?

    Two readings, and only the first is honest:

    ★ TRANSFER (held out over seeds). Take seed s's oracle, and in seed t's
      INDEPENDENTLY SAMPLED pool pick the candidate nearest to it, then score
      that pick. Nothing about pool t's errors is used to make the choice, so a
      lucky draw in s cannot carry itself over: only a direction that survives
      resampling can. Recovery is measured on the random -> oracle range of pool
      t, the same scale analyze_got_csv uses for the score.
      The control transfers a RANDOM candidate of s instead of its oracle. It is
      not optional: the pools of two seeds overlap in shape, so "nearest to
      something from the other seed" already picks a typical candidate, and the
      control is what separates that from real agreement.

    ⚠ CONSENSUS (in-sample ceiling only -- see the header). Averaging m oracles
      keeps m GT-selected picks, so it points at GT even when nothing is
      learnable. Reported because a SMALL value still kills the idea, never as
      evidence for it.
    """
    print(f"\n{'=' * 78}\n[3] TARGET STABILITY across seeds\n{'=' * 78}")
    grouped = _by_token(pools)
    seeds = sorted({p["seed"] for p in pools})
    full = [t for t, d in grouped.items() if len(d) == len(seeds)]
    if len(seeds) < 2 or len(full) < 30:
        print(f"  skipped: needs >= 2 seeds on the same records "
              f"(have {len(seeds)} seeds, {len(full)} shared records).")
        print("  Run the seed arms (results/headline/ref, ref_s43, ref_s44) and "
              "pass all three csvs.")
        return None
    print(f"  {len(full)} records present in all {len(seeds)} seeds "
          f"({len({grouped[t][seeds[0]]['scene'] for t in full})} scenes)")

    def oracle(p):
        return p["wps"][int(np.argmin(p["vals"]))]

    # -- transfer -----------------------------------------------------------
    rng = np.random.RandomState(rng_seed)
    tr, ctl, orc, rnd, ref_pools = [], [], [], [], []
    for t in full:
        for s_from, s_to in itertools.permutations(seeds, 2):
            src, dst = grouped[t][s_from], grouped[t][s_to]
            o = oracle(src)
            rand_src = src["wps"][rng.randint(len(src["wps"]))]
            # nearest in trajectory space, using the SAME functional as the
            # metric so "near" means near in the units the paper reports
            d_o = [avg_l2(c, o) for c in dst["wps"]]
            d_c = [avg_l2(c, rand_src) for c in dst["wps"]]
            tr.append(dst["vals"][int(np.argmin(d_o))])
            ctl.append(dst["vals"][int(np.argmin(d_c))])
            orc.append(dst["vals"].min())
            rnd.append(dst["vals"].mean())
            ref_pools.append(dst)
    tr, ctl, orc, rnd = map(np.asarray, (tr, ctl, orc, rnd))
    span = rnd.mean() - orc.mean()
    rec = (rnd.mean() - tr.mean()) / span if span > 1e-9 else float("nan")
    rec_c = (rnd.mean() - ctl.mean()) / span if span > 1e-9 else float("nan")
    lo, hi = _ci(ref_pools, tr - ctl, n_boot)

    print(f"\n  -- transfer: pick seed t's candidate nearest seed s's oracle "
          f"(n={len(tr)} ordered pairs)")
    print(f"     random pick in pool t     {rnd.mean():.4f}")
    print(f"     transfer of s's ORACLE    {tr.mean():.4f}   recovers {rec:6.1%}")
    print(f"     control: s's random cand  {ctl.mean():.4f}   recovers {rec_c:6.1%}")
    print(f"     oracle of pool t          {orc.mean():.4f}")
    print(f"     oracle - control          {tr.mean() - ctl.mean():+.4f}  "
          f"ci_sc [{lo:+.4f}, {hi:+.4f}]  <- the actual test")

    # -- oracle agreement ---------------------------------------------------
    agree, scale = [], []
    for t in full:
        os_ = [oracle(grouped[t][s]) for s in seeds]
        agree += [avg_l2(a, b) for a, b in itertools.combinations(os_, 2)]
        allc = np.concatenate([grouped[t][s]["wps"] for s in seeds], axis=0)
        mu = allc.mean(axis=0)
        # mean distance of a candidate to the pooled centre, doubled: the scale
        # two INDEPENDENT random candidates would be apart under symmetry
        scale.append(2.0 * float(np.mean([avg_l2(c, mu) for c in allc])))
    ratio = float(np.mean(agree)) / float(np.mean(scale))
    print(f"\n  -- oracle agreement: mean pairwise distance between seeds' oracles")
    print(f"     {np.mean(agree):.4f} vs {np.mean(scale):.4f} expected for two "
          f"unrelated candidates   ratio {ratio:.3f}")
    print(f"     (ratio 1.0 = the oracles are as far apart as random candidates; "
          f"0 = identical)")

    # -- consensus ceiling --------------------------------------------------
    print(f"\n  -- consensus of m oracles  IN-SAMPLE CEILING, NOT LEARNABILITY")
    ms, vals_m = [], []
    for m in range(1, len(seeds) + 1):
        v = []
        for t in full:
            for sub in itertools.combinations(seeds, m):
                stack = np.stack([oracle(grouped[t][s]) for s in sub], axis=0)
                v.append(avg_l2(stack.mean(axis=0), grouped[t][sub[0]]["gt"]))
        ms.append(m)
        vals_m.append(float(np.mean(v)))
        print(f"     m={m}  {vals_m[-1]:.4f}")
    ceiling = vals_m[-1]
    if len(ms) >= 3:
        # L2(m) ~ A + B/m; A is where averaging out the sampling noise lands
        A, B = np.polyfit(1.0 / np.asarray(ms, float), np.asarray(vals_m), 1)[::-1]
        ceiling = float(A)
        print(f"     m->inf (fit A + B/m over {len(ms)} points)  {A:.4f}   "
              f"B {B:+.4f}")
    greedy = float(np.mean([p["base"] for p in pools]))
    print(f"     greedy {greedy:.4f}  ->  in-sample ceiling is "
          f"{greedy - ceiling:+.4f} m of headroom")
    print("     Read this as an upper bound that a distilled policy could not")
    print("     exceed even with a perfect fit, not as an achievable number.")
    return {"recovery": float(rec), "recovery_control": float(rec_c),
            "transfer_edge": float(tr.mean() - ctl.mean()),
            "transfer_ci": (lo, hi), "agreement_ratio": ratio,
            "ceiling_gain": float(greedy - ceiling)}


# --------------------------------------------------------------------------- #
# [4] is the spread aligned with the model's own error?
# --------------------------------------------------------------------------- #

def _spread(wps, mu=None):
    """Mean distance of a candidate to its pool centre, in avgL2 units."""
    mu = wps.mean(axis=0) if mu is None else mu
    return float(np.mean([avg_l2(c, mu) for c in wps]))


def _matched_null(pools, centers, rng, n_bins=5):
    """min-of-n after replacing each pool's scatter with a donor's, matched.

    The donor comes from the same spread quintile and its deviations are
    rescaled so its mean deviation equals the recipient's exactly, so scale,
    one-sidedness and temporal growth all survive and only this record's
    alignment with its own GT does not.

    ★ `centers` decides WHICH QUESTION is asked, and the two are not the same:

      pool mean   re-points the whole cloud, INCLUDING the cluster the model
                  actually decodes from. Part of any excess is then just "the
                  model's prediction beats a randomly rotated one", which is
                  true of any model that works at all.
    ★ THERE IS NO UNBIASED NULL CENTRED ON THE DECODE, and the attempt is worth
    recording so nobody rebuilds it. Candidate 0 is itself a draw, so c_j - c_0
    and c_0 - mu share the -c_0 term. Every way of scrambling orientation about
    c_0 -- transplanting another pool's deviations, or rigidly rotating this
    one's -- destroys that correlation and widens the cloud about the true
    centre (variance 3*sigma^2 against sigma^2), so it reports alignment that is
    not there. The pool mean is the only neutral centre, because sum(c_j - mu)
    is identically zero. What that costs is stated in block [4]: a pool whose
    candidates cluster away from mu cannot be adjudicated here at all.
    """
    spreads = np.array([_spread(p["wps"], c) for p, c in zip(pools, centers)])
    n = len(pools)
    order = np.argsort(spreads)
    binof = np.empty(n, dtype=int)
    for rank, idx in enumerate(order):
        binof[idx] = min(rank * n_bins // n, n_bins - 1)
    members = defaultdict(list)
    for i, b in enumerate(binof):
        members[b].append(i)

    out = np.empty(n)
    for i, p in enumerate(pools):
        m = len(p["wps"])
        k = None
        pool_idx = members[binof[i]]
        for _ in range(16):
            j = pool_idx[rng.randint(len(pool_idx))]
            if j != i and len(pools[j]["wps"]) >= m:
                k = j
                break
        if k is None or spreads[k] < 1e-9:
            out[i] = p["vals"].min()
            continue
        dev = (pools[k]["wps"] - centers[k])[:m] * (spreads[i] / spreads[k])
        out[i] = min(avg_l2(centers[i] + d, p["gt"]) for d in dev)
    return out


def _rotation_null(pools, centers, rng, n_rot=4):
    """min-of-n after rigidly rotating each pool's deviations about its centre.

    Every internal property of the cloud -- spread, one-sidedness, temporal
    growth, the relative placement of the candidates -- survives exactly, since
    one rotation is applied to all of it. Nothing is transplanted, so there is
    no scale mismatch. Only the cloud's ORIENTATION relative to this record's GT
    is destroyed. Valid ONLY about the pool mean (see _matched_null's header).
    """
    out = np.empty(len(pools))
    for i, (p, c) in enumerate(zip(pools, centers)):
        dev = p["wps"] - c
        vals = []
        for _ in range(n_rot):
            th = rng.uniform(0.0, 2.0 * np.pi)
            R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
            vals.append(min(avg_l2(c + d, p["gt"]) for d in dev @ R.T))
        out[i] = float(np.mean(vals))
    return out


def block_nulls(pools, n_boot=5000, rng_seed=0, n_bins=5):
    """Three nulls that keep the pool's geometry and destroy only its alignment.

    Each keeps this record's pool mean and replaces the DEVIATIONS around it, so
    "the model aimed at the wrong place" is held fixed and only "the candidates
    scattered toward the right place" is destroyed.

    reflection   c -> 2mu - c. Magnitudes and temporal shape are preserved
                 EXACTLY and only the sign flips.
                 ⚠ VALID ONLY IF THE DEVIATIONS ARE SYMMETRIC. A pool with a
                 one-sided tail (say a few candidates that overshoot far) has a
                 mean dragged into the tail, so mirroring a typical candidate
                 lands it further out than any real candidate ever goes, and the
                 null inflates on asymmetry alone. The asymmetry diagnostic
                 below says whether this pool is in that regime.
    swap         another record's deviations, verbatim.
                 ⚠ CONFOUNDED BY SCALE: a parked record inherits a highway
                 record's spread. Kept only to show the size of that confound.
    ★ matched    another record's deviations, drawn from the same spread
                 quintile and then rescaled so the donor's mean deviation equals
                 the recipient's exactly. Asymmetry, temporal growth and scale
                 all survive; only this record's alignment with its own GT does
                 not. THIS is the null the verdict uses.

    An observed minADE_C below the MATCHED null means the model's candidate
    spread leans toward where its own prediction is wrong -- record-specific
    information rather than a wholesale property of how it samples.
    """
    print(f"\n{'=' * 78}\n[4] IS THE SPREAD ALIGNED WITH THE ERROR? (nulls)\n{'=' * 78}")
    rng = np.random.RandomState(rng_seed)
    mus = [p["wps"].mean(axis=0) for p in pools]
    c0s = [p["wps"][0] for p in pools]

    obs = np.array([p["vals"].min() for p in pools])
    refl = np.array([min(avg_l2(2 * mu - c, p["gt"]) for c in p["wps"])
                     for p, mu in zip(pools, mus)])
    m_mu = _matched_null(pools, mus, rng, n_bins)
    r_mu = _rotation_null(pools, mus, rng)

    print(f"\n  {'pool':<38} {'minADE_C':>9} {'vs observed':>12} {'ci_sc':>21}")
    print(f"  {'observed':<38} {obs.mean():>9.4f}")
    for name, arr in (("reflected (c -> 2mu - c)", refl),
                      ("matched donor, about the pool mean", m_mu),
                      ("rotated, about the pool mean", r_mu)):
        lo, hi = _ci(pools, obs - arr, n_boot)
        print(f"  {name:<38} {arr.mean():>9.4f} "
              f"{obs.mean() - arr.mean():>+12.4f} [{lo:>+9.4f},{hi:>+9.4f}]")

    sf = float(np.mean([np.mean(p["wps"][:, HZ, 0] < mu[HZ, 0])
                        for p, mu in zip(pools, mus)]))
    e_mu = float(m_mu.mean() - obs.mean())
    e_rot = float(r_mu.mean() - obs.mean())
    excess = min(e_mu, e_rot)
    print(f"\n  asymmetry: {sf:.3f} of candidates fall SHORT of their own pool "
          f"centroid at 3s" + ("   (one-sided: the REFLECTED row is invalid here)"
                               if abs(sf - 0.5) > 0.05 else ""))
    print(f"  alignment excess   matched donor {e_mu:+.4f} m   "
          f"rotated {e_rot:+.4f} m   -> {excess:+.4f} m")
    print("  !! these three rows are DIAGNOSTIC ONLY -- see the displacement test")
    print("    below for why no null centred anywhere can settle this.")

    # ------------------------------------------------------------------ #
    # ★ the test that needs no null at all
    # ------------------------------------------------------------------ #
    loc, rnd_half, keep = [], [], []
    for p in pools:
        c0, w, v = p["wps"][0], p["wps"][1:], p["vals"][1:]
        if len(v) < 4:
            continue
        k = max(1, len(v) // 2)
        near = np.argsort([avg_l2(c, c0) for c in w])[:k]
        loc.append(float(v[near].min()))
        # a RANDOM half of the same size: identical order statistics, so the
        # only thing the comparison can see is whether nearness to the decode
        # is what makes a candidate good
        rnd_half.append(float(np.mean([v[rng.choice(len(v), k, replace=False)].min()
                                       for _ in range(8)])))
        keep.append(p)
    loc, rnd_half = np.asarray(loc), np.asarray(rnd_half)
    displacement = float(loc.mean() - rnd_half.mean())
    lo, hi = _ci(keep, loc - rnd_half, n_boot)

    print(f"\n  -- DISPLACEMENT: are the good candidates NEIGHBOURS of the "
          f"trajectory the\n     model already emits, or do they sit somewhere "
          f"else? (n={len(loc)})")
    print(f"     best of the half NEAREST the model's own pick   {loc.mean():.4f}")
    print(f"     best of a RANDOM half of the same size          {rnd_half.mean():.4f}")
    print(f"     difference {displacement:+.4f}  ci_sc [{lo:+.4f}, {hi:+.4f}]"
          f"   (equal sizes, so order statistics cancel)")
    print("     Why this and not the nulls above: the pool geometry of 'a better")
    print("     mode sits displaced from the decode' and of 'the good candidates")
    print("     ARE the decode's neighbours, with a tail elsewhere dragging the")
    print("     mean' is IDENTICAL. Every null centred on the pool mean scores")
    print("     both the same, and no null can be centred on candidate 0 (it is")
    print("     itself a draw, so scrambling about it widens the cloud). What")
    print("     separates them is whether staying near the decode COSTS anything,")
    print("     and that comparison needs no null at all.")
    displaced = displacement > T_PRIZE
    if displaced:
        print(f"     -> DISPLACED (> {T_PRIZE:.2f} m): staying near the model's own")
        print("        output costs real accuracy, so there is a systematically")
        print("        different trajectory to teach it. This is the premise")
        print("        best-of-n distillation needs, and it holds here.")
    else:
        print(f"     -> NOT DISPLACED (<= {T_PRIZE:.2f} m): the good candidates are")
        print("        the decode's own neighbours, so min-of-n is order")
        print("        statistics around the trajectory the model already emits.")
        print("        Distilling it teaches the model to hit its own")
        print("        neighbourhood more precisely -- which is exactly what")
        print("        training on GT already asked of it.")
    return {"alignment_excess": excess, "excess_matched": e_mu,
            "excess_rotated": e_rot, "displacement_m": displacement,
            "displaced": bool(displaced),
            "minADE_obs": float(obs.mean()), "minADE_reflected": float(refl.mean()),
            "minADE_matched": float(m_mu.mean()), "minADE_rotated": float(r_mu.mean()),
            "short_frac": sf}


# --------------------------------------------------------------------------- #
# [5] which way does the oracle move? (descriptive)
# --------------------------------------------------------------------------- #

def block_direction(pools, n_boot=5000):
    """Is the oracle systematically slower/faster, or just wherever GT is?

    A shift that is systematic in the ego frame (say "always decelerate") is
    something a policy can absorb wholesale. A shift that merely tracks
    GT - mu record by record is the model being wrong in a record-specific way,
    which is what training on GT was already asking it to fix.
    """
    print(f"\n{'=' * 78}\n[5] DIRECTION OF THE ORACLE SHIFT (descriptive)\n{'=' * 78}")
    d_lon, d_lat, g_lon, g_lat = [], [], [], []
    for p in pools:
        mu = p["wps"].mean(axis=0)
        o = p["wps"][int(np.argmin(p["vals"]))]
        d_lon.append(o[HZ, 0] - mu[HZ, 0])
        d_lat.append(o[HZ, 1] - mu[HZ, 1])
        g_lon.append(p["gt"][HZ, 0] - mu[HZ, 0])
        g_lat.append(p["gt"][HZ, 1] - mu[HZ, 1])
    out = {}
    print(f"\n  final waypoint, oracle minus pool centroid (m)")
    print(f"  {'axis':<16} {'mean':>9} {'ci_sc':>21} {'mean|.|':>9} "
          f"{'corr with GT-mu':>16}")
    for name, d, g in (("longitudinal", d_lon, g_lon), ("lateral", d_lat, g_lat)):
        d, g = np.asarray(d), np.asarray(g)
        lo, hi = _ci(pools, d, n_boot)
        c = float(np.corrcoef(d, g)[0, 1]) if d.std() > 1e-9 and g.std() > 1e-9 else float("nan")
        out[name] = (float(d.mean()), lo, hi, c)
        print(f"  {name:<16} {d.mean():>+9.4f} [{lo:>+9.4f},{hi:>+9.4f}] "
              f"{np.abs(d).mean():>9.4f} {c:>+16.3f}")
    print("\n  A mean whose CI excludes 0 is a wholesale bias the policy could")
    print("  absorb. A high correlation with GT-mu and a mean near 0 means the")
    print("  oracle is only ever 'the draw that leaned toward this record's GT'.")
    return out


# --------------------------------------------------------------------------- #
# verdict
# --------------------------------------------------------------------------- #

def verdict(dec, stab, nul):
    """-> 'GO' | 'WEAK' | 'STOP' | 'INCONCLUSIVE'.

    ★ The structure of the rule matters more than the thresholds. Stability is
    NECESSARY (labels that do not survive resampling are noise) but never
    SUFFICIENT, because the direction "toward this record's GT" survives
    resampling even in a world made entirely of isotropic noise -- and that
    direction is what training on GT was already asking the model to find. So a
    GO additionally requires a mechanism that ordinary GT training does not
    already cover: either the pool's spread is genuinely aligned with the error
    [4], or greedy is decoding the wrong point estimate [2].
    """
    print(f"\n{'=' * 78}\nVERDICT -- thresholds fixed in the source before the run\n{'=' * 78}")
    ceiling = stab["ceiling_gain"] if stab else float("nan")
    rec_edge = stab["recovery"] - stab["recovery_control"] if stab else float("nan")

    c_ceiling = bool(stab) and ceiling > T_CEILING
    c_transfer = bool(stab) and stab["recovery"] > T_RECOVERY and rec_edge > 0
    c_agree = bool(stab) and stab["agreement_ratio"] < T_AGREE
    c_align = bool(nul["displaced"])
    c_decode = dec["best_aggregate_gain"] > T_PRIZE

    for name, v, ok in (
            (f"in-sample ceiling      > {T_CEILING:.2f} m", ceiling, c_ceiling),
            (f"transfer recovery      > {T_RECOVERY:.0%} and beats its control",
             rec_edge, c_transfer),
            (f"oracle agreement ratio < {T_AGREE:.2f}",
             stab["agreement_ratio"] if stab else float("nan"), c_agree),
            (f"displacement           > {T_PRIZE:.2f} m",
             nul["displacement_m"], c_align),
            (f"decoding gap           > {T_PRIZE:.2f} m",
             dec["best_aggregate_gain"], c_decode)):
        print(f"  [{'PASS' if ok else 'FAIL'}]  {name:<48} {v:+.4f}")

    print()
    if not stab and not (c_align or c_decode):
        # stability is necessary, never sufficient, so a missing stability block
        # cannot rescue a missing mechanism: this is a real stop, not a gap in
        # the evidence, and it saves re-running seed arms to learn nothing.
        label = "STOP"
        print("  STOP (on a single seed -- and more seeds could not change it).")
        print("  Neither mechanism is present: the good candidates are the")
        print("  decode's own neighbours, and no aggregate of them beats greedy.")
        print("  Stability is a NECESSARY condition, so confirming it would still")
        print("  leave nothing for distillation to exploit that training on GT did")
        print("  not already offer. Do not spend GPU on the seed arms to check.")
    elif not stab:
        label = "INCONCLUSIVE"
        print("  INCONCLUSIVE -- a mechanism fired, but the stability block did")
        print("  not run, and a mechanism on one seed can be a lucky draw. Pass")
        print("  >= 2 seed arms of the same config to settle it.")
    elif not (c_ceiling and c_transfer):
        label = "STOP"
        print("  STOP. The oracle target does not survive resampling, or the")
        print("  ceiling it implies is inside the noise floor. best-of-n")
        print("  distillation would be fitting labels that are sampling luck, and")
        print("  sec.10 already prescribes terminating here. Write the paper.")
    elif not (c_align or c_decode):
        label = "WEAK"
        print("  WEAK. The target is stable, but the pool looks like symmetric")
        print("  noise and greedy already sits at its centre -- so distillation")
        print("  would be handing the model an attenuated copy of the GT target")
        print("  it was already trained on. Needs an argument beyond this data")
        print("  before 10-12 GPU-h; 'the ceiling is large' is not one, because")
        print("  the ceiling is in-sample by construction.")
    else:
        label = "GO"
        print("  GO -- with the cheap step first. Run (b) on a 2-3k train subset")
        print("  (~10-12 GPU-h, sec.10). If the decoding gap is what passed, try")
        print("  aggregating at decode time BEFORE any training: it costs nothing")
        print("  and captures the same mechanism.")
    print("\n  Either way this is a result: it is the first measurement of whether")
    print("  the 16.3% pool headroom is a target or an order statistic.")
    return label


# --------------------------------------------------------------------------- #
# self-test
# --------------------------------------------------------------------------- #

def _mk_pool(token, seed, scene, wps, gt, base):
    wps = np.asarray(wps, float)
    return {"token": token, "seed": str(seed), "scene": scene, "command": "straight",
            "wps": wps, "logged_vals": np.array([avg_l2(w, gt) for w in wps]),
            "vals": np.array([avg_l2(w, gt) for w in wps]), "gt": gt,
            "base": base, "got": None, "fused": False}


def _world(kind, n_rec=200, n_cand=8, n_seed=3, sigma=1.0, seed=0):
    """Two synthetic worlds with the SAME pool spread and the same greedy error.

    lottery  candidates are isotropic noise around mu; mu's error is a fixed
             per-record vector unrelated to the spread. min-of-n still beats mu
             by ~1.4 sigma -- the whole point is that this must NOT read as a
             learnable target.
    mode     the pool is bimodal: a fixed fraction of candidates sits on a
             second mode that is genuinely closer to GT, in the same place on
             every seed. Greedy sits on the wrong mode. This is what a real
             decoding mismatch looks like and it MUST be detected.
    ★ skew   the pool has a one-sided tail (half the candidates overshoot far
             along +x) that has NOTHING to do with where GT is. Nothing is
             learnable here, but the pool mean is dragged into the tail, so the
             REFLECTED null reports a large fake alignment excess. This world
             exists to prove the matched null removes that artefact -- it is the
             failure mode the real nuScenes pool turned out to be in.
    """
    rng = np.random.RandomState(seed)
    T = HZ + 1
    pools = []
    for r in range(n_rec):
        mu = np.stack([np.arange(1, T + 1) * 4.0, np.zeros(T)], 1)
        err = rng.randn(T, 2) * 1.5                 # where GT sits vs mu
        gt = mu + err
        # a second mode displaced TOWARD gt, identical on every seed
        shift = 0.6 * err / (np.linalg.norm(err) + 1e-9) * np.sqrt(T)
        # A one-sided overshoot that grows with the horizon, pointing along +x
        # regardless of where GT is: the shape of the sampler, not information.
        # It must be a MINORITY of the pool (2 of 8) -- an even 50/50 split is
        # symmetric about the pool mean and produces no artefact at all.
        over = np.stack([np.linspace(1.0, 8.0, T), np.zeros(T)], 1)
        # ...and the same tail in a RECORD-SPECIFIC direction, which is what
        # makes the pool-mean-centred null fail while the decode-centred one
        # stays honest (world "offset")
        u = rng.randn(2)
        odir = np.outer(np.linspace(1.0, 8.0, T), u / (np.linalg.norm(u) + 1e-9))
        for s in range(n_seed):
            cands = []
            for j in range(n_cand):
                if kind == "mode" and j % 2 == 1:
                    cands.append(mu + shift * 2.0 + rng.randn(T, 2) * sigma * 0.3)
                elif kind == "skew" and j % 4 == 3:
                    cands.append(mu + over + rng.randn(T, 2) * sigma * 0.3)
                elif kind == "offset" and j % 4 == 3:
                    cands.append(mu + odir + rng.randn(T, 2) * sigma * 0.3)
                else:
                    cands.append(mu + rng.randn(T, 2) * sigma)
            pools.append(_mk_pool(f"t{r}", 42 + s, f"sc{r // 4}", cands, gt,
                                  base=avg_l2(mu, gt)))
    # the arm's own pick is candidate 0 by convention (pool sorted best-first by
    # SCORE); the synthetic score is uninformative, so leave the order as drawn
    return pools


def _selftest():
    print("[selftest] metric")
    gt = np.zeros((6, 2))
    t = np.ones((6, 2))
    assert abs(avg_l2(t, gt) - float(np.sqrt(2.0))) < 1e-12
    t2 = np.zeros((6, 2))
    t2[5] = [3.0, 4.0]
    assert abs(avg_l2(t2, gt) - 5.0 / 6.0) < 1e-12, avg_l2(t2, gt)
    print("  ok  avg_l2 averages per-waypoint distance over wp 0..5 "
          "(ST-P3 avgL2@3s, not a flattened norm)")

    # the horizon must be the metric's, not the array's: a 7-point trajectory
    # must give the same number as its first 6 points
    long = np.vstack([t2, [[9.0, 9.0]]])
    assert abs(avg_l2(long, np.vstack([gt, [[0.0, 0.0]]])) - avg_l2(t2, gt)) < 1e-12
    print("  ok  extra waypoints beyond 3s are ignored")

    print("\n[selftest] LOTTERY world (isotropic noise; nothing to learn)")
    lot = _world("lottery", seed=1)
    d_l = block_decoding(lot, n_boot=400)
    n_l = block_nulls(lot, n_boot=400)
    s_l = block_stability(lot, n_boot=400)
    # the magnitude does not have to match the real 0.579 m, but the SIGN does:
    # if min-of-8 did not look good in a world with nothing to learn, this world
    # would not be reproducing the trap the script exists for
    assert d_l["oracle"] < d_l["greedy"] - 0.10, (
        d_l["greedy"], d_l["oracle"])
    assert d_l["best_aggregate_gain"] < T_PRIZE, d_l["best_aggregate_gain"]
    assert n_l["alignment_excess"] < T_PRIZE, n_l["alignment_excess"]
    print(f"\n  ok  lottery: oracle looks {d_l['greedy'] - d_l['oracle']:.3f} m "
          f"better yet the decoding gap ({d_l['best_aggregate_gain']:+.3f}) and "
          f"alignment excess ({n_l['alignment_excess']:+.3f}) are both null")

    print("\n[selftest] MODE world (real bimodality, same spread)")
    mod = _world("mode", seed=1)
    d_m = block_decoding(mod, n_boot=400)
    n_m = block_nulls(mod, n_boot=400)
    s_m = block_stability(mod, n_boot=400)
    assert d_m["best_aggregate_gain"] > T_PRIZE, d_m["best_aggregate_gain"]
    assert n_m["alignment_excess"] > T_PRIZE, n_m["alignment_excess"]
    print(f"\n  ok  mode: decoding gap {d_m['best_aggregate_gain']:+.3f}, "
          f"alignment excess {n_m['alignment_excess']:+.3f}, transfer edge "
          f"{s_m['transfer_edge']:+.3f} (lottery {s_l['transfer_edge']:+.3f})")

    print("\n[selftest] SKEW world (one-sided tail, still nothing to learn)")
    skw = _world("skew", seed=1)
    n_s = block_nulls(skw, n_boot=400)
    fake = n_s["minADE_reflected"] - n_s["minADE_obs"]
    assert fake > T_PRIZE, (
        "the skew world must fool the REFLECTED null -- if it stops doing so "
        f"this world no longer reproduces the artefact ({fake:+.4f})")
    assert n_s["alignment_excess"] < T_PRIZE, (
        "the matched null must remove the artefact the reflected null reports "
        f"({n_s['alignment_excess']:+.4f})")
    assert n_s["short_frac"] > 0.55, n_s["short_frac"]
    print(f"\n  ok  skew: reflected null reports a FAKE {fake:+.3f} m excess, the "
          f"matched null reports {n_s['alignment_excess']:+.3f}, and the "
          f"asymmetry diagnostic flags it ({n_s['short_frac']:.2f} short)")

    # and the fix must not cost sensitivity: the mode world's alignment is real
    # and asymmetric at the same time, and must survive matching
    assert n_m["alignment_excess"] > T_PRIZE, n_m["alignment_excess"]
    print(f"  ok  the matched null keeps the REAL signal "
          f"({n_m['alignment_excess']:+.3f} in the mode world)")

    print("\n[selftest] OFFSET world (tail in a record-specific direction)")
    off = _world("offset", seed=1)
    d_o = block_decoding(off, n_boot=400)
    n_o = block_nulls(off, n_boot=400)
    # ★ Nothing is learnable here -- the cluster IS where the model decodes and
    # min-of-8 over it is pure order statistics -- yet the pool mean is dragged
    # off that cluster, so every mean-centred null reports a large excess. The
    # script must not read that as evidence; it must recognise the shape and say
    # so. This world is the one the real nuScenes pool turned out to match.
    # every mean-centred null reports a large excess here, and it is meaningless
    assert n_o["excess_matched"] > T_PRIZE, n_o["excess_matched"]
    # ...but the oracle is a typical neighbour of the decode, which is the whole
    # difference between this world and the mode world, and needs no null
    assert not n_o["displaced"], n_o["displacement_m"]
    assert verdict(d_o, None, n_o) == "STOP"
    print(f"\n  ok  offset: the nulls report {n_o['excess_matched']:+.3f} m of "
          f"'alignment' and the\n      displacement test refuses it "
          f"({n_o['displacement_m']:+.3f} m) -> STOP")
    assert n_m["displaced"], n_m["displacement_m"]
    assert not n_l["displaced"], n_l["displacement_m"]
    print(f"  ok  displacement separates the two identical-looking geometries: "
          f"mode {n_m['displacement_m']:+.3f} vs offset "
          f"{n_o['displacement_m']:+.3f} vs lottery {n_l['displacement_m']:+.3f}")

    # ★ transfer alone must NOT separate the worlds. The lottery's oracle also
    # points at GT, so it also transfers; that is precisely why the verdict
    # requires a mechanism beyond stability. If this ever starts failing, the
    # rule in verdict() has become stricter than the evidence supports.
    print(f"\n  transfer recovery  lottery {s_l['recovery']:.1%}  "
          f"mode {s_m['recovery']:.1%}   (both can be high -- not a separator)")

    # the two worlds must not be separable by the in-sample ceiling: that is the
    # header's central claim, and if this assertion ever fails the ceiling is
    # measuring something real and the warnings should be rewritten
    print(f"\n  in-sample ceiling  lottery {s_l['ceiling_gain']:+.3f}  "
          f"mode {s_m['ceiling_gain']:+.3f}")
    assert s_l["ceiling_gain"] > T_CEILING, (
        "the lottery world must ALSO show a large consensus ceiling -- that is "
        "why the ceiling cannot bless the idea")
    print("  ok  the in-sample ceiling is large in BOTH worlds (it cannot "
          "distinguish them -- the header's central warning)")

    print("\n[selftest] verdict wiring")
    v_l = verdict(d_l, s_l, n_l)
    v_m = verdict(d_m, s_m, n_m)
    assert v_l in ("STOP", "WEAK"), f"the lottery world must not read as GO: {v_l}"
    assert v_m == "GO", f"the mode world must read as GO: {v_m}"
    # single-seed runs: a missing stability block cannot rescue a missing
    # mechanism (STOP), but it does hold back a mechanism that did fire
    assert verdict(d_m, None, n_m) == "INCONCLUSIVE"
    assert verdict(d_l, None, n_l) == "STOP"
    print(f"\n  ok  verdict: lottery -> {v_l}, mode -> {v_m}; on one seed "
          f"lottery -> STOP, mode -> INCONCLUSIVE")
    print("\nself-test PASS")


# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(
        "gate (a): is the pool headroom a stable target or order statistics? (0 GPU)")
    ap.add_argument("csv", nargs="*", help="per_sample.csv files (one per seed arm)")
    ap.add_argument("--records_json", default=None,
                    help="the evaluation set the run used -- REQUIRED, the "
                         "candidate errors are recomputed from its GT")
    ap.add_argument("--n_boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0, help="for the swap/control draws")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        _selftest()
        return
    if not a.csv:
        ap.error("give per_sample.csv files (or --selftest)")
    if not a.records_json:
        ap.error("--records_json is required: this script recomputes every "
                 "candidate's error from GT and refuses to run on the rounded "
                 "logged values alone")

    gt = load_gt(a.records_json)
    pools, skipped = load_pools(a.csv, gt)
    if not pools:
        print(f"[fatal] no usable pools. skipped: {skipped}")
        column_report(a.csv)
        print(f"\n  {len(gt)} records carry GT waypoints in {a.records_json}")
        sys.exit(
            "\n  If got_cand_wps is ABSENT the run predates 2026-08-03 and the "
            "column\n  cannot be rebuilt offline -- re-run that arm of the eval. "
            "Blocks [2] [4] [5]\n  work on a SINGLE csv, so one arm with the "
            "column is already worth running:\n  only the stability block [3] "
            "needs several seeds.")
    if skipped:
        print(f"[load] skipped {skipped}")
    pools = block_wiring(pools)
    dec = block_decoding(pools, a.n_boot)
    stab = block_stability(pools, a.n_boot, a.seed)
    nul = block_nulls(pools, a.n_boot, a.seed)
    block_direction(pools, a.n_boot)
    verdict(dec, stab, nul)


if __name__ == "__main__":
    main()
