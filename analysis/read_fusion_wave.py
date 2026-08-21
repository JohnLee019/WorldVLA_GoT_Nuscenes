#!/usr/bin/env python
"""Single readout for the fusion wave, in the pre-registered order.

Why this exists as a file instead of a heredoc. The fusion wave needs four
checks run in a fixed order, and two of them were established only after the
run started (2026-08-03), so they are not in run_fusion.sh:

  1. WIRING       every arm's greedy free-run must be 3.5557 (deterministic).
  2. FEASIBILITY  BEFORE any L2. run_fusion.sh demanded n_infeasible_output == 0,
                  which is too strict: the non-fused arm's own whole-trajectory
                  rate is 15/600 = 2.5%. The audit `_feasible(merged)` looks at
                  all 6 waypoints while the veto inside the pipeline only ever
                  saw one 2-point segment plus its origin, so junction
                  acceleration is invisible to the veto and visible here. The
                  gate is therefore "not elevated above the reference arm",
                  tested as a two-proportion z-test, not "exactly zero".
  3. RULE         advance to seeds 43/44 only if |d_output| >= 0.06 at seed 42.
                  Below that it is under the scene-clustered CI half-width
                  (~0.031) and more seeds cannot resolve it. temp_tight and
                  lik_full both died this way in wave 5.
  4. RNG PAIRING  measured, not assumed: seg_all and seg_top2 emit identical
                  last-segment steps in 600/600 records. Every fusion arm makes
                  12 generation calls with the seed derived from (seed, record
                  index), so they enter segment 3 with the same RNG state, and
                  the action space is discrete, so the same uniform draw picks
                  the same token unless the prefix shifts the distribution far
                  enough to flip it. At this perturbation size it never does.
                  That is what makes seg_top1 a CONTROLLED EXPERIMENT rather
                  than a plumbing check: the continuation is held literally
                  constant and only the prefix changes.

Reads finished runs only; per_sample.csv and summary.json are written in one
shot when the loop ends, so a missing file means "still running", never
"partial". No torch, no GPU. Run from the repo root.

    python read_fusion_wave.py --runs results/fusion/*/ \
        --ref results/headline/ref/per_sample.csv
    python read_fusion_wave.py --selftest
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
import ast
import csv
import json
import math
import os
import sys
from collections import Counter

import numpy as np

# Reuse the eval's own predicate rather than reimplementing it: a private copy
# that drifted from got_drive/scoring_driving.py would silently answer a
# different question than the number in summary.json.
try:
    from got_drive.scoring_driving import (DEFAULT_DT, FEAS_A_MAX,
                                           FEAS_LAT_STEP_MAX, FEAS_V_MAX,
                                           _feasible, _finite_diff,
                                           _with_origin)
except ImportError as e:                                    # pragma: no cover
    sys.exit(f"[fatal] run this from the repo root, not from got_drive/: {e}")

GREEDY_EXPECTED = 3.5557        # deterministic; fusion cannot touch it
D_OUTPUT_GATE = 0.06            # pre-registered, see module docstring
SEGMENT_LEN = 2                 # 3 segments x 2 waypoints -> junctions at 2 and 4


# ---------------------------------------------------------------- csv loading

def _cell(row, col):
    """Parse a list-valued CSV cell, or None when absent/blank/ragged."""
    v = row.get(col) or ""
    if not v:
        return None
    try:
        return ast.literal_eval(v)
    except (ValueError, SyntaxError):
        return None


def load_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


# ------------------------------------------------------------- feasibility

def violations(traj):
    """[(limit, index)] for every hard limit the trajectory breaks.

    Index is into the origin-prepended trajectory, so with segment_len 2 the
    segment junctions sit at 2 and 4. A plumbing fault in the frame composition
    would hit both junctions equally (same code path); damage that grows with
    the number of accumulated fusions would not.
    """
    t = _with_origin(np.asarray(traj, dtype=float), True)
    out = []
    if t.shape[0] < 2:
        return out
    v = _finite_diff(t, DEFAULT_DT)
    for i, s in enumerate(np.linalg.norm(v, axis=1)):
        if s > FEAS_V_MAX:
            out.append(("speed", i))
    for i, d in enumerate(np.abs(np.diff(t[:, 1]))):
        if d > FEAS_LAT_STEP_MAX:
            out.append(("lat_step", i))
    if t.shape[0] >= 3:
        a = _finite_diff(v, DEFAULT_DT)
        for i, s in enumerate(np.linalg.norm(a, axis=1)):
            if s > FEAS_A_MAX:
                out.append(("accel", i))
    return out


def infeasible_rate(rows, col):
    """(n_bad, n_total) over a trajectory column, skipping unparsable cells."""
    bad = tot = 0
    for r in rows:
        obj = _cell(r, col)
        if obj is None:
            continue
        items = obj if (obj and isinstance(obj[0][0], (list, tuple))) else [obj]
        for t in items:
            tot += 1
            bad += 0 if _feasible(np.asarray(t, dtype=float)) else 1
    return bad, tot


def two_proportion_z(k1, n1, k2, n2):
    """z for H0: p1 == p2. Returns 0.0 when either sample is empty."""
    if n1 == 0 or n2 == 0:
        return 0.0
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    return 0.0 if se == 0 else (p1 - p2) / se


# --------------------------------------------------------------- rng pairing

def last_steps(rows):
    """Per record, the sorted multiset of each candidate's final step.

    Translation-invariant, so it isolates what the generator emitted for the
    last segment from where the fused prefix put it.
    """
    out = []
    for r in rows:
        pool = _cell(r, "got_cand_wps")
        if not pool:
            out.append(None)
            continue
        out.append(sorted(tuple(np.round(np.asarray(t, float)[-1]
                                         - np.asarray(t, float)[-2], 4))
                          for t in pool))
    return out


# ------------------------------------------------------------------- report

def report(run_dirs, ref_csv):
    ref_bad = ref_tot = 0
    if ref_csv and os.path.exists(ref_csv):
        ref_bad, ref_tot = infeasible_rate(load_csv(ref_csv), "got_pred")
        print(f"reference arm (no fusion): infeasible outputs "
              f"{ref_bad}/{ref_tot} = {ref_bad / max(ref_tot, 1):.1%}\n")
    else:
        print("reference arm: not given; feasibility judged against 0 instead\n")

    arms = {}
    for d in run_dirs:
        name = os.path.basename(os.path.normpath(d))
        sp, cp = os.path.join(d, "summary.json"), os.path.join(d, "per_sample.csv")
        if not (os.path.exists(sp) and os.path.exists(cp)):
            print(f"{name:22s} STILL RUNNING (outputs are written in one shot "
                  f"at the end; check the .log for 'plans ok')")
            continue
        s = json.load(open(sp))
        rows = load_csv(cp)
        got = list(s["got_per_seed"].values())[0]
        base = s.get("baseline_free_run", {}).get("avgL2@3s", float("nan"))
        fus = s.get("fusion", {})
        n_inf = fus.get("n_infeasible_output")
        n_rec = got.get("n_evaluated", len(rows))
        d_out = got["avgL2@3s"] - base

        wiring = "ok" if abs(base - GREEDY_EXPECTED) < 5e-4 else "<<< DIFFERS"
        if n_inf is None:
            feas = "n/a (not a fusion arm)"
        elif ref_tot:
            z = two_proportion_z(n_inf, n_rec, ref_bad, ref_tot)
            feas = (f"{n_inf}/{n_rec} = {n_inf / max(n_rec, 1):.1%}  z={z:+.1f}  "
                    + ("ELEVATED -> L2 not readable" if z > 2 else "at reference rate -> ok"))
        else:
            feas = f"{n_inf}/{n_rec}"

        print(f"{name}")
        print(f"  1 wiring       greedy {base:.4f}  [{wiring}]")
        print(f"  2 feasibility  {feas}")
        print(f"  3 output       GoT {got['avgL2@3s']:.4f}  d_output {d_out:+.4f}  "
              + ("[ADVANCE to 43/44]" if abs(d_out) >= D_OUTPUT_GATE
                 else "[below resolution -> STOP]"))
        print(f"    cost         {got.get('cost', {}).get('forward_calls_per_record')} calls  "
              f"{got.get('cost', {}).get('sec_per_record')} s/rec")

        kind, where = Counter(), Counter()
        for r in rows:
            pool = _cell(r, "got_cand_wps")
            for t in (pool or []):
                for k, i in violations(t):
                    kind[k] += 1
                    where[i] += 1
        if kind:
            junc = [i for i in where if i and i % SEGMENT_LEN == 0]
            print(f"  4 candidates   limits {dict(kind)}")
            print(f"                 by index {dict(sorted(where.items()))}   "
                  f"(junctions at {sorted(junc)})")
        arms[name] = last_steps(rows)
        print()

    names = list(arms)
    if len(names) > 1:
        print("rng pairing (identical last-segment steps => continuation held constant)")
        a = names[0]
        for b in names[1:]:
            pairs = [(x, y) for x, y in zip(arms[a], arms[b])
                     if x is not None and y is not None]
            same = sum(x == y for x, y in pairs)
            print(f"  {a} vs {b}: {same}/{len(pairs)}")

    print("\nreminders: minADE_C of a `segment` arm is the last segment's pool only "
          "-- never in one column with `final` or with results/headline/ref. "
          "Mode averaging hides in the aggregate; stratify by command "
          "(analyze_got_csv.py --command left right).")


# ----------------------------------------------------------------- self-test

def selftest():
    ok = True

    straight = np.array([[2.0, 0.0], [4.0, 0.0], [6.0, 0.0],
                         [8.0, 0.0], [10.0, 0.0], [12.0, 0.0]])
    assert _feasible(straight), "a constant-speed straight line must be feasible"
    print(f"  ok  clean trajectory: {len(violations(straight))} violations")
    ok &= not violations(straight)

    jump = straight.copy()
    jump[4:, 1] += 9.0                      # lateral fling entering segment 3
    v = violations(jump)
    idx = {i for _, i in v}
    print(f"  ok  junction fault at waypoint 4 -> indices {sorted(idx)} "
          f"limits {sorted({k for k, _ in v})}")
    ok &= bool(v) and max(idx) >= 4 and not _feasible(jump)

    # a violation planted inside segment 1 must NOT be reported at a junction
    early = straight.copy()
    early[0, 1] += 9.0
    ei = {i for _, i in violations(early)}
    print(f"  ok  early fault -> indices {sorted(ei)} (must include 0 or 1)")
    ok &= bool(ei & {0, 1})

    z = two_proportion_z(251, 600, 15, 600)
    z0 = two_proportion_z(17, 600, 15, 600)
    print(f"  ok  z(251/600 vs 15/600)={z:.1f}  z(17/600 vs 15/600)={z0:.1f}")
    ok &= z > 2 and abs(z0) < 2

    # convexity: the mean of two feasible trajectories is feasible, because
    # every limit bounds a norm of a linear functional of the waypoints. This
    # is why 2-way fusion cannot itself manufacture an infeasible output.
    other = straight * np.array([1.0, 1.0]) + np.array([0.0, 0.5])
    assert _feasible(other)
    ok &= _feasible((straight + other) / 2.0)
    print("  ok  convexity: mean of two feasible trajectories is feasible")

    assert _cell({"got_cand_wps": ""}, "got_cand_wps") is None
    assert _cell({"got_cand_wps": "[[[1,2]]]"}, "got_cand_wps") == [[[1, 2]]]
    print("  ok  cell parsing: blank and nested handled")

    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="*", default=[], help="run directories")
    p.add_argument("--ref", default="results/headline/ref/per_sample.csv",
                   help="non-fused arm, for the feasibility reference rate")
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    if a.selftest:
        return selftest()
    if not a.runs:
        p.error("give --runs, or --selftest")
    report(a.runs, a.ref)
    return 0


if __name__ == "__main__":
    sys.exit(main())
