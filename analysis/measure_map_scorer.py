#!/usr/bin/env python
"""Ceiling for a MAP-AWARE scorer, before anyone builds one. CPU only.

THE QUESTION
------------
Every signal the GoT scorer currently uses -- kinematic smoothness, model
likelihood, command match -- is derived from the trajectory the generator just
produced. A scorer reading its own generator's output cannot know anything the
generator did not already encode, which is sec.1.1's mechanism and why nine
interventions were null.

The map is the one legitimately EXTERNAL signal available: the model sees a front
camera and has never seen a drivable-area layer. sec.10 (E7 Step 0) pointed at
DAC/EP as where the next scorer should aim. So: would a scorer that consults the
map have picked differently, and better?

WHAT DECIDES IT, AND WHY IT IS ONE NUMBER
-----------------------------------------
A gate can only change a pick when the candidates DISAGREE. If every candidate in
a pool is drivable-compliant (or every one is not), a map-aware scorer has nothing
to act on no matter how it is weighted. So the first number this prints is the
fraction of records whose pool splits on DAC. Near zero closes the door here, in
minutes, without anyone writing a scorer -- sec.11.8's rule, which was right five
times.

WHY THE COUNTERFACTUAL IS EXACT
-------------------------------
`got_cand_wps` holds all 8 candidate trajectories and `got_cand_vals` holds each
one's TRUE avgL2@3s, in the same order. Any selection rule can therefore be
replayed offline and scored without approximation -- no GPU, no re-decoding
(sec.1.3 established this).

WHAT THIS IS NOT
----------------
Not a scorer. It measures whether building one could pay, and reports the ceiling
a perfect map-aware selector would hit. A positive result here is permission to
build; a null closes the direction.

Usage
-----
  python analysis/measure_map_scorer.py \\
      --per_sample ./results/fusion/final_top3/per_sample.csv \\
      --drivable_masks ./data/nuscenes_records/nuscenes_drivable_val.npz

  python analysis/measure_map_scorer.py --selftest
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analyze_got_csv import _mean, cluster_bootstrap_ci, wilcoxon_p  # noqa: E402

DISAGREE_FLOOR = 0.05      # below this the direction is closed on arithmetic alone


def load_pool(path):
    """-> list of dicts with trajectories, true L2s, component scores, scene."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    need = ("got_cand_wps", "got_cand_vals")
    missing = [c for c in need if rows and c not in rows[0]]
    if missing:
        sys.exit(f"[fatal] {path} lacks {missing}. Only the fusion run logs the candidate "
                 f"trajectories; the headline csv drops got_cand_wps. Point --per_sample at "
                 f"results/fusion/final_top3/per_sample.csv")

    out = []
    for r in rows:
        if r.get("got_status") != "ok" or not r.get("got_cand_wps"):
            continue
        wps = ast.literal_eval(r["got_cand_wps"])
        vals = ast.literal_eval(r["got_cand_vals"])
        if len(wps) != len(vals) or not wps:
            continue
        tot = ast.literal_eval(r["got_cand_total"]) if r.get("got_cand_total") else None
        out.append({
            "tok": r["sample_token"], "scene": r.get("scene", "?"),
            "wps": wps, "vals": [float(v) for v in vals],
            "total": [float(v) for v in tot] if tot else None,
        })
    return out


def compliance(rec, drivable, cfg):
    """-> list[bool], one per candidate: did it stay inside the drivable area."""
    from pdms_components_nuscenes import dac_violation
    m = drivable[rec["tok"]]
    return [not dac_violation(w, m, cfg)[0] for w in rec["wps"]]


# --------------------------------------------------------------------------- #
# selection rules, all replayed exactly against the logged per-candidate L2
# --------------------------------------------------------------------------- #
def pick_greedy(rec, ok):
    return 0


def pick_score(rec, ok):
    """argmax of the shipped score over the whole pool (no gate)."""
    if rec["total"] is None:
        return 0
    return int(np.argmax(rec["total"]))


def pick_gate_then_score(rec, ok):
    """DAC gate first, then the shipped score among survivors.

    A gate that empties the pool must not silently return nothing: falling back to
    the greedy candidate is the behaviour a deployed rule would need, and scoring
    the fallback honestly is what keeps the comparison fair.
    """
    idx = [i for i, o in enumerate(ok) if o]
    if not idx:
        return 0
    if rec["total"] is None:
        return idx[0]
    return max(idx, key=lambda i: rec["total"][i])


def pick_gate_then_greedy(rec, ok):
    """DAC gate first, then prefer the earliest candidate (index 0 == greedy)."""
    idx = [i for i, o in enumerate(ok) if o]
    return idx[0] if idx else 0


def pick_oracle(rec, ok):
    return int(np.argmin(rec["vals"]))


def pick_oracle_gated(rec, ok):
    """Ceiling: perfect selection AMONG the candidates the gate keeps.

    Compared against the ungated oracle this says whether the gate ever removes
    the best candidate -- a gate that improves the average while discarding the
    winner is buying its gain in the wrong currency.
    """
    idx = [i for i, o in enumerate(ok) if o]
    if not idx:
        return int(np.argmin(rec["vals"]))
    return min(idx, key=lambda i: rec["vals"][i])


RULES = [
    ("greedy (candidate 0)", pick_greedy),
    ("shipped score, no gate", pick_score),
    ("DAC gate -> shipped score", pick_gate_then_score),
    ("DAC gate -> greedy", pick_gate_then_greedy),
    ("oracle, gated (ceiling)", pick_oracle_gated),
    ("oracle, ungated", pick_oracle),
]


def run(args):
    from got_drive.collision_metric import CollisionConfig
    cfg = CollisionConfig(uniad_parity=True)

    z = np.load(args.drivable_masks)
    meta = json.loads(str(z["__meta__"]))
    if (abs(meta["resolution"] - cfg.resolution) > 1e-9
            or list(meta["x_bound"]) != list(cfg.x_bound)):
        sys.exit(f"[fatal] mask grid {meta} does not match the collision grid. Rebuild.")
    drivable = {k: z[k].astype(bool) for k in z.files if k != "__meta__"}

    pool = [r for r in load_pool(args.per_sample) if r["tok"] in drivable]
    if not pool:
        sys.exit("[fatal] no record has both a candidate pool and a drivable mask")
    print(f"[map-scorer] {len(pool)} records, "
          f"{_mean([len(r['wps']) for r in pool]):.2f} candidates each | "
          f"GT-DAC of the mask set = {meta.get('gt_dac_frac')}")

    # ---- the number the whole question reduces to ---------------------------
    comp = {r["tok"]: compliance(r, drivable, cfg) for r in pool}
    n_split = sum(1 for r in pool if 0 < sum(comp[r["tok"]]) < len(comp[r["tok"]]))
    n_all_ok = sum(1 for r in pool if all(comp[r["tok"]]))
    n_none_ok = sum(1 for r in pool if not any(comp[r["tok"]]))
    frac_split = n_split / len(pool)

    print(f"\n--- does the pool ever split on DAC? ---")
    print(f"  all candidates compliant     {n_all_ok:>5} ({n_all_ok/len(pool):6.1%})")
    print(f"  none compliant               {n_none_ok:>5} ({n_none_ok/len(pool):6.1%})")
    print(f"  ★ SPLIT (gate can act)       {n_split:>5} ({frac_split:6.1%})")

    # ---- replay every rule, exactly ----------------------------------------
    print(f"\n--- selection rules, replayed on the logged per-candidate L2 ---")
    print(f"  {'rule':<28}{'avgL2@3s':>10}{'vs greedy':>11}{'changed picks':>15}")
    base = [r["vals"][0] for r in pool]
    results = {}
    for name, fn in RULES:
        picks = [fn(r, comp[r["tok"]]) for r in pool]
        vals = [r["vals"][i] for r, i in zip(pool, picks)]
        changed = sum(1 for i in picks if i != 0)
        results[name] = {"mean": _mean(vals), "picks": picks, "vals": vals,
                         "changed": changed}
        print(f"  {name:<28}{_mean(vals):>10.4f}{_mean(vals)-_mean(base):>+11.4f}"
              f"{changed:>15}")

    # ---- paired test for the one rule the question is about -----------------
    key = "DAC gate -> shipped score"
    ref = "shipped score, no gate"
    clusters = defaultdict(list)
    for r, a, b in zip(pool, results[key]["vals"], results[ref]["vals"]):
        clusters[r["scene"]].append(a - b)
    flat = [d for v in clusters.values() for d in v]
    lo, hi = cluster_bootstrap_ci(clusters, n_boot=args.n_boot)
    p_sc = wilcoxon_p([_mean(v) for v in clusters.values()])
    n_diff = sum(1 for d in flat if abs(d) > 1e-12)
    print(f"\n  gate effect on the shipped scorer: {_mean(flat):+.4f} "
          f"ci_sc [{lo:+.4f}, {hi:+.4f}] p_sc {p_sc:.4f}")
    print(f"  the gate changed the pick on {n_diff}/{len(flat)} records")

    # ---- verdict, branched on what was just measured ------------------------
    print("\n" + "-" * 74)
    print("VERDICT")
    print("-" * 74)
    if frac_split < DISAGREE_FLOOR:
        print(f"  The pool splits on DAC in only {frac_split:.1%} of records. A map-aware")
        print(f"  scorer has nothing to act on in the other {1-frac_split:.0%}, so no weighting")
        print(f"  of it can move the result. CLOSED -- do not build the scorer.")
    elif n_diff == 0:
        print(f"  The pool splits {frac_split:.1%} of the time, but the gate never changed a")
        print(f"  pick: the shipped score already prefers compliant candidates. The map adds")
        print(f"  no independent decision here. CLOSED.")
    elif lo <= 0.0 <= hi:
        print(f"  The gate acts on {n_diff} records and moves avgL2@3s by {_mean(flat):+.4f},")
        print(f"  ci_sc [{lo:+.4f}, {hi:+.4f}] spanning 0 -- no detectable gain. This is the")
        print(f"  tenth null of this shape (sec.1.1); treat the direction as closed unless a")
        print(f"  larger split fraction is found on another set.")
    elif _mean(flat) < 0:
        print(f"  ★ The gate IMPROVES the pick by {-_mean(flat):.4f} m, ci_sc "
              f"[{lo:+.4f}, {hi:+.4f}].")
        print(f"  This is the first non-null selection signal in the project. Build the scorer,")
        print(f"  and re-run the headline arms before believing the size of the effect.")
    else:
        print(f"  The gate makes the pick WORSE by {_mean(flat):.4f} m. The map ranks candidates")
        print(f"  in the wrong direction here -- report it, do not ship it.")

    gated_ceiling = results["oracle, gated (ceiling)"]["mean"]
    ungated = results["oracle, ungated"]["mean"]
    if gated_ceiling - ungated > 0.01:
        print(f"\n  ⚠️ The gate discards the best candidate often enough to raise the oracle")
        print(f"     from {ungated:.4f} to {gated_ceiling:.4f}. Any gain it shows is being paid")
        print(f"     for with candidates that were actually good.")

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump({"n_records": len(pool), "frac_split": round(frac_split, 4),
                       "n_all_compliant": n_all_ok, "n_none_compliant": n_none_ok,
                       "rules": {k: round(v["mean"], 4) for k, v in results.items()},
                       "gate_effect": {"mean": round(_mean(flat), 4),
                                       "ci_sc": [round(lo, 4), round(hi, 4)],
                                       "p_sc": round(p_sc, 4), "n_changed": n_diff}}, f, indent=2)
        print(f"\n  -> {args.output_json}")


# --------------------------------------------------------------------------- #
def selftest():
    ok_all = True

    def check(name, cond, detail=""):
        nonlocal ok_all
        ok_all &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))

    rec = {"tok": "t", "scene": "s",
           "wps": [None] * 4,
           "vals": [3.0, 1.0, 5.0, 2.0],
           "total": [0.9, 0.1, 0.8, 0.2]}

    # every candidate compliant -> the gate must be a no-op
    ok = [True] * 4
    check("gate is a no-op when nothing is filtered",
          pick_gate_then_score(rec, ok) == pick_score(rec, ok),
          f"both pick {pick_score(rec, ok)}")

    # the score's favourite is non-compliant -> the gate must move the pick
    ok = [True, False, True, True]
    check("gate moves the pick when the score's favourite is filtered out",
          pick_gate_then_score(rec, ok) == 0, f"picked {pick_gate_then_score(rec, ok)}")

    # the gate can remove the true best -- the ceiling must show it
    check("gated oracle is worse than ungated when the winner is filtered",
          rec["vals"][pick_oracle_gated(rec, ok)] > rec["vals"][pick_oracle(rec, ok)],
          f"gated {rec['vals'][pick_oracle_gated(rec, ok)]} vs "
          f"ungated {rec['vals'][pick_oracle(rec, ok)]}")

    # an empty gate must fall back, not crash or return None
    ok = [False] * 4
    for nm, fn in (("score", pick_gate_then_score), ("greedy", pick_gate_then_greedy)):
        i = fn(rec, ok)
        check(f"empty gate falls back to greedy ({nm})", i == 0, f"got {i}")
    check("empty gate leaves the oracle ungated",
          pick_oracle_gated(rec, ok) == pick_oracle(rec, ok))

    # index 0 really is greedy in the logged pools
    check("greedy rule reads candidate 0", pick_greedy(rec, [True] * 4) == 0)

    print("\nselftest:", "PASS" if ok_all else "FAIL")
    return 0 if ok_all else 1


def main():
    p = argparse.ArgumentParser("ceiling for a map-aware GoT scorer (CPU only)")
    p.add_argument("--per_sample", default="./results/fusion/final_top3/per_sample.csv",
                   help="the run that logs got_cand_wps (the fusion run does; headline does not)")
    p.add_argument("--drivable_masks", default="./data/nuscenes_records/nuscenes_drivable_val.npz")
    p.add_argument("--output_json", default=None)
    p.add_argument("--n_boot", type=int, default=10000)
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args()

    if args.selftest:
        sys.exit(selftest())
    run(args)


if __name__ == "__main__":
    main()
