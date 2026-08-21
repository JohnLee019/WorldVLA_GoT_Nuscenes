"""
Paired comparison BETWEEN ablation arms, clustered by scene.

Why this exists. Comparing arms by their summary means throws away the pairing:
every arm ran the same records, so arm-vs-arm is a paired design and should be
tested as one. The unpaired route is also hopeless at this scale -- on the
619-record turn split (84 scenes) the scene-clustered CI on a GoT-vs-baseline
diff is about +/-0.13 m while the arms differ by 0.02-0.08 m, so no amount of
extra arms will separate them. Pairing removes the between-record variance,
which is where nearly all of that width comes from.

What it tests, per arm against a reference arm:

  got_<key>                   did the arm change the OUTPUT
  got_minADE_C                did it change the POOL (the generator's ceiling)
  got_selection_gap_<key>     did it change the SELECTOR's loss

Those three are the decomposition GoT = minADE_C + gap, so running them together
answers the question the whole ablation is circling: when an arm improves the
pool, does the output follow, or does the gap absorb it?

Every test is Wilcoxon over per-scene mean differences plus a scene-cluster
bootstrap CI. Records inside a scene are consecutive keyframes of one manoeuvre;
treating them as independent is what made a six-scene run look significant.

    python compare_arms.py --ref results/abl3/ref_turn/per_sample.csv \
        results/abl3/temp_tight/per_sample.csv results/abl3/seg011/per_sample.csv
"""

import argparse
import csv
import os
from collections import defaultdict

from analyze_got_csv import _f, _mean, cluster_bootstrap_ci, wilcoxon_p


def load(path):
    """{sample_token: row} for rows where both predictors succeeded."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = {}
    for r in rows:
        if r.get("got_status") != "ok" or r.get("base_status") != "ok":
            continue
        out[r["sample_token"]] = r
    return out


def paired(ref, arm, metric):
    """(per-scene diffs, flat diffs) for arm - ref over the shared records."""
    scenes, flat = defaultdict(list), []
    for tok, ra in arm.items():
        rr = ref.get(tok)
        if rr is None:
            continue
        a, b = _f(ra, metric), _f(rr, metric)
        if a is None or b is None:
            continue
        d = a - b
        flat.append(d)
        scenes[ra.get("scene", "?")].append(d)
    return scenes, flat


def report(ref, ref_name, arms, key):
    metrics = [(f"got_{key}", "output"),
               ("got_minADE_C", "pool ceiling"),
               (f"got_selection_gap_{key}", "selector loss")]

    # A deterministic greedy baseline must be bit-identical across arms that
    # share k/beam. Where it is not, the arms are not strictly comparable --
    # wave 2's 'wide' differed (1.6720 vs 1.7156), consistent with the bf16
    # argmax tie-flips in handoff section 7.2 surfacing under a changed
    # allocation pattern.
    print(f"\nbaseline check (base_{key}, must match {ref_name}):")
    b_ref = _mean([v for v in (_f(r, f"base_{key}") for r in ref.values())
                   if v is not None])
    print(f"  {ref_name:<22} {b_ref:.4f}")
    for name, arm in arms:
        b = _mean([v for v in (_f(r, f"base_{key}") for r in arm.values())
                   if v is not None])
        flag = "" if abs(b - b_ref) < 1e-9 else "   <-- DIFFERS"
        print(f"  {name:<22} {b:.4f}{flag}")

    for metric, label in metrics:
        print(f"\n=== {metric}  ({label})   arm - {ref_name}, "
              f"negative = arm better ===")
        base_val = _mean([v for v in (_f(r, metric) for r in ref.values())
                          if v is not None])
        print(f"  {ref_name} = {base_val:.4f}")
        print(f"  {'arm':<20} {'n':>5} {'sc':>4} {'value':>8} {'diff':>9} "
              f"{'p_sc':>7} {'ci_sc':>19}")
        for name, arm in arms:
            scenes, flat = paired(ref, arm, metric)
            if not flat:
                print(f"  {name:<20} (no shared records with a value)")
                continue
            v = _mean([x for x in (_f(r, metric) for r in arm.values())
                       if x is not None])
            lo, hi = cluster_bootstrap_ci(scenes)
            p = wilcoxon_p([_mean(d) for d in scenes.values()])
            print(f"  {name:<20} {len(flat):>5} {len(scenes):>4} {v:>8.4f} "
                  f"{_mean(flat):>+9.4f} {p:>7.4f} [{lo:>+8.4f},{hi:>+8.4f}]")

    # The decomposition is an identity (output = ceiling + gap), so the two
    # right-hand diffs must sum to the left-hand one. Printing the absorbed
    # fraction makes the central claim checkable at a glance rather than
    # something the reader has to recompute.
    print(f"\n=== pool improvement absorbed by the selector ===")
    print("  an arm that improves the pool should improve the output by as much;")
    print("  'absorbed' is how much of it the widening selection gap ate instead.")
    print(f"  {'arm':<20} {'d_pool':>9} {'d_gap':>9} {'d_output':>9} {'absorbed':>9}")
    for name, arm in arms:
        _, d_pool = paired(ref, arm, "got_minADE_C")
        _, d_gap = paired(ref, arm, f"got_selection_gap_{key}")
        _, d_out = paired(ref, arm, f"got_{key}")
        if not d_pool or not d_gap:
            continue
        dp, dg, do = _mean(d_pool), _mean(d_gap), _mean(d_out)
        frac = f"{-dg / dp:>8.0%}" if abs(dp) > 1e-6 else "       -"
        print(f"  {name:<20} {dp:>+9.4f} {dg:>+9.4f} {do:>+9.4f} {frac}")
    print("  (absorbed > 100% means the output got WORSE despite a better pool)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csvs", nargs="+", help="per_sample.csv of the arms to compare")
    ap.add_argument("--ref", required=True, help="per_sample.csv of the reference arm")
    ap.add_argument("--key", default="avgL2@3s")
    ap.add_argument("--command", nargs="+", default=None,
                    help="restrict to these commands before pairing")
    args = ap.parse_args()

    def name_of(p):
        return os.path.basename(os.path.dirname(os.path.abspath(p)))

    def maybe_filter(d):
        if not args.command:
            return d
        keep = set(args.command)
        return {k: v for k, v in d.items() if v.get("command") in keep}

    ref = maybe_filter(load(args.ref))
    arms = [(name_of(p), maybe_filter(load(p))) for p in args.csvs
            if os.path.abspath(p) != os.path.abspath(args.ref)]
    print(f"reference: {name_of(args.ref)}  ({len(ref)} records"
          + (f", command in {sorted(set(args.command))}" if args.command else "") + ")")
    if not arms:
        print("no arms to compare against")
        return
    report(ref, name_of(args.ref), arms, args.key)


if __name__ == "__main__":
    main()
