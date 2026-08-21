#!/usr/bin/env python
"""Paired comparison of base checkpoints from eval_nuscenes per_sample.csv files.

Why not just read the two summary.json means: comparing summary means is an
unpaired comparison, and this project measured what that costs -- CI ~0.27
unpaired against ~0.08 paired (sec.9). The records are identical across runs
(same --records_json), so pairing on sample_token is free and the difference
between two checkpoints is usually smaller than the unpaired interval.

And the p-value must be scene-clustered. nuScenes records are consecutive
keyframes, so the ~4 records of one scene in the spread set are repeated looks
at one manoeuvre, not independent trials. Report p_sc / ci_sc only.

Decision rule this exists to serve (fixed before the numbers arrive):
  the scene-cluster CI half-width on this set is ~0.031 m, so a gain whose CI
  includes 0 is not a gain, and adopting a new base costs a full re-run of every
  number in PROJECT_HANDOFF sec.1 (headline 3 seeds ~7.5 GPU-h + fusion + WM
  ~6 GPU-h). Report a small real gain; adopt only a large one.

Usage
-----
    python compare_base_ckpts.py --ref results/base_ckpt/incumbent_cont2_ep1 \
        results/base_ckpt/cont3_ep0 results/base_ckpt/cont3_ep1
    python compare_base_ckpts.py --selftest
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict

from analyze_got_csv import _f, _mean, cluster_bootstrap_ci, wilcoxon_p

KEY = "avgL2@3s"
# scene-cluster CI half-width measured on the 600-record spread set
NOISE_FLOOR = 0.05


def load(run_dir, key=KEY):
    """-> {sample_token: (scene, value)} for the rows that evaluated cleanly."""
    path = os.path.join(run_dir, "per_sample.csv")
    if not os.path.exists(path):
        path2 = run_dir if run_dir.endswith(".csv") else None
        if path2 and os.path.exists(path2):
            path = path2
        else:
            sys.exit(f"[fatal] {path} does not exist. eval_nuscenes writes it when "
                     f"the run finishes, so a running eval has no partial file.")
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("status") != "ok":
                continue
            v = _f(r, key)
            if v is not None:
                out[r["sample_token"]] = (r.get("scene", "?"), v)
    return out


def compare(ref, arm, name, n_boot=10000):
    common = sorted(set(ref) & set(arm))
    if not common:
        print(f"  {name:<26} no overlapping sample_tokens with the reference")
        return
    scenes = defaultdict(list)
    diffs = []
    for t in common:
        d = arm[t][1] - ref[t][1]          # negative = the arm is BETTER
        diffs.append(d)
        scenes[ref[t][0]].append(d)
    lo, hi = cluster_bootstrap_ci(scenes, n_boot=n_boot)
    win = sum(1 for d in diffs if d < 0) / len(diffs)
    p_sc = wilcoxon_p([_mean(v) for v in scenes.values()])
    print(f"  {name:<26} {_mean([arm[t][1] for t in common]):>8.4f} "
          f"{_mean(diffs):>+9.4f} [{lo:>+8.4f},{hi:>+8.4f}] {p_sc:>8.4f} "
          f"{win:>6.3f} {len(common):>5} {len(scenes):>4}")
    return {"mean_diff": _mean(diffs), "ci": (lo, hi), "p_sc": p_sc,
            "win": win, "n": len(common), "n_scenes": len(scenes)}


def report(ref_dir, arm_dirs, key=KEY, n_boot=10000):
    ref = load(ref_dir, key)
    print(f"\n{'=' * 92}\nBASE CHECKPOINT COMPARISON (paired on sample_token, "
          f"key={key})\n{'=' * 92}")
    print(f"\n  reference: {ref_dir}   {len(ref)} records, "
          f"{len({s for s, _ in ref.values()})} scenes, mean "
          f"{_mean([v for _, v in ref.values()]):.4f}")
    print(f"\n  {'arm':<26} {'mean':>8} {'vs ref':>9} {'ci_sc':>19} "
          f"{'p_sc':>8} {'win':>6} {'n':>5} {'sc':>4}")
    results = {}
    for d in arm_dirs:
        results[d] = compare(ref, load(d, key), os.path.basename(d.rstrip("/\\")),
                             n_boot)

    print(f"\n{'-' * 92}\nVERDICT (negative diff = better; the floor is "
          f"{NOISE_FLOOR:.2f} m)\n{'-' * 92}")
    best, best_name = None, None
    for d, r in results.items():
        if r and (best is None or r["mean_diff"] < best["mean_diff"]):
            best, best_name = r, os.path.basename(d.rstrip("/\\"))
    if best is None:
        print("  nothing comparable.")
        return results
    lo, hi = best["ci"]
    if best["mean_diff"] < -NOISE_FLOOR and hi < 0:
        print(f"  {best_name} improves the base by {-best['mean_diff']:.4f} m "
              f"(CI excludes 0).")
        print("  -> a real gain. Adopting it invalidates every number in sec.1, so")
        print("     budget the re-run chain (headline 3 seeds ~7.5 GPU-h + fusion")
        print("     + WM main ~6 GPU-h) before switching, and re-run ALL arms in a")
        print("     directory rather than some of them (sec.9 stale-arm trap).")
    elif hi < 0:
        print(f"  {best_name} is better by {-best['mean_diff']:.4f} m -- significant "
              f"but under the {NOISE_FLOOR:.2f} m floor.")
        print("  -> report it, do not adopt it: the re-run chain costs more than")
        print("     the gain is worth, and mixing checkpoints across arms is the")
        print("     stale-arm trap that already contaminated one table (sec.9).")
    else:
        print(f"  No checkpoint beats the incumbent ({best_name} is the closest at "
              f"{best['mean_diff']:+.4f}, CI [{lo:+.4f},{hi:+.4f}]).")
        print("  -> the incumbent stays. This is a result: it says the base was")
        print("     converged, which until now was an assumption rather than a")
        print("     measurement.")
    return results


def _selftest():
    import random
    import tempfile

    def write(path, shift, seed):
        rng = random.Random(seed)
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "per_sample.csv"), "w", newline="",
                  encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["sample_token", "scene", "status", KEY])
            w.writeheader()
            for s in range(40):
                for i in range(4):
                    w.writerow({"sample_token": f"t{s:03d}{i}", "scene": f"sc{s}",
                                "status": "ok",
                                KEY: round(3.5 + rng.gauss(0, 0.5) + shift, 4)})

    d = tempfile.mkdtemp()
    a, b, c = (os.path.join(d, x) for x in ("ref", "better", "same"))
    # the SAME seed for all three: paired differences then isolate the shift,
    # which is the entire reason this compares per record instead of per mean
    write(a, 0.0, 7)
    write(b, -0.30, 7)
    write(c, 0.0, 7)
    res = report(a, [b, c], n_boot=2000)
    rb, rc = res[b], res[c]
    assert abs(rb["mean_diff"] + 0.30) < 1e-6, rb["mean_diff"]
    assert rb["ci"][1] < 0, rb["ci"]
    assert abs(rc["mean_diff"]) < 1e-9, rc["mean_diff"]
    print("\n  ok  a -0.30 m shift is recovered exactly and its CI excludes 0")
    print("  ok  an identical arm reads 0.0000")

    # unpaired would be far noisier: same shift, different noise draw
    write(c, -0.30, 99)
    res2 = report(a, [c], n_boot=2000)
    assert abs(res2[c]["mean_diff"] + 0.30) > 0.02, (
        "with an independent noise draw the difference must NOT come out exact -- "
        "otherwise the pairing is not doing anything and this test is vacuous")
    print("\n  ok  pairing is what makes the exact recovery possible "
          f"(independent draw gives {res2[c]['mean_diff']:+.4f} instead)")
    print("\nself-test PASS")


def main():
    p = argparse.ArgumentParser("paired base-checkpoint comparison (0 GPU)")
    p.add_argument("arms", nargs="*", help="eval_nuscenes output dirs")
    p.add_argument("--ref", help="the incumbent checkpoint's run dir")
    p.add_argument("--key", default=KEY)
    p.add_argument("--n_boot", type=int, default=10000)
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    if a.selftest:
        _selftest()
        return
    if not a.ref or not a.arms:
        p.error("need --ref REF_DIR and at least one arm dir (or --selftest)")
    report(a.ref, a.arms, a.key, a.n_boot)


if __name__ == "__main__":
    main()
