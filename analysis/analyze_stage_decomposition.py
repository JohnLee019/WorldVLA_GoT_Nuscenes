"""Where in the deliberation does the loss appear? (CPU, no GPU, no torch)

sec.1 reports one number for the whole pipeline: GoT is +0.0397 worse than greedy.
That number cannot say WHICH of the three thoughts went wrong, and "show the
figures at every stage and explain what broke" is the question this answers.

The reconstruction needs no new inference. `got_cand_wps` stores the FULL 6-point
trajectory of every surviving candidate, and beam search makes candidates that
survived the same stage share a prefix -- so grouping the candidates by their
prefix at each segment boundary recovers the branching structure exactly. For
each stage we then report what was on offer, what the scorer took, and what the
best option would have been.

!! THE ONE NUMBER TO READ IS `hit-best` AGAINST `chance`, NOT AGAINST 100%.
A stage with 2 live options gives a coin flip 50% for free. Reporting 57% without
that column reads like a working scorer; against chance it is +7 points. The tool
always prints chance = mean(1 / n_live) beside it and refuses to omit it.

!! THE ATTRIBUTION IS EXACT, NOT APPORTIONED. `avgL2@3s` is the mean over 6
waypoints and each segment holds exactly 2, so the per-segment mean errors average
to the headline. The shares therefore sum to 100% by construction rather than by
a modelling choice -- and the total doubles as a wiring check against sec.7.2.

Usage
-----
  python analysis/analyze_stage_decomposition.py --selftest
  python analysis/analyze_stage_decomposition.py \
      --csv results/headline/ref_basepred/per_sample.csv \
      --records_json data/nuscenes_records/nuscenes_val_scenespread.json
"""
import argparse
import ast
import csv as csvmod
import json
import sys

import numpy as np

N_BOOT = 4000


def prefix_groups(cands, end):
    """Candidates that share the trajectory up to waypoint `end`, in score order.

    `got_cand_wps` is stored SCORE-DESCENDING (session 19 correction), so the
    group containing index 0 is the scorer's own pick -- never greedy.
    """
    groups = {}
    for i, w in enumerate(cands):
        groups.setdefault(tuple(np.round(w[:end].ravel(), 4)), []).append(i)
    return groups


def stage_table(cands, gt, base, n_segments, segment_len):
    """Per-stage record: what was live, what was taken, what was best."""
    out = []
    for s in range(n_segments):
        beg, end = s * segment_len, (s + 1) * segment_len
        groups = prefix_groups(cands, end)
        keys = list(groups)
        pick_key = tuple(np.round(cands[0][:end].ravel(), 4))

        def cum(i):   # cumulative prefix error, waypoints 0..end
            return float(np.linalg.norm(cands[i][:end] - gt[:end], axis=1).mean())

        def seg(i):   # this segment's own two waypoints only
            return float(np.linalg.norm(cands[i][beg:end] - gt[beg:end], axis=1).mean())

        cum_e = {k: cum(v[0]) for k, v in groups.items()}
        seg_e = {k: seg(v[0]) for k, v in groups.items()}
        out.append({
            "n_live": len(keys),
            "cum_pick": cum_e[pick_key], "cum_best": min(cum_e.values()),
            "cum_greedy": float(np.linalg.norm(base[:end] - gt[:end], axis=1).mean()),
            "seg_pick": seg_e[pick_key], "seg_best": min(seg_e.values()),
            "seg_greedy": float(np.linalg.norm(base[beg:end] - gt[beg:end], axis=1).mean()),
            "hit_best": cum_e[pick_key] <= min(cum_e.values()) + 1e-9,
            "chance": 1.0 / len(keys),
        })
    return out


def scene_ci(x, scenes, rng, n_boot=N_BOOT):
    """Scene-cluster bootstrap (sec.9): records in one scene are not independent."""
    uq = np.unique(scenes)
    idx = {s: np.where(scenes == s)[0] for s in uq}
    boots = np.empty(n_boot)
    for b in range(n_boot):
        pick = np.concatenate([idx[s] for s in rng.choice(uq, len(uq))])
        boots[b] = x[pick].mean()
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def analyse(rows, gt_map, n_segments, segment_len):
    per, scenes, seg_delta = [], [], []
    for r in rows:
        tok = r["sample_token"]
        if tok not in gt_map:
            continue
        n_wp = n_segments * segment_len
        cands = np.asarray(ast.literal_eval(r["got_cand_wps"]), float)
        base = np.asarray(ast.literal_eval(r["base_pred"]), float)[:n_wp]
        gt = gt_map[tok][:n_wp]
        if cands.shape[1] < n_wp or base.shape[0] < n_wp or gt.shape[0] < n_wp:
            continue
        st = stage_table(cands, gt, base, n_segments, segment_len)
        per.append(st)
        scenes.append(r["scene"])
        seg_delta.append([st[s]["seg_pick"] - st[s]["seg_greedy"] for s in range(n_segments)])
    return per, np.asarray(scenes), np.asarray(seg_delta)


def report(per, scenes, seg_delta, n_segments, seed=42):
    rng = np.random.default_rng(seed)
    n = len(per)
    g = lambda s, k: np.array([p[s][k] for p in per], float)
    print(f"[n] {n} records / {len(np.unique(scenes))} scenes\n")

    print("STAGE 1 -- what was on offer, and did the scorer take the best of it?")
    print(f"{'stage':>6} {'live opts':>10} {'had a choice':>13} {'hit-best':>9} "
          f"{'chance':>8} {'lift':>7}")
    print("-" * 60)
    for s in range(n_segments):
        live, hit, ch = g(s, "n_live"), g(s, "hit_best"), g(s, "chance")
        multi = live > 1
        hr = hit[multi].mean() if multi.any() else float("nan")
        cr = ch[multi].mean() if multi.any() else float("nan")
        print(f"{s+1:>6} {live.mean():>10.2f} {multi.mean()*100:>12.0f}% "
              f"{hr*100:>8.1f}% {cr*100:>7.1f}% {(hr-cr)*100:>+6.1f}p")

    print("\nSTAGE 2 -- error at each stage (mean over records, metres)")
    print(f"{'stage':>6} | {'cumulative prefix':^28} | {'this segment only':^28}")
    print(f"{'':>6} | {'GoT':>8} {'greedy':>8} {'best':>8} | {'GoT':>8} {'greedy':>8} {'best':>8}")
    print("-" * 74)
    for s in range(n_segments):
        print(f"{s+1:>6} | {g(s,'cum_pick').mean():>8.4f} {g(s,'cum_greedy').mean():>8.4f} "
              f"{g(s,'cum_best').mean():>8.4f} | {g(s,'seg_pick').mean():>8.4f} "
              f"{g(s,'seg_greedy').mean():>8.4f} {g(s,'seg_best').mean():>8.4f}")

    print("\nSTAGE 3 -- where the loss against greedy is made (exact attribution)")
    total = seg_delta.mean(1).mean()
    print(f"{'stage':>6} {'GoT-greedy':>12} {'ci_sc':>24} {'share':>8}")
    print("-" * 54)
    for s in range(n_segments):
        lo, hi = scene_ci(seg_delta[:, s], scenes, rng)
        share = (seg_delta[:, s].mean() / n_segments) / total * 100 if total else float("nan")
        print(f"{s+1:>6} {seg_delta[:,s].mean():>+12.4f} "
              f"{f'[{lo:+.4f}, {hi:+.4f}]':>24} {share:>7.1f}%")
    lo, hi = scene_ci(seg_delta.mean(1), scenes, rng)
    print("-" * 54)
    print(f"{'total':>6} {total:>+12.4f} {f'[{lo:+.4f}, {hi:+.4f}]':>24} {100.0:>7.1f}%")
    print("\n  The shares sum to 100% by construction (each segment holds an equal")
    print("  share of the waypoints avgL2@3s averages over), so `total` is also a")
    print("  wiring check: it must equal this run's GoT-minus-greedy from sec.7.2.")
    return total


# ---------------------------------------------------------------------------
# Self-test. sec.11.7: every check is paired with a world where the right answer
# is different, so a tool that always says the same thing cannot pass both.
# ---------------------------------------------------------------------------
def _world(kind, n_rec=120, n_seg=3, seg_len=2, seed=0, greedy="is_cand0"):
    """Build (rows, gt_map) whose correct answer is known by construction.

    `greedy="is_cand0"` makes base_pred identical to the scorer's pick, so the gap
    must come out exactly 0. `greedy="separate"` gives greedy its own trajectory,
    so the gap is non-zero and the attribution has something real to split.
    """
    rng = np.random.default_rng(seed)
    rows, gt_map = [], {}
    n_wp = n_seg * seg_len
    for i in range(n_rec):
        tok = f"t{i:04d}"
        gt = np.cumsum(rng.normal(4, 1, (n_wp, 2)) * [1, 0.1], 0)
        cands = np.stack([gt + rng.normal(0, 1.5, (n_wp, 2)) for _ in range(8)])
        if kind == "no_branching":
            cands[:] = cands[0]                       # every candidate identical
        errs = np.linalg.norm(cands - gt, axis=2).mean(1)
        if kind == "perfect_scorer":
            cands = cands[np.argsort(errs)]           # best first
        elif kind == "random_scorer":
            cands = cands[rng.permutation(8)]
        elif kind == "worst_scorer":
            cands = cands[np.argsort(-errs)]          # worst first
        if greedy == "separate":
            # a distinct arm, and deliberately BETTER on the last segment only, so
            # a tool that attributed loss to the wrong stage would be caught
            base = gt + rng.normal(0, 1.5, (n_wp, 2))
            base[-seg_len:] = gt[-seg_len:] + rng.normal(0, 0.2, (seg_len, 2))
        else:
            base = cands[0]
        rows.append({"sample_token": tok, "scene": f"s{i//8}",
                     "got_cand_wps": repr(cands.tolist()),
                     "base_pred": repr(base.tolist())})
        gt_map[tok] = gt
    return rows, gt_map


def selftest():
    res = []
    def check(name, ok, detail=""):
        res.append((name, bool(ok), detail))

    def run(kind, seed=0):
        rows, gt = _world(kind, seed=seed)
        return analyse(rows, gt, 3, 2)

    # 1/2/3 -- the discriminating triple: same tool, three scorer qualities
    hits = {}
    for kind in ("perfect_scorer", "random_scorer", "worst_scorer"):
        per, sc, sd = run(kind)
        h = np.mean([p[2]["hit_best"] for p in per])
        hits[kind] = h
    check("perfect scorer -> hit-best 100%", hits["perfect_scorer"] > 0.999,
          f"{hits['perfect_scorer']*100:.1f}%")
    check("worst scorer -> hit-best 0%", hits["worst_scorer"] < 0.001,
          f"{hits['worst_scorer']*100:.1f}%")
    check("random scorer lands between the two",
          hits["worst_scorer"] < hits["random_scorer"] < hits["perfect_scorer"],
          f"{hits['random_scorer']*100:.1f}%")

    # 4 -- a pool with no branching must report no choice, not a lucky scorer
    per, _, _ = run("no_branching")
    live = np.mean([p[2]["n_live"] for p in per])
    check("no branching -> 1 live option at every stage", abs(live - 1.0) < 1e-9,
          f"n_live={live:.2f}")

    # 5 -- attribution is exact, on a world where the gap is NOT zero (otherwise
    #      the check passes trivially -- greedy defaults to candidate 0)
    rows, gt = _world("random_scorer", seed=3, greedy="separate")
    per, sc, sd = analyse(rows, gt, 3, 2)
    direct = []
    for r in rows:
        c = np.asarray(ast.literal_eval(r["got_cand_wps"]), float)
        b = np.asarray(ast.literal_eval(r["base_pred"]), float)
        g = gt[r["sample_token"]]
        direct.append(np.linalg.norm(c[0] - g, axis=1).mean()
                      - np.linalg.norm(b - g, axis=1).mean())
    check("segment shares reconstruct the headline gap (non-zero world)",
          abs(sd.mean(1).mean() - np.mean(direct)) < 1e-9 and abs(np.mean(direct)) > 0.1,
          f"{sd.mean(1).mean():.6f} vs {np.mean(direct):.6f}")

    # 5b -- and it must put the loss in the RIGHT stage: that world made greedy
    #       much better on the last segment only, so stage 3 must dominate
    shares = [sd[:, s].mean() / sd.sum(1).mean() for s in range(3)]
    check("loss lands in the stage that actually differs",
          shares[2] > shares[0] and shares[2] > shares[1],
          "shares " + str(["%.2f" % x for x in shares]))

    # 6 -- the mirror of 5: when greedy IS candidate 0 the gap must be exactly 0.
    #      Together the pair pins the arm wiring -- a tool that swapped the two
    #      arms would fail one or the other. (Builds its own world; reusing 5's
    #      would silently test the wrong thing.)
    rows0, gt0 = _world("random_scorer", seed=3, greedy="is_cand0")
    _, _, sd0 = analyse(rows0, gt0, 3, 2)
    check("greedy == candidate 0 gives exactly zero gap",
          abs(sd0.mean()) < 1e-12, f"{sd0.mean():.2e}")

    # 7 -- prefix grouping: candidates sharing a prefix must collapse to one option
    a = np.zeros((4, 6, 2)); a[2:, 4:] = 1.0     # split only in the last segment
    gr = [len(prefix_groups(a, e)) for e in (2, 4, 6)]
    check("prefix grouping follows the split point", gr == [1, 1, 2], str(gr))

    # 8 -- chance is 1/n_live, the column that keeps hit-best honest
    per, _, _ = run("random_scorer", seed=5)
    ch = np.mean([p[2]["chance"] for p in per])
    nl = np.mean([p[2]["n_live"] for p in per])
    check("chance == 1 / live options", abs(ch - 1.0 / nl) < 0.02,
          f"chance {ch:.3f} vs 1/{nl:.2f}={1/nl:.3f}")

    n_ok = sum(o for _, o, _ in res)
    for name, ok, detail in res:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"   ({detail})" if detail else ""))
    print(f"\nself-test {n_ok}/{len(res)} " + ("PASS" if n_ok == len(res) else "FAIL"))
    return 0 if n_ok == len(res) else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="per_sample.csv with `got_cand_wps` AND `base_pred`")
    ap.add_argument("--records_json", help="the eval set the csv was produced on")
    ap.add_argument("--n_segments", type=int, default=3)
    ap.add_argument("--segment_len", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(selftest())
    if not (args.csv and args.records_json):
        ap.error("--csv and --records_json are required (or pass --selftest)")

    with open(args.records_json, encoding="utf-8") as f:
        gt_map = {r["sample_token"]: np.asarray(r["waypoints"], float) for r in json.load(f)}
    with open(args.csv, encoding="utf-8") as f:
        rows = [r for r in csvmod.DictReader(f) if r.get("got_cand_wps") and r.get("base_pred")]
    if not rows:
        sys.exit(f"{args.csv} has no rows with BOTH `got_cand_wps` and `base_pred`.\n"
                 "  `base_pred` was added to eval_nuscenes late -- results/headline/ref_basepred\n"
                 "  is the run that carries it. Without greedy there is nothing to attribute against.")

    per, scenes, seg_delta = analyse(rows, gt_map, args.n_segments, args.segment_len)
    if not per:
        sys.exit("no record joined the eval set -- is --records_json the right one?")
    report(per, scenes, seg_delta, args.n_segments, args.seed)


if __name__ == "__main__":
    main()
