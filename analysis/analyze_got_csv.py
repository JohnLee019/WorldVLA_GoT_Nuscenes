"""
Offline re-analysis of an eval_got_nuscenes per_sample.csv. No GPU, no model.

The summary.json reports one number per metric over the whole split, but the
val split is ~87% 'straight' (data/preprocess_nuscenes.derive_command uses a
+/-2 m lateral threshold), so a deliberation layer that helps on turns and hurts
on straights averages out to nothing. Everything here is a BREAKDOWN of numbers
already sitting in the csv -- it re-reads the run, it does not re-run it.

Three questions it answers:

  1. per-command / per-tail-bucket: where does GoT win and where does it lose?
     If the win concentrates on left|right, the headline should be conditional,
     not global.
  2. selection: how often does the score take the pool's best candidate, and
     what does the rank histogram look like? A flat histogram means the score is
     noise; a mass at rank 1-2 with a long tail means it is right on the easy
     calls and wrong on the ones that matter.
  3. score components (only if the csv was produced by an instrumented run, i.e.
     got_cand_kin / got_cand_cmd / got_cand_total present): the rank correlation
     between each score component and the candidate's TRUE error. A negative
     correlation means that component is actively steering the selection the
     wrong way -- which names the term to fix.

Usage:
    python analyze_got_csv.py ./results/got_ep1_s42/per_sample.csv
    python analyze_got_csv.py ./results/got_ep1_s42/per_sample.csv --key L2@3s
"""

import argparse
import ast
import csv
import math
from collections import Counter, defaultdict


# --------------------------------------------------------------------------- #
# small stats helpers (no scipy: gpu-server's eval env does not always have it,
# and planning_metrics.py already established the no-scipy convention)
# --------------------------------------------------------------------------- #

def _mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def _median(xs):
    if not xs:
        return float("nan")
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def _ranks(xs):
    """Average ranks, ties shared (needed for a correct Spearman under ties)."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    out = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        r = 0.5 * (i + j) + 1.0
        for k in range(i, j + 1):
            out[order[k]] = r
        i = j + 1
    return out


def spearman(a, b):
    """Rank correlation. nan for n < 3 or a constant input (no ordering info)."""
    if len(a) != len(b) or len(a) < 3:
        return float("nan")
    ra, rb = _ranks(a), _ranks(b)
    ma, mb = _mean(ra), _mean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = math.sqrt(sum((x - ma) ** 2 for x in ra))
    db = math.sqrt(sum((y - mb) ** 2 for y in rb))
    if da < 1e-12 or db < 1e-12:
        return float("nan")
    return num / (da * db)


def wilcoxon_p(diffs):
    """Two-sided signed-rank p via the tie-corrected normal approximation.

    Same fallback planning_metrics.paired_comparison uses (verified to match
    scipy for n >= 60); this file only ever sees n in the hundreds.
    """
    nz = [d for d in diffs if d != 0.0]
    n = len(nz)
    if n < 10:
        return float("nan")
    r = _ranks([abs(d) for d in nz])
    w = sum(rk for d, rk in zip(nz, r) if d > 0)
    mu = n * (n + 1) / 4.0
    tie = Counter(r)
    corr = sum(t ** 3 - t for t in tie.values()) / 48.0
    sd = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0 - corr)
    if sd < 1e-12:
        return float("nan")
    z = (abs(w - mu) - 0.5) / sd
    return 2.0 * (1.0 - 0.5 * (1.0 + math.erf(z / math.sqrt(2.0))))


def cluster_bootstrap_ci(clusters, n_boot=5000, seed=0):
    """95% CI for the mean paired diff, resampling SCENES (not records).

    nuScenes records are consecutive keyframes, so the ~34 records of one scene
    are repeated looks at the same manoeuvre, not 34 independent trials. A
    record-level bootstrap (or Wilcoxon) treats them as independent and reports
    an interval that is far too tight: the val split is 150 scenes, but the
    first 500 records cover 15 of them and the first 200 cover SIX. Resampling
    whole scenes keeps the within-scene correlation intact.
    """
    import random
    keys = list(clusters)
    if len(keys) < 3:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        vals = []
        for _ in range(len(keys)):
            vals.extend(clusters[keys[rng.randrange(len(keys))]])
        if vals:
            means.append(sum(vals) / len(vals))
    means.sort()
    lo = means[int(0.025 * len(means))]
    hi = means[min(int(0.975 * len(means)), len(means) - 1)]
    return (lo, hi)


def _f(row, key):
    """Float from a csv cell, or None when the column is absent/empty."""
    v = row.get(key, "")
    if v is None or v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def _list(row, key):
    """Python-literal list column (got_cand_vals etc.), or None."""
    v = row.get(key, "")
    if not v:
        return None
    try:
        out = ast.literal_eval(v)
        return [float(x) for x in out] if isinstance(out, list) else None
    except (ValueError, SyntaxError):
        return None


# --------------------------------------------------------------------------- #
# reports
# --------------------------------------------------------------------------- #

def _bucket_report(name, groups, key):
    """Paired GoT-vs-baseline table for a dict of {bucket: [rows]}.

    Reports BOTH the record-level p (p_rec) and the scene-clustered one (p_sc,
    Wilcoxon over per-scene mean diffs) plus a scene-cluster bootstrap CI.
    p_rec is kept only so the gap between the two is visible -- it is the number
    that made a 6-scene run look significant. Report p_sc and ci_sc.
    """
    print(f"\n=== {name} (paired, key={key}) ===")
    print(f"{'bucket':<12} {'n':>5} {'sc':>4} {'GoT':>8} {'base':>8} {'diff':>8} "
          f"{'win':>6} {'p_rec':>7} {'p_sc':>7} {'ci_sc':>17} {'minADE_C':>9} {'gap':>7}")
    for b in sorted(groups, key=lambda x: -len(groups[x])):
        rs = groups[b]
        gv, bv, d, scenes = [], [], [], defaultdict(list)
        for r in rs:
            x, y = _f(r, f"got_{key}"), _f(r, f"base_{key}")
            if x is None or y is None:
                continue
            gv.append(x)
            bv.append(y)
            d.append(x - y)
            scenes[r.get("scene", "?")].append(x - y)
        if not d:
            continue
        win = sum(1 for x in d if x < 0) / len(d)
        orc = [v for v in (_f(r, "got_minADE_C") for r in rs) if v is not None]
        gap = [v for v in (_f(r, f"got_selection_gap_{key}") for r in rs) if v is not None]
        p_sc = wilcoxon_p([_mean(v) for v in scenes.values()])
        lo, hi = cluster_bootstrap_ci(scenes)
        print(f"{b:<12} {len(d):>5} {len(scenes):>4} {_mean(gv):>8.4f} {_mean(bv):>8.4f} "
              f"{_mean(d):>+8.4f} {win:>6.3f} {wilcoxon_p(d):>7.4f} {p_sc:>7.4f} "
              f"[{lo:>+7.4f},{hi:>+7.4f}] {_mean(orc):>9.4f} {_mean(gap):>7.4f}")


def _selection_report(rows, key):
    print(f"\n=== selection quality (key={key}) ===")
    rank = [v for v in (_f(r, "got_selection_rank") for r in rows) if v is not None]
    top1 = [v for v in (_f(r, "got_selection_top1") for r in rows) if v is not None]
    ncand = [v for v in (_f(r, "got_n_candidates") for r in rows) if v is not None]
    if rank:
        hist = Counter(int(round(v)) for v in rank)
        n = len(rank)
        exp = (_mean(ncand) + 1) / 2 if ncand else float("nan")
        print(f"  n={n}  mean_rank={_mean(rank):.3f} (random ~{exp:.2f})  "
              f"top1={_mean(top1):.3f} (random ~{1/_mean(ncand):.3f})")
        for r in sorted(hist):
            bar = "#" * int(60 * hist[r] / n)
            print(f"   rank {r}: {hist[r]:>4} ({hist[r]/n:>5.1%}) {bar}")

    # How much of the available headroom does each rank position hold? If the
    # pool's 2nd-best is nearly as good as its best, chasing top1 is the wrong
    # target and the score only has to avoid the bottom half.
    pools = [p for p in (_list(r, "got_cand_vals") for r in rows) if p and len(p) >= 2]
    if pools:
        # Ragged pools (dedup drops repeats, so |C| varies): average each position
        # over the pools that HAVE it, and print the count. Truncating to the
        # shortest pool instead would silently drop the worst candidates -- which
        # are exactly the ones that set the "pick at random" floor below.
        sizes = Counter(len(p) for p in pools)
        m = max(len(p) for p in pools)
        srt = [sorted(p) for p in pools]
        print(f"\n  pool sizes: {dict(sorted(sizes.items()))}  (n={len(pools)})")
        print(f"  mean true {key} by position -- 'score' = score order (0 = GoT's "
              f"pick), 'sorted' = true order (0 = oracle)")
        print(f"   {'pos':>4} {'n':>5} {'score':>9} {'sorted':>9}")
        for i in range(m):
            a = [p[i] for p in pools if len(p) > i]
            b = [p[i] for p in srt if len(p) > i]
            print(f"   {i:>4} {len(a):>5} {_mean(a):>9.4f} {_mean(b):>9.4f}")

        # How good is the score AS A RANKER? top1 vs 1/|C| is the wrong yardstick:
        # it only asks about the single best candidate and ignores that picking
        # 2nd is nearly as good. The honest scale runs from "pick at random from
        # this pool" to "pick its best", and says how far along it the score got.
        rnd = _mean([_mean(p) for p in pools])
        orc = _mean([p[0] for p in srt])
        selv = _mean([p[0] for p in pools])
        span = rnd - orc
        rec = (rnd - selv) / span if abs(span) > 1e-9 else float("nan")
        print(f"\n  random pick={rnd:.4f}  score pick={selv:.4f}  oracle={orc:.4f}")
        print(f"  -> score recovers {rec:.1%} of the random->oracle range")
        base = [v for v in (_f(r, f"base_{key}") for r in rows) if v is not None]
        if base:
            b = _mean(base)
            need = (rnd - b) / span if abs(span) > 1e-9 else float("nan")
            print(f"  greedy baseline={b:.4f} sits at {need:.1%} of that range: the score "
                  f"must clear\n     that line to beat it, and pool mean > greedy means most "
                  f"candidates are worse\n     than the incumbent. Raising pool quality "
                  f"(--temperatures) moves the line, not just\n     the score.")


def _component_report(rows, key):
    """Only fires on an instrumented run (see got_drive/got_pipeline_drive.py
    final_candidate_scores + the eval_got csv columns)."""
    cols = [("got_cand_kin", "kinematic"), ("got_cand_cmd", "command"),
            ("got_cand_lik", "likelihood"), ("got_cand_total", "path_score")]
    have = [c for c, _ in cols if any(_list(r, c) for r in rows)]
    if not have:
        print("\n=== score components ===\n  (not instrumented: no got_cand_kin / "
              "got_cand_cmd / got_cand_total columns -- run with the patched\n"
              "   got_pipeline_drive.final_candidate_scores to get this table)")
        return

    print(f"\n=== score components vs TRUE error (per-record Spearman, key={key}) ===")
    print("  a component that RANKS candidates well is positively correlated with")
    print("  -true_error; <= 0 means it is steering the choice the wrong way.")
    print(f"{'component':<12} {'n_rec':>6} {'mean_rho':>9} {'median':>8} {'frac<0':>7}")
    for col, label in cols:
        rhos = []
        for r in rows:
            vals = _list(r, "got_cand_vals")
            comp = _list(r, col)
            if not vals or not comp or len(vals) != len(comp) or len(vals) < 3:
                continue
            rho = spearman(comp, [-v for v in vals])
            if not math.isnan(rho):
                rhos.append(rho)
        if rhos:
            frac_neg = sum(1 for x in rhos if x < 0) / len(rhos)
            print(f"{label:<12} {len(rhos):>6} {_mean(rhos):>+9.4f} "
                  f"{_median(rhos):>+8.4f} {frac_neg:>7.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("--key", default="avgL2@3s",
                    help="metric to pair on; must match the run's --rank_key")
    ap.add_argument("--seed", type=int, default=None,
                    help="restrict to one seed (default: all rows in the file)")
    ap.add_argument("--records_json", default=None,
                    help="the val records the run used; enables GT-speed / GT-turn buckets, "
                         "which are the unconfounded way to ask where GoT helps")
    ap.add_argument("--command", nargs="+", default=None,
                    help="restrict to these commands, e.g. --command left right. The score "
                         "components behave oppositely on turns and straights (command is a "
                         "direction signal when there IS a direction and a crude 'stay at y=0' "
                         "prior otherwise), so the pooled correlations average them away")
    args = ap.parse_args()

    with open(args.csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if args.seed is not None:
        rows = [r for r in rows if r.get("seed") == str(args.seed)]
    ok = [r for r in rows
          if r.get("got_status") == "ok" and r.get("base_status") == "ok"]
    print(f"{args.csv_path}: {len(rows)} rows, {len(ok)} with both GoT and baseline ok")
    if args.command:
        keep = set(args.command)
        ok = [r for r in ok if r.get("command") in keep]
        print(f"  filtered to command in {sorted(keep)}: {len(ok)} records")
    if not ok:
        return

    # The headline: GoT vs the greedy free-run over the whole eval set, scene
    # clustered. Every other table here is a breakdown of this one row, and it
    # is the number that goes in the paper, so it is printed first and on its
    # own rather than left for the reader to reconstruct from the buckets.
    _bucket_report("OVERALL", {"all records": ok}, args.key)

    # command is derived from the GT future, not from either predictor, so this
    # split is clean: no regression-to-the-mean, the diffs mean what they say.
    by_cmd = defaultdict(list)
    for r in ok:
        by_cmd[r.get("command", "?")].append(r)
    _bucket_report("by command", by_cmd, args.key)

    # turns pooled: left and right are the same phenomenon (a steering decision
    # exists) and 57+35 records separately are underpowered.
    turns = by_cmd.get("left", []) + by_cmd.get("right", [])
    if turns:
        _bucket_report("turn vs straight", {"turn (l+r)": turns,
                                            "straight": by_cmd.get("straight", [])},
                       args.key)

    # GT-geometry buckets. Independent of BOTH predictors, so unlike a
    # baseline-error split these are not confounded (see the warning printed
    # below); needs the records json the run was evaluated on.
    if args.records_json:
        import json
        with open(args.records_json) as f:
            recs = {r["sample_token"]: r for r in json.load(f)}
        by_spd, by_turn, n_join = defaultdict(list), defaultdict(list), 0
        for r in ok:
            rec = recs.get(r.get("sample_token"))
            if rec is None:
                continue
            wp = rec.get("waypoints") or []
            if len(wp) < 2:
                continue
            n_join += 1
            # mean speed over the 3 s horizon, and how far the GT actually turns
            spd = float(wp[-1][0]) / 3.0
            lat = abs(float(wp[-1][1]))
            for lo, hi, name in [(-1e9, 1.0, "0 stopped <1"), (1.0, 4.0, "1 slow 1-4"),
                                 (4.0, 8.0, "2 mid 4-8"), (8.0, 1e9, "3 fast >8")]:
                if lo <= spd < hi:
                    by_spd[f"{name} m/s"].append(r)
                    break
            for lo, hi, name in [(0.0, 0.5, "0 |dy|<0.5"), (0.5, 2.0, "1 |dy| 0.5-2"),
                                 (2.0, 5.0, "2 |dy| 2-5"), (5.0, 1e9, "3 |dy|>5")]:
                if lo <= lat < hi:
                    by_turn[f"{name} m"].append(r)
                    break
        print(f"\n[joined GT geometry for {n_join}/{len(ok)} records]")
        _bucket_report("by GT speed (unconfounded)", by_spd, args.key)
        _bucket_report("by GT lateral travel (unconfounded)", by_turn, args.key)
    else:
        print("\n[hint] pass --records_json ...val.json for GT-speed / GT-turn "
              "buckets, which are the unconfounded version of the table below")

    # Buckets on the BASELINE's own error. CONFOUNDED -- kept only because it is
    # the obvious thing to look at and someone will compute it anyway.
    # Conditioning on the baseline being accurate selects records where the
    # baseline happened to get lucky, so ANY other predictor scores worse there
    # by regression to the mean, and better in the high-error bucket. The
    # +diff / -diff split this produces is therefore expected even for two
    # equally good predictors. Read the GT-geometry tables instead.
    edges = [(0, 2), (2, 5), (5, 10), (10, 1e9)]
    by_hard = defaultdict(list)
    for r in ok:
        b = _f(r, f"base_{args.key}")
        if b is None:
            continue
        for lo, hi in edges:
            if lo <= b < hi:
                by_hard[f"base {lo}-{hi if hi < 1e9 else 'inf'}"].append(r)
                break
    _bucket_report("by baseline difficulty [CONFOUNDED, see comment]", by_hard, args.key)
    print("  ^ regression to the mean: conditioning on the baseline's own error\n"
          "    biases these diffs (low bucket against GoT, high bucket for it).\n"
          "    Do not report. Use the GT-geometry tables.")

    _selection_report(ok, args.key)
    _component_report(ok, args.key)


if __name__ == "__main__":
    main()
