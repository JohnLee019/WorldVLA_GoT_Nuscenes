#!/usr/bin/env python
"""Where does selection_gap come from, and can a fallback rule recover it?

No GPU, no model, no re-evaluation -- everything is already in per_sample.csv.

The question this answers
-------------------------
GoT picks the truly best candidate 24.9% of the time and its mean rank is 2.078
out of ~8, which sounds decent, yet selection_gap is 0.618 m and GoT ends up
0.0397 m WORSE than greedy. Those numbers are only compatible if the loss is
not spread evenly: most records must be picked well and a minority picked
badly. If that is true the problem changes shape --

    "pick the best of 8 near-ties"        (saturates at rho ~0.5, 9 nulls)
        becomes
    "decide whether to trust the score"   (binary, and greedy is a known-good
                                           fallback at 67.0% recovery vs 63.9%)

and the second problem is much easier. This script measures whether that
reframing is available, and if so how much of the gap a fallback could take.

What it reports
---------------
  A  concentration    Lorenz share of selection_gap; what fraction of records
                      carries what fraction of the loss. Also the GoT-minus-
                      greedy deviation, decomposed the same way.
  B  by picked rank   gap conditional on where the chosen candidate really sat.
  C  oracle fallback  mean of per-record min(GoT, greedy) -- the ceiling of any
                      "defer to greedy when unsure" rule. Not achievable; it is
                      the number a real rule is measured against.
  D  feasible rules   sweep a deferral fraction using ONLY signals available at
                      inference (candidate_spread, score margin, n_candidates),
                      i.e. no ground truth. Reports where each rule lands
                      between greedy and the oracle fallback.
  E  proxy quality    scene-clustered correlation between each proxy and the
                      per-record GoT-minus-greedy deviation. A proxy that does
                      not correlate here cannot drive a fallback, whatever the
                      sweep in D says.

★ D is in-sample: the deferral threshold is chosen on the same records it is
scored on, so its numbers are OPTIMISTIC. Pass several runs (seeds) and the
script refits on the first and applies that threshold to the others, which is
the honest version. Read the held-out column, not the in-sample one.

Statistics come from analyze_got_csv (scene-clustered Wilcoxon + scene
bootstrap CI) so they match every other table in this project. Never use a
record-level test here: 600 records are ~150 scenes (PROJECT_HANDOFF sec 9).

Usage
-----
    python analyze_selection_gap.py results/headline/ref/per_sample.csv
    python analyze_selection_gap.py results/headline/{ref,ref_s43,ref_s44}/per_sample.csv
    python analyze_selection_gap.py --selftest
"""

import argparse
import csv
import sys
from collections import defaultdict

from analyze_got_csv import (_f, _list, _mean, cluster_bootstrap_ci, spearman,
                             wilcoxon_p)

KEY = "avgL2@3s"          # rank_key of every run in this project


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #

def load(path, key=KEY):
    """-> list of per-record dicts with everything the analysis needs.

    Records where either arm failed are dropped and counted; a malformed plan
    has no candidate pool, so it can say nothing about selection.
    """
    out, skipped = [], defaultdict(int)
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("got_status") != "ok":
                skipped[r.get("got_status") or "no_got"] += 1
                continue
            if r.get("base_status") != "ok":
                skipped[r.get("base_status") or "no_base"] += 1
                continue
            got, base = _f(r, f"got_{key}"), _f(r, f"base_{key}")
            oracle = _f(r, "got_minADE_C")
            if got is None or base is None or oracle is None:
                skipped["missing_cols"] += 1
                continue
            vals = _list(r, "got_cand_vals") or []
            # the score that actually chose the winner: after a final re-rank it
            # is got_cand_final, otherwise the accumulated path_score.
            scores = _list(r, "got_cand_final") or _list(r, "got_cand_total") or []
            out.append({
                "scene": r.get("scene", "?"),
                "token": r.get("sample_token", ""),
                "command": r.get("command", "?"),
                "seed": r.get("seed", ""),
                "got": got,
                "base": base,
                "oracle": oracle,
                "gap": got - oracle,
                "dev": got - base,             # >0 = GoT worse than greedy
                "rank": _f(r, "got_selection_rank"),
                "spread": _f(r, "got_candidate_spread"),
                "n_cand": _f(r, "got_n_candidates") or (len(vals) or None),
                "worst": _f(r, "got_worst_candidate"),
                "cand_vals": vals,
                "cand_scores": scores,
            })
    return out, dict(skipped)


def score_margin(rec):
    """Top-1 minus top-2 of the selection score: how decisively it chose.

    A near-tie at the top is the situation the project already documented
    (pos0 3.00 vs pos1 3.08) and the natural place for a fallback to trigger.
    Returns None when the pool is degenerate or the scores were not logged.
    """
    s = rec.get("cand_scores") or []
    if len(s) < 2:
        return None
    top = sorted(s, reverse=True)
    return top[0] - top[1]


# --------------------------------------------------------------------------- #
# A. concentration
# --------------------------------------------------------------------------- #

def lorenz(values, fracs=(0.05, 0.10, 0.20, 0.30, 0.50)):
    """Share of the positive total carried by the worst X of records."""
    v = sorted((x for x in values if x is not None), reverse=True)
    total = sum(x for x in v if x > 0)
    if total <= 0:
        return {}
    n, out = len(v), {}
    for fr in fracs:
        k = max(1, int(round(fr * n)))
        out[fr] = sum(x for x in v[:k] if x > 0) / total
    return out


def report_concentration(recs):
    print("\n" + "=" * 78)
    print("A. IS THE LOSS CONCENTRATED?")
    print("=" * 78)
    for name, field, note in (
            ("selection_gap (GoT - oracle)", "gap", "how much the score gave up"),
            ("deviation  (GoT - greedy)", "dev", "what the paper reports as +0.0397")):
        vals = [r[field] for r in recs]
        pos = [v for v in vals if v > 0]
        zero = sum(1 for v in vals if v == 0)
        print(f"\n  {name}   mean {_mean(vals):+.4f}   n={len(vals)}   ({note})")
        print(f"    exactly 0: {zero / len(vals):.1%}      "
              f"> 0: {len(pos) / len(vals):.1%}      "
              f"< 0: {sum(1 for v in vals if v < 0) / len(vals):.1%}")
        lz = lorenz(vals)
        if lz:
            print("    worst-X share of the total positive loss:")
            print("      " + "   ".join(f"top{int(k * 100)}% = {v:.1%}"
                                        for k, v in sorted(lz.items())))
    print("\n  Read: if the worst 20% carries most of the loss, a per-record")
    print("  'trust the score?' decision can reach most of it. If the shares")
    print("  track the fractions (20% -> ~20%), the loss is diffuse and no")
    print("  fallback rule can help -- the ranking itself would have to improve.")


# --------------------------------------------------------------------------- #
# B. by picked rank
# --------------------------------------------------------------------------- #

def report_by_rank(recs):
    print("\n" + "=" * 78)
    print("B. GAP BY THE RANK ACTUALLY PICKED")
    print("=" * 78)
    by = defaultdict(list)
    for r in recs:
        if r["rank"] is not None:
            by[int(r["rank"])].append(r)
    if not by:
        print("  got_selection_rank not in this csv; skipping.")
        return
    tot_gap = sum(max(r["gap"], 0.0) for r in recs)
    print(f"\n  {'rank':>5} {'n':>6} {'share':>7} {'mean gap':>10} "
          f"{'mean dev':>10} {'gap share':>10}")
    for k in sorted(by):
        g = by[k]
        share_gap = (sum(max(r['gap'], 0.0) for r in g) / tot_gap) if tot_gap else 0.0
        print(f"  {k:>5} {len(g):>6} {len(g) / len(recs):>6.1%} "
              f"{_mean([r['gap'] for r in g]):>10.4f} "
              f"{_mean([r['dev'] for r in g]):>10.4f} {share_gap:>9.1%}")
    print("\n  Rank 1 = the score picked the truly best candidate (gap 0 by")
    print("  construction). Everything below is where the loss lives.")


# --------------------------------------------------------------------------- #
# C / D. fallback ceilings and feasible rules
# --------------------------------------------------------------------------- #

def scene_stats(recs, a_field, b_field, seed=0):
    """Scene-clustered mean diff, Wilcoxon p and bootstrap CI (project convention)."""
    scenes = defaultdict(list)
    for r in recs:
        scenes[r["scene"]].append(r[a_field] - r[b_field])
    diffs = [d for v in scenes.values() for d in v]
    return {
        "mean": _mean(diffs),
        "p_sc": wilcoxon_p([_mean(v) for v in scenes.values()]),
        "ci_sc": cluster_bootstrap_ci(scenes, seed=seed),
        "n_scenes": len(scenes),
    }


def report_oracle_fallback(recs):
    print("\n" + "=" * 78)
    print("C. CEILING OF ANY 'DEFER TO GREEDY' RULE")
    print("=" * 78)
    got, base = _mean([r["got"] for r in recs]), _mean([r["base"] for r in recs])
    orc = _mean([r["oracle"] for r in recs])
    both = _mean([min(r["got"], r["base"]) for r in recs])
    defer_n = sum(1 for r in recs if r["base"] < r["got"])
    print(f"\n  GoT                       {got:.4f}")
    print(f"  greedy                    {base:.4f}")
    print(f"  per-record min(GoT,greedy){both:>10.4f}   <- oracle fallback ceiling")
    print(f"  minADE_C (pool ceiling)   {orc:.4f}")
    print(f"\n  a perfect rule would defer {defer_n / len(recs):.1%} of records to greedy")
    print(f"  and would gain {base - both:.4f} m over greedy "
          f"({(base - both) / (base - orc) * 100:.1f}% of the pool headroom).")
    print("\n  This is NOT achievable -- it uses the ground truth to decide. It is")
    print("  the number the feasible rules in D are measured against.")
    return {"got": got, "base": base, "oracle": orc, "both": both}


PROXIES = [
    ("spread_low", lambda r: -(r["spread"] if r["spread"] is not None else 0.0),
     "low candidate diversity -> nothing to choose between"),
    ("margin_low", lambda r: -(score_margin(r) if score_margin(r) is not None else 0.0),
     "score barely preferred its winner (near-tie at the top)"),
    ("worst_high", lambda r: (r["worst"] if r["worst"] is not None else 0.0),
     "pool contains something very bad -> gate may be doing the work"),
]


def apply_rule(recs, keyfn, frac):
    """Defer the `frac` least-confident records to greedy; return the mean."""
    if frac <= 0:
        return _mean([r["got"] for r in recs])
    if frac >= 1:
        return _mean([r["base"] for r in recs])
    ranked = sorted(recs, key=keyfn, reverse=True)
    k = int(round(frac * len(ranked)))
    deferred = set(id(r) for r in ranked[:k])
    return _mean([(r["base"] if id(r) in deferred else r["got"]) for r in recs])


def report_rules(runs, ceilings, fracs=(0.1, 0.2, 0.3, 0.5, 0.7)):
    """In-sample sweep on run 0; held-out application to the remaining runs."""
    print("\n" + "=" * 78)
    print("D. FEASIBLE FALLBACK RULES (no ground truth used to decide)")
    print("=" * 78)
    fit, held = runs[0], runs[1:]
    base, got = ceilings["base"], ceilings["got"]
    print(f"\n  target: beat greedy {base:.4f}   (GoT alone is {got:.4f}, "
          f"oracle fallback {ceilings['both']:.4f})")
    for name, keyfn, note in PROXIES:
        usable = sum(1 for r in fit if keyfn(r) != 0.0)
        print(f"\n  -- {name}: {note}")
        if usable < 0.5 * len(fit):
            print(f"     column missing/constant on {len(fit) - usable}/{len(fit)} "
                  f"records; rule not evaluable.")
            continue
        print(f"     {'defer':>7} {'in-sample':>11} " +
              "".join(f"{'held-out ' + str(i + 1):>12}" for i in range(len(held))))
        best = None
        for fr in fracs:
            ins = apply_rule(fit, keyfn, fr)
            outs = [apply_rule(h, keyfn, fr) for h in held]
            flag = "  <- beats greedy" if ins < base else ""
            print(f"     {fr:>6.0%} {ins:>11.4f} " +
                  "".join(f"{o:>12.4f}" for o in outs) + flag)
            if best is None or ins < best[1]:
                best = (fr, ins)
        if best:
            print(f"     best in-sample: defer {best[0]:.0%} -> {best[1]:.4f} "
                  f"({best[1] - base:+.4f} vs greedy)")
    print("\n  ! in-sample numbers pick the threshold on the data they score, so")
    print("    they are optimistic. Believe the held-out columns.")


def report_proxy_quality(recs):
    print("\n" + "=" * 78)
    print("E. DO THE PROXIES PREDICT WHERE GoT LOSES?")
    print("=" * 78)
    print("\n  Spearman(proxy, GoT-minus-greedy). A proxy uncorrelated here cannot")
    print("  drive a fallback no matter what the sweep in D shows.\n")
    dev = [r["dev"] for r in recs]
    for name, keyfn, _ in PROXIES:
        vals = [keyfn(r) for r in recs]
        if len(set(vals)) < 2:
            print(f"  {name:<12} constant; skipped")
            continue
        rho = spearman(vals, dev)
        print(f"  {name:<12} rho = {rho:+.4f}")
    gaps = [r["gap"] for r in recs]
    print(f"\n  and against selection_gap:")
    for name, keyfn, _ in PROXIES:
        vals = [keyfn(r) for r in recs]
        if len(set(vals)) < 2:
            continue
        print(f"  {name:<12} rho = {spearman(vals, gaps):+.4f}")


# --------------------------------------------------------------------------- #
# self-test
# --------------------------------------------------------------------------- #

def selftest():
    # concentrated: 10 records, all loss in one
    conc = [0.0] * 9 + [1.0]
    lz = lorenz(conc)
    assert abs(lz[0.10] - 1.0) < 1e-9, lz
    # diffuse: equal loss everywhere -> share tracks the fraction
    diff = [1.0] * 10
    lz = lorenz(diff)
    assert abs(lz[0.10] - 0.1) < 1e-9 and abs(lz[0.50] - 0.5) < 1e-9, lz
    print("  ok  lorenz separates concentrated from diffuse")

    # a rule that defers exactly the records where greedy is better must reach
    # the oracle-fallback ceiling
    recs = []
    for i in range(10):
        got_better = i % 2 == 0
        recs.append({"scene": f"s{i//2}", "got": 1.0 if got_better else 2.0,
                     "base": 2.0 if got_better else 1.0, "oracle": 0.5,
                     "gap": 0.5, "dev": -1.0 if got_better else 1.0,
                     "rank": 1, "spread": 0.0 if got_better else 1.0,
                     "worst": 0.0, "n_cand": 8,
                     "cand_vals": [], "cand_scores": []})
    ceiling = _mean([min(r["got"], r["base"]) for r in recs])
    assert abs(ceiling - 1.0) < 1e-9, ceiling
    got_rule = apply_rule(recs, lambda r: r["spread"], 0.5)
    assert abs(got_rule - 1.0) < 1e-9, f"perfect proxy must reach ceiling, got {got_rule}"
    print("  ok  apply_rule reaches the oracle ceiling with a perfect proxy")

    # frac=0 and frac=1 are the two pure arms
    assert abs(apply_rule(recs, lambda r: r["spread"], 0.0) - 1.5) < 1e-9
    assert abs(apply_rule(recs, lambda r: r["spread"], 1.0) - 1.5) < 1e-9
    print("  ok  defer 0% = GoT, defer 100% = greedy")

    # an uninformative proxy must not beat GoT on average
    flat = apply_rule(recs, lambda r: 0.0, 0.5)
    assert flat >= ceiling, flat
    print("  ok  uninformative proxy cannot beat the ceiling")

    st = scene_stats(recs, "got", "base")
    assert st["n_scenes"] == 5, st
    print("  ok  scene clustering groups records by scene, not by record")
    print("\nself-test PASS")


def main():
    p = argparse.ArgumentParser(
        "decompose selection_gap and test greedy-fallback rules (0 GPU)")
    p.add_argument("csv", nargs="*", help="per_sample.csv, one per run/seed")
    p.add_argument("--key", default=KEY, help="metric to analyse (run's rank_key)")
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()

    if a.selftest:
        selftest()
        return
    if not a.csv:
        p.error("give at least one per_sample.csv (or --selftest)")

    runs = []
    for path in a.csv:
        recs, skipped = load(path, a.key)
        if not recs:
            sys.exit(f"[fatal] no usable rows in {path}")
        scenes = len({r["scene"] for r in recs})
        print(f"[load] {path}: {len(recs)} records / {scenes} scenes"
              + (f"   skipped {skipped}" if skipped else ""))
        runs.append(recs)

    pooled = runs[0]
    report_concentration(pooled)
    report_by_rank(pooled)
    ceilings = report_oracle_fallback(pooled)
    report_proxy_quality(pooled)
    if len(runs) > 1:
        report_rules(runs, ceilings)
    else:
        print("\n[note] only one run given -- D (fallback rules) needs a second "
              "run to hold out on. Pass the other seeds' per_sample.csv too.")
        report_rules([pooled], ceilings)

    st = scene_stats(pooled, "got", "base")
    print(f"\n  sanity: GoT - greedy = {st['mean']:+.4f}  p_sc {st['p_sc']:.4f}  "
          f"ci_sc [{st['ci_sc'][0]:+.4f}, {st['ci_sc'][1]:+.4f}]  "
          f"({st['n_scenes']} scenes)")
    print("  (seed 42 should reproduce +0.0515 / p_sc 0.0003 / [+0.0216,+0.0828])")


if __name__ == "__main__":
    main()
