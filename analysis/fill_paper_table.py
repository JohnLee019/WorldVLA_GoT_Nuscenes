#!/usr/bin/env python
"""Fill the blank cells of the paper's main table (PROJECT_HANDOFF §7.1.1).

Everything here is read back out of summaries that already exist on disk:
no GPU, no torch, no model load, no re-evaluation. The cells still missing at
the end of session 7 are greedy's L2@{1,2,3}s, coll@3s for all three rows and
frac(L2@3s > 10m) for all three rows -- the table quotes seed 42 alone for
those while every other cell is a 3-seed mean, which is exactly the kind of
mixed-provenance row this project already got burned by once (§0, the stale
`wide/` arm).

Two things it does beyond averaging:

  * the greedy baseline is deterministic and is recomputed identically by every
    run, so all three summaries must report the SAME baseline numbers. That is
    a free wiring check (§0 wave 4) and it is checked, not assumed.
  * the mean-trajectory prior has no `tail` block in summary.json (the eval
    only tails GoT and greedy), so frac(L2@3s>10m) for that row is recomputed
    offline from the records. It is model-free -- mean GT waypoints per
    command -- so this reproduces the eval's own number exactly rather than
    approximating it.

★ std is the SAMPLE std (ddof=1). The snippet currently pasted in §7.1.1 uses
np.std, i.e. ddof=0, which does NOT reproduce the +-0.0105 already written into
the table (that value is ddof=1). Do not mix the two in one table.

Usage (gpu-server, from VLA-GoT-release/):

    python fill_paper_table.py \
      --runs results/headline/ref results/headline/ref_s43 results/headline/ref_s44 \
      --records_json ./data/nuscenes_records/nuscenes_val_scenespread.json \
      --train_records_json ./data/nuscenes_records/nuscenes_v1.0-trainval_train.json

--records_json/--train_records_json are optional; without them every cell is
still produced except the mean-trajectory tail.

    python fill_paper_table.py --selftest      # no data needed
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np


# -- verbatim copies of the eval's own helpers --------------------------------
# Copied rather than imported: eval_nuscenes imports torch and the 7B model at
# module scope, which this script has no use for. Both functions are pure and
# a few lines long; keep them in sync with the source if it ever changes.
#   eval_nuscenes.compute_mean_trajectories  (eval_nuscenes.py:64)
#   eval_nuscenes.l2_metrics                 (eval_nuscenes.py:190)

def compute_mean_trajectories(records, time_horizon):
    acc = defaultdict(list)
    for r in records:
        wp = np.array(r["waypoints"], dtype=np.float64)
        if wp.shape[0] < time_horizon:
            continue
        acc[r.get("command", "__all__")].append(wp[:time_horizon])
        acc["__all__"].append(wp[:time_horizon])
    return {k: np.mean(np.stack(v), axis=0) for k, v in acc.items() if v}


def l2_metrics(pred, gt, hz_idx):
    per_step = np.linalg.norm(pred - gt, axis=-1)
    out = {}
    for label, idx in hz_idx.items():
        out[f"L2@{label}"] = float(per_step[idx])
        out[f"avgL2@{label}"] = float(per_step[: idx + 1].mean())
    return out, per_step


# -- aggregation --------------------------------------------------------------

def mean_std(values):
    """mean +- SAMPLE std (ddof=1), matching the numbers already in §7.1.1."""
    v = [x for x in values if x is not None]
    if not v:
        return None, None
    m = float(np.mean(v))
    s = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
    return m, s


def fmt(m, s, nd=4, n=None):
    if m is None:
        return "n/a"
    if n is not None and n < 2:
        return f"{m:.{nd}f}"
    return f"{m:.{nd}f} +- {s:.{nd}f}"


def sole_seed_block(summary, path):
    """The seed arms were run one seed per process, so got_per_seed has one key."""
    per_seed = summary.get("got_per_seed", {})
    if len(per_seed) != 1:
        print(f"[warn] {path}: got_per_seed has {len(per_seed)} seeds "
              f"({sorted(per_seed)}); using the first. Expected one seed per run.")
    seed = sorted(per_seed)[0]
    return seed, per_seed[seed]


def dig(block, *path, default=None):
    cur = block
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def collect(runs, tail_key="L2@3s", cat_m=10.0):
    """-> (rows, seeds, baseline_disagreement) with one entry per metric."""
    got, base, mean_t, seeds = [], [], [], []
    for r in runs:
        p = os.path.join(r, "summary.json")
        if not os.path.exists(p):
            sys.exit(f"[fatal] missing {p}")
        with open(p) as f:
            s = json.load(f)
        seed, blk = sole_seed_block(s, r)
        seeds.append(seed)
        got.append(blk)
        base.append(s.get("baseline_free_run", {}))
        mean_t.append(s.get("baseline_mean_traj", {}))
        if s.get("tail_key") not in (None, tail_key):
            print(f"[warn] {r}: tail_key={s['tail_key']}, expected {tail_key}")

    frac_key = f"{tail_key}_frac_gt_{cat_m:g}m"

    def pull(blocks, key):
        return [b.get(key) for b in blocks]

    def pull_coll(blocks, k="coll@3s_pct"):
        return [dig(b, "collision", k) for b in blocks]

    def pull_tail(blocks, k):
        return [dig(b, "tail", k) for b in blocks]

    def row_for(blocks, with_tail=True):
        r = {}
        for h in ("1s", "2s", "3s"):
            r[f"L2@{h}"] = pull(blocks, f"L2@{h}")
            r[f"coll@{h}_pct"] = pull_coll(blocks, f"coll@{h}_pct")
        r["avgL2@3s"] = pull(blocks, "avgL2@3s")
        # ★ the "Avg." column of the UniAD/VAD/ST-P3 tables is the mean of the
        # THREE horizon values -- NOT our avgL2@3s, which is the cumulative
        # mean of the per-step L2 over 0..3s. Both are printed as "Avg" in the
        # literature; that ambiguity is one of the things BEV-Planner flagged.
        # Keep them as separate columns and never let one stand in for the other.
        r["L2 Avg."] = [
            None if any(x is None for x in trio) else float(np.mean(trio))
            for trio in zip(r["L2@1s"], r["L2@2s"], r["L2@3s"])]
        r["Coll Avg."] = [
            None if any(x is None for x in trio) else float(np.mean(trio))
            for trio in zip(r["coll@1s_pct"], r["coll@2s_pct"], r["coll@3s_pct"])]
        n = len(blocks)
        r["frac>10m"] = pull_tail(blocks, frac_key) if with_tail else [None] * n
        r["P90"] = pull_tail(blocks, f"{tail_key}_p90") if with_tail else [None] * n
        r["P95"] = pull_tail(blocks, f"{tail_key}_p95") if with_tail else [None] * n
        return r

    rows = {
        "GoT": row_for(got),
        "greedy": row_for(base),
        # the eval tails only GoT and greedy; the mean-traj tail is recomputed
        # offline in mean_traj_tail() instead.
        "mean-traj": row_for(mean_t, with_tail=False),
    }
    extras = {
        "minADE_C": [dig(b, "oracle_selection", "minADE_C") for b in got],
        "selection_gap": [dig(b, "oracle_selection", "selection_gap_avgL2@3s") for b in got],
        "selection_top1": [dig(b, "oracle_selection", "selection_top1") for b in got],
        "sec_per_record": [dig(b, "cost", "sec_per_record") for b in got],
        "calls": [dig(b, "cost", "forward_calls_per_record") for b in got],
        "n_evaluated": [b.get("n_evaluated") for b in got],
    }
    return rows, extras, seeds, base, mean_t


def check_baselines(rows, runs, base_blocks):
    """The deterministic rows must be bit-identical across runs (§0 wave 4).

    A mismatch is the signature of a stale arm left over from a different code
    or preprocessing state -- the exact failure that put session-6 numbers into
    a session-7 table. Loud on purpose.
    """
    bad = []
    for label in ("greedy", "mean-traj"):
        for metric, vals in rows[label].items():
            v = [x for x in vals if x is not None]
            if len(v) > 1 and max(v) - min(v) > 1e-9:
                bad.append((label, metric, vals))
    print("\n-- deterministic-baseline check " + "-" * 44)
    if not bad:
        print("  OK  greedy and mean-traj identical across all runs "
              "(no stale arm mixed in).")
    else:
        print("  XX DIFFERS -- these runs are NOT from the same condition. STOP.")
        for label, metric, vals in bad:
            print(f"     {label:10s} {metric:12s} {vals}")
        print("     See PROJECT_HANDOFF sec 0: a runner only overwrites the "
              "arms it runs, so a stale arm survives a code change.")
    return not bad


def mean_traj_tail(records_json, train_records_json, time_horizon, cat_m, tail_key):
    """Recompute the mean-trajectory row's tail offline (model-free, 0 GPU)."""
    with open(records_json) as f:
        records = json.load(f)
    with open(train_records_json) as f:
        mean_src = json.load(f)
    mean_trajs = compute_mean_trajectories(mean_src, time_horizon)

    hz_idx = {"1s": 1, "2s": 3, "3s": 5}
    hz_idx = {k: v for k, v in hz_idx.items() if v < time_horizon}

    vals, n_short, n_missing = [], 0, 0
    l2_all = defaultdict(list)
    for rec in records:
        gt = np.array(rec["waypoints"], dtype=np.float64)
        if gt.shape[0] < time_horizon:
            n_short += 1
            continue
        pred = mean_trajs.get(rec["command"], mean_trajs.get("__all__"))
        if pred is None:
            n_missing += 1
            continue
        m, _ = l2_metrics(pred, gt[:time_horizon], hz_idx)
        vals.append(m[tail_key])
        for k, v in m.items():
            l2_all[k].append(v)
    v = np.asarray(vals)
    return {
        "n": int(v.size), "n_skipped_short_gt": n_short, "n_no_mean_traj": n_missing,
        "frac>10m": float(np.mean(v > cat_m)),
        "P50": float(np.percentile(v, 50)), "P90": float(np.percentile(v, 90)),
        "P95": float(np.percentile(v, 95)), "max": float(v.max()),
        "L2": {k: float(np.mean(x)) for k, x in l2_all.items()},
    }


# -- report -------------------------------------------------------------------

METRICS = [("L2@1s", 4), ("L2@2s", 4), ("L2@3s", 4), ("L2 Avg.", 4),
           ("avgL2@3s", 4), ("coll@1s_pct", 3), ("coll@2s_pct", 3),
           ("coll@3s_pct", 3), ("Coll Avg.", 3), ("frac>10m", 4),
           ("P90", 3), ("P95", 3)]

# The conventional nuScenes open-loop planning table (UniAD / VAD / ST-P3):
# rows = methods, columns = L2 at 1/2/3 s + Avg, then collision at 1/2/3 s + Avg.
# Rows here are OUR arms only -- public SOTA numbers must not share this table,
# because the input differs (1 front camera, no ego status, no history); see
# PROJECT_HANDOFF sec 7.1.0.
STD_ROW_NAMES = {
    "GoT": "VLA + GoT (ours)",
    "greedy": "VLA greedy free-run",
    "mean-traj": "Mean-trajectory prior (model-free)",
}


def standard_table(rows, runs, mt_tail=None, nd=2):
    """Print the conventional table, markdown-ready, with +- only where a seed
    axis exists (i.e. the GoT row -- the two baselines are deterministic)."""
    cols = [("L2@1s", "1s"), ("L2@2s", "2s"), ("L2@3s", "3s"), ("L2 Avg.", "Avg."),
            ("coll@1s_pct", "1s"), ("coll@2s_pct", "2s"),
            ("coll@3s_pct", "3s"), ("Coll Avg.", "Avg.")]
    print(f"\n{'=' * 78}\nSTANDARD TABLE (UniAD/VAD/ST-P3 layout) -- markdown, "
          f"paste into the paper\n{'=' * 78}\n")
    print("| Method | " + " | ".join(h for _, h in cols) + " |")
    print("|---|" + "---|" * len(cols))
    for label in ("GoT", "greedy", "mean-traj"):
        cells = []
        for key, _ in cols:
            vals = rows[label][key]
            m, s = mean_std(vals)
            if m is None:
                cells.append("--")
                continue
            uniq = len({round(x, 9) for x in vals if x is not None})
            cells.append(f"{m:.{nd}f}" if uniq == 1
                         else f"{m:.{nd}f} +- {s:.{nd}f}")
        print(f"| {STD_ROW_NAMES[label]} | " + " | ".join(cells) + " |")
    print("\n  L2 in metres (lower better); Collision in % (lower better).")
    print("  'Avg.' = mean of the three horizons, the UniAD/VAD convention.")
    print("  ! NOT the same as avgL2@3s (cumulative mean of per-step L2 over")
    print("    0..3 s, the ST-P3 convention). Both are printed 'Avg' in the")
    print("    literature -- say which one the column is, in the caption.")
    print("  +- is the sample std over seeds and exists only for the GoT row;")
    print("    both baselines are deterministic (identical in every run).")


def report(runs, rows, extras, seeds, mt_tail):
    n = len(runs)
    print(f"\n{'=' * 78}\nPAPER MAIN TABLE  (sec 7.1.1)   runs={n}  seeds={seeds}")
    print(f"n_evaluated per run: {extras['n_evaluated']}")
    print("mean +- SAMPLE std (ddof=1)  |  deterministic rows show one value\n")

    hdr = f"{'':12s}" + "".join(f"{m:>20s}" for m, _ in METRICS)
    print(hdr)
    print("-" * len(hdr))
    for label in ("GoT", "greedy", "mean-traj"):
        cells = []
        for metric, nd in METRICS:
            vals = rows[label][metric]
            if label == "mean-traj" and metric in ("frac>10m", "P90", "P95") and mt_tail:
                key = {"frac>10m": "frac>10m", "P90": "P90", "P95": "P95"}[metric]
                cells.append(f"{mt_tail[key]:.{nd}f}*")
                continue
            m, s = mean_std(vals)
            uniq = len({round(x, 9) for x in vals if x is not None})
            cells.append(fmt(m, s, nd, n=1 if uniq == 1 else n))
        print(f"{label:12s}" + "".join(f"{c:>20s}" for c in cells))
    if mt_tail:
        print("\n  * recomputed offline from the records (model-free); the eval "
              "does not tail this row.")
        print(f"    n={mt_tail['n']}  skipped_short_gt={mt_tail['n_skipped_short_gt']}"
              f"  P50={mt_tail['P50']:.3f}  max={mt_tail['max']:.3f}")
        got_l2 = mt_tail["L2"]
        print("    cross-check vs summary.json mean-traj L2: "
              + "  ".join(f"{k}={got_l2[k]:.4f}" for k in ("L2@3s", "avgL2@3s")
                          if k in got_l2))

    print("\n-- diagnostics already in the table (regression check) " + "-" * 24)
    for k, nd in (("minADE_C", 4), ("selection_gap", 4), ("selection_top1", 4),
                  ("sec_per_record", 2), ("calls", 1)):
        m, s = mean_std(extras[k])
        uniq = len({round(x, 9) for x in extras[k] if x is not None})
        print(f"  {k:16s} {fmt(m, s, nd, n=1 if uniq == 1 else len(runs))}"
              f"   {[round(x, nd) for x in extras[k] if x is not None]}")

    g = rows["GoT"]["avgL2@3s"]
    b = rows["greedy"]["avgL2@3s"]
    d = [x - y for x, y in zip(g, b) if x is not None and y is not None]
    m, s = mean_std(d)
    print(f"\n  GoT - greedy (avgL2@3s)  {m:+.4f} +- {s:.4f}   "
          f"{[round(x, 4) for x in d]}   <- positive = GoT worse")
    mc, _ = mean_std(extras["minADE_C"])
    bb, _ = mean_std(rows["greedy"]["avgL2@3s"])
    if mc and bb:
        print(f"  headroom (greedy - minADE_C)/greedy = {(bb - mc) / bb * 100:.1f}%")

    print("\n-- expected values (PROJECT_HANDOFF sec 7.1.1, session 7) " + "-" * 24)
    for name, want in (("GoT avgL2@3s", "3.5954 +- 0.0105"),
                       ("greedy avgL2@3s", "3.5557 (deterministic)"),
                       ("GoT - greedy", "+0.0397 +- 0.0105"),
                       ("minADE_C", "2.9770 +- 0.0264"),
                       ("selection_gap", "0.6184 +- 0.0186"),
                       ("selection_top1", "0.2489 +- 0.0142")):
        print(f"  {name:20s} {want}")
    print("  Any mismatch means this script is reading the wrong field -- fix "
          "the script, not the table.")


# -- self-test ----------------------------------------------------------------

def selftest():
    import shutil
    import tempfile

    tmp = tempfile.mkdtemp(prefix="fill_paper_table_")
    try:
        got_vals = [3.6072, 3.5874, 3.5915]
        runs = []
        for i, (seed, g) in enumerate(zip((42, 43, 44), got_vals)):
            d = os.path.join(tmp, f"run{i}")
            os.makedirs(d)
            runs.append(d)
            json.dump({
                "tail_key": "L2@3s",
                "got_per_seed": {str(seed): {
                    "n_evaluated": 600,
                    "L2@1s": 2.0 + i * 0.01, "L2@2s": 4.0 + i * 0.01,
                    "L2@3s": 6.3 + i * 0.01, "avgL2@3s": g,
                    "oracle_selection": {"minADE_C": 2.97 + i * 0.01,
                                         "selection_gap_avgL2@3s": 0.61,
                                         "selection_top1": 0.25},
                    "tail": {"L2@3s_p90": 8.4, "L2@3s_p95": 9.5,
                             "L2@3s_frac_gt_10m": 0.03 + i * 0.001},
                    "collision": {"coll@3s": 0.048, "coll@3s_pct": 4.8 + i * 0.1},
                    "cost": {"forward_calls_per_record": 20.0, "sec_per_record": 14.5},
                }},
                "baseline_free_run": {
                    "L2@1s": 1.9, "L2@2s": 3.9, "L2@3s": 6.2, "avgL2@3s": 3.5557,
                    "tail": {"L2@3s_p90": 8.3, "L2@3s_p95": 9.4,
                             "L2@3s_frac_gt_10m": 0.031},
                    "collision": {"coll@3s_pct": 4.83},
                },
                "baseline_mean_traj": {"L2@1s": 3.0, "L2@2s": 5.5, "L2@3s": 9.3,
                                       "avgL2@3s": 5.4369,
                                       "collision": {"coll@3s_pct": 10.33}},
            }, open(os.path.join(d, "summary.json"), "w"))

        rows, extras, seeds, base, _ = collect(runs)
        assert seeds == ["42", "43", "44"], seeds

        m, s = mean_std(rows["GoT"]["avgL2@3s"])
        assert abs(m - 3.5953667) < 1e-6, m
        assert abs(s - 0.0104510) < 1e-6, f"std must be ddof=1, got {s}"
        assert abs(float(np.std(got_vals, ddof=0)) - 0.0085327) < 1e-6
        print(f"  ok  mean+-std ddof=1: {m:.4f} +- {s:.4f} "
              f"(ddof=0 would be {np.std(got_vals, ddof=0):.4f} -- the value in "
              f"sec 7.1.1 is the ddof=1 one)")

        assert check_baselines(rows, runs, base), "identical baselines must pass"

        # a stale arm must be caught
        s2 = json.load(open(os.path.join(runs[2], "summary.json")))
        s2["baseline_free_run"]["avgL2@3s"] = 3.5727      # the session-6 value
        json.dump(s2, open(os.path.join(runs[2], "summary.json"), "w"))
        rows2, _, _, base2, _ = collect(runs)
        assert not check_baselines(rows2, runs, base2), "stale arm must be caught"
        print("  ok  stale-arm detection fires on a drifted greedy baseline")

        # mean-traj tail recompute, against a hand-checkable case
        recs = os.path.join(tmp, "val.json")
        wp_a = [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0]]
        wp_b = [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [25, 0]]
        json.dump([{"waypoints": wp_a, "command": "straight", "sample_token": "a"},
                   {"waypoints": wp_b, "command": "straight", "sample_token": "b"}],
                  open(recs, "w"))
        t = mean_traj_tail(recs, recs, 6, 10.0, "L2@3s")
        # mean traj last point = (5+25)/2 = 15 -> |15-5|=10 (not >10), |15-25|=10
        assert t["n"] == 2 and t["frac>10m"] == 0.0, t
        json.dump([{"waypoints": wp_a, "command": "straight", "sample_token": "a"},
                   {"waypoints": [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [40, 0]],
                    "command": "straight", "sample_token": "b"}], open(recs, "w"))
        t = mean_traj_tail(recs, recs, 6, 10.0, "L2@3s")
        assert t["n"] == 2 and abs(t["frac>10m"] - 1.0) < 1e-9, t
        print("  ok  mean-traj tail recompute (frac>10m boundary is strict >)")

        # short-GT records are skipped exactly like the eval does
        json.dump([{"waypoints": wp_a, "command": "straight", "sample_token": "a"},
                   {"waypoints": [[0, 0], [1, 0]], "command": "straight",
                    "sample_token": "s"}], open(recs, "w"))
        t = mean_traj_tail(recs, recs, 6, 10.0, "L2@3s")
        assert t["n"] == 1 and t["n_skipped_short_gt"] == 1, t
        print("  ok  short-GT records skipped (matches eval_got_nuscenes:483)")

        print("\nself-test PASS")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    p = argparse.ArgumentParser(
        "fill the blank cells of the paper main table -- 0 GPU, reads summaries")
    p.add_argument("--runs", nargs="+",
                   default=["results/headline/ref", "results/headline/ref_s43",
                            "results/headline/ref_s44"],
                   help="run dirs holding summary.json (one seed each)")
    p.add_argument("--records_json", default=None,
                   help="eval records; enables the mean-traj tail recompute")
    p.add_argument("--train_records_json", default=None,
                   help="train records the mean trajectory is built from")
    p.add_argument("--time_horizon", type=int, default=6)
    p.add_argument("--catastrophe_m", type=float, default=10.0)
    p.add_argument("--tail_key", default="L2@3s")
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args()

    if args.selftest:
        selftest()
        return

    rows, extras, seeds, base, mean_t = collect(
        args.runs, tail_key=args.tail_key, cat_m=args.catastrophe_m)
    ok = check_baselines(rows, args.runs, base)

    mt_tail = None
    if args.records_json and args.train_records_json:
        mt_tail = mean_traj_tail(args.records_json, args.train_records_json,
                                 args.time_horizon, args.catastrophe_m, args.tail_key)
    else:
        print("[note] --records_json/--train_records_json not given; the "
              "mean-traj tail cells stay n/a.")

    report(args.runs, rows, extras, seeds, mt_tail)
    standard_table(rows, args.runs, mt_tail)
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
