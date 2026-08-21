#!/usr/bin/env python
"""Does the selector fail BECAUSE it cannot see the longitudinal axis?
No GPU -- per_sample.csv (got_cand_wps + score components) and the records' GT.

sec.1.9 measured that the 3 s error is 7.5x longitudinal (6.011 m against
0.797 m) while the score's two terms are kinematic smoothness and a lateral
command. That is a suggestive coincidence, not a mechanism. This script turns it
into one, or kills it.

★ THE TRAP, AND WHY THE OBVIOUS TEST IS USELESS
-----------------------------------------------
The obvious test -- decompose the selection loss (score's pick minus oracle) by
axis and see if it is ~7:1 longitudinal -- CANNOT FAIL. Essentially all the
error and all the between-candidate variation already live on the longitudinal
axis, so a selector that picks UNIFORMLY AT RANDOM also produces a ~7:1
longitudinal loss. The number would be reproducing the geometry of the data, not
a property of the selector. (This is the third time in this project that a
statistic looked decisive and was measuring the data's shape instead: see the
alignment nulls in analyze_oracle_structure and the constant-velocity fit in
analyze_error_structure.)

WHAT IS MEASURED INSTEAD: per axis, how much of the room that EXISTS on that
axis does the score actually collect?

    room_a      = random_a - oracle_a       (mean candidate vs best candidate)
    recovery_a  = (random_a - picked_a) / room_a

Recovery is scale-free, so the 7:1 asymmetry in the raw metres divides out and
the axes become comparable. A random selector scores ~0 on both. A selector that
sees both axes scores alike on both. The mechanism claim predicts something
specific and falsifiable: HIGH ON LATERAL, LOW ON LONGITUDINAL.

The self-test builds both worlds WITH the same 7:1 scale asymmetry, so a pass
cannot come from the scale difference alone.

Component correlations are reported next to it as the direct reading: `command`
is a lateral term by construction, so rho(command, -lateral error) is the
positive control, and rho(kinematic, -longitudinal error) is the question --
does smoothness say anything about whether the SPEED LEVEL is right?

⚠️ The per-axis oracles are different candidates in general, and mean|dx| +
mean|dy| is not avgL2 (L1 against L2). These numbers size the axes against each
other; they do not decompose the headline metric.

Usage
-----
    python analyze_axis_selection.py results/fusion/final_top3/per_sample.csv \
        --records_json ./data/nuscenes_records/nuscenes_val_scenespread.json
    python analyze_axis_selection.py --selftest
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np

from analyze_got_csv import _list, spearman
from learn_reranker import _nested

HZ = 5                      # waypoints 0..5 = 0.5 s .. 3.0 s
AXES = ("longitudinal", "lateral")


def axis_err(traj, gt, axis):
    """Mean absolute per-waypoint error on one axis, over 0..3 s."""
    return float(np.abs(traj[:HZ + 1, axis] - gt[:HZ + 1, axis]).mean())


def load(paths, records_json):
    gt_by_tok = {}
    for r in json.load(open(records_json)):
        wp = np.asarray(r.get("waypoints") or [], dtype=np.float64)
        if wp.ndim == 2 and len(wp) > HZ:
            gt_by_tok[r["sample_token"]] = wp
    pools, skipped, seen = [], defaultdict(int), set()
    for p in paths:
        if not os.path.exists(p):
            sys.exit(f"[fatal] {p} does not exist")
        with open(p, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("got_status") != "ok":
                    skipped["arm_failed"] += 1
                    continue
                tok, seed = row.get("sample_token"), str(row.get("seed", ""))
                gt = gt_by_tok.get(tok)
                wps = _nested(row, "got_cand_wps")
                if gt is None or wps is None or len(wps) < 3:
                    skipped["no_gt_or_pool"] += 1
                    continue
                if (tok, seed) in seen:
                    skipped["duplicate"] += 1
                    continue
                seen.add((tok, seed))
                W = np.stack(wps, 0)
                comp = {}
                for name, key in (("kinematic", "got_cand_kin"),
                                  ("command", "got_cand_cmd"),
                                  ("path_score", "got_cand_total")):
                    v = _list(row, key)
                    if v is not None and len(v) == len(W):
                        comp[name] = np.asarray(v, float)
                pools.append({"scene": row.get("scene", "?"), "gt": gt,
                              "wps": W, "comp": comp})
    return pools, dict(skipped)


def _scene_boot(pools, num, den, n_boot=5000, seed=0):
    """95% CI for a RATIO OF MEANS, resampling scenes.

    Recovery is sum(numerator)/sum(denominator), not a mean of per-record
    ratios: a record whose pool has almost no room on an axis has a denominator
    near zero and its ratio explodes. Bootstrapping the ratio of the resampled
    sums keeps that from dominating.
    """
    idx = defaultdict(list)
    for i, p in enumerate(pools):
        idx[p["scene"]].append(i)
    keys = list(idx)
    if len(keys) < 3:
        return float("nan"), float("nan")
    rng = np.random.RandomState(seed)
    num, den = np.asarray(num), np.asarray(den)
    out = []
    for _ in range(n_boot):
        take = []
        for _ in range(len(keys)):
            take.extend(idx[keys[rng.randint(len(keys))]])
        d = den[take].sum()
        if abs(d) > 1e-12:
            out.append(num[take].sum() / d)
    if not out:
        return float("nan"), float("nan")
    out.sort()
    return out[int(0.025 * len(out))], out[min(int(0.975 * len(out)), len(out) - 1)]


def run(pools, n_boot=5000):
    print(f"\n{'=' * 78}\nAXIS OF THE SELECTION LOSS\n{'=' * 78}")
    print(f"\n  {len(pools)} record-plans / {len({p['scene'] for p in pools})} scenes")

    stats = {}
    for ax_i, ax in enumerate(AXES):
        rnd, pick, orc = [], [], []
        for p in pools:
            e = np.array([axis_err(c, p["gt"], ax_i) for c in p["wps"]])
            rnd.append(e.mean())
            pick.append(e[0])            # pool is sorted best-first BY SCORE
            orc.append(e.min())
        rnd, pick, orc = map(np.asarray, (rnd, pick, orc))
        gained, room = rnd - pick, rnd - orc
        rec = gained.sum() / room.sum() if room.sum() > 1e-12 else float("nan")
        lo, hi = _scene_boot(pools, gained, room, n_boot)
        stats[ax] = {"random": rnd.mean(), "picked": pick.mean(),
                     "oracle": orc.mean(), "room": room.mean(),
                     "recovery": rec, "ci": (lo, hi)}

    print(f"\n  {'axis':<14} {'random':>8} {'score pick':>11} {'oracle':>8} "
          f"{'room':>7} {'recovery':>9} {'ci_sc':>19}")
    for ax in AXES:
        s = stats[ax]
        print(f"  {ax:<14} {s['random']:>8.3f} {s['picked']:>11.3f} "
              f"{s['oracle']:>8.3f} {s['room']:>7.3f} {s['recovery']:>9.1%} "
              f"[{s['ci'][0]:>+8.1%},{s['ci'][1]:>+8.1%}]")
    print("  (metres, mean |error| per waypoint on that axis; recovery is the "
          "share of\n   the random->oracle room the score collected, so the 7:1 "
          "scale divides out)")

    # ★ the baseline the raw metre loss has to be read against. A selector that
    # picks uniformly at random loses exactly the room on each axis, so its raw
    # lon:lat loss ratio IS the geometry of the data. If the score's raw ratio
    # sits near it, the raw ratio is telling you about nuScenes, not about the
    # score -- which is why the verdict below uses recovery instead.
    geo = stats["longitudinal"]["room"] / max(stats["lateral"]["room"], 1e-9)
    sc_lon = stats["longitudinal"]["picked"] - stats["longitudinal"]["oracle"]
    sc_lat = stats["lateral"]["picked"] - stats["lateral"]["oracle"]
    print(f"\n  raw metre loss, lon:lat   score {sc_lon / max(sc_lat, 1e-9):>7.1f}:1"
          f"    random selector (= the geometry) {geo:>5.1f}:1")

    # ---- component correlations, within pool -------------------------------
    have = [k for k in ("kinematic", "command", "path_score")
            if all(k in p["comp"] for p in pools)]
    if have:
        print(f"\n  -- within-pool Spearman(component, -error) by axis")
        print(f"  {'component':<14} {'longitudinal':>14} {'lateral':>10}")
        for name in have:
            row = []
            for ax_i in range(2):
                rr = []
                for p in pools:
                    e = [axis_err(c, p["gt"], ax_i) for c in p["wps"]]
                    r = spearman(list(p["comp"][name]), [-x for x in e])
                    if r == r:
                        rr.append(r)
                row.append(float(np.mean(rr)) if rr else float("nan"))
            print(f"  {name:<14} {row[0]:>+14.4f} {row[1]:>+10.4f}")
            stats.setdefault("rho", {})[name] = row

    # ---- verdict ------------------------------------------------------------
    lon, lat = stats["longitudinal"], stats["lateral"]
    print(f"\n{'-' * 78}\nVERDICT\n{'-' * 78}")
    print(f"  recovery  longitudinal {lon['recovery']:.1%}   "
          f"lateral {lat['recovery']:.1%}")
    blind = lat["recovery"] - lon["recovery"] > 0.15 and lon["recovery"] < 0.25
    if blind:
        print("  MECHANISM SUPPORTED: the score collects the room on the axis it")
        print("  measures (lateral, via `command`) and leaves the room on the axis")
        print("  it does not (longitudinal). The selection failure is not generic")
        print("  noise -- it is a blind spot with a name, and it explains why every")
        print("  reranking of these same candidates was null: none of the signals")
        print("  reranked on speed correctness.")
    elif lon["recovery"] >= 0.25:
        print("  NOT A BLIND SPOT: the score does collect longitudinal room, so")
        print("  'it cannot see the axis that carries the error' is FALSE as a")
        print("  mechanism. sec.1 claim 15 stays a coincidence of the geometry and")
        print("  must not be written as causal.")
    else:
        print("  INCONCLUSIVE: the score collects little room on EITHER axis, so")
        print("  the axes do not separate its failure. It is failing everywhere,")
        print("  which is a weaker (and already known) statement.")
    return stats


# --------------------------------------------------------------------------- #

def _selftest():
    def world(kind, n=400, n_cand=8, seed=0):
        """Both worlds carry the SAME 7:1 longitudinal:lateral error scale.

        That is the control that matters: if the script reported an asymmetry
        from the scale alone, 'sees_both' would pass too and the test would be
        measuring the data's shape again.
        """
        rng = np.random.RandomState(seed)
        pools = []
        for i in range(n):
            gt = np.stack([np.arange(1, HZ + 2) * 4.0, np.zeros(HZ + 1)], 1)
            W, score = [], []
            for _ in range(n_cand):
                dlon, dlat = rng.randn() * 3.5, rng.randn() * 0.5   # 7:1
                W.append(gt + np.stack([np.full(HZ + 1, dlon),
                                        np.full(HZ + 1, dlat)], 1))
                # higher = better, as path_score is used
                noise = rng.randn() * 0.15
                score.append(-abs(dlat) + noise if kind == "lat_only"
                             else -(abs(dlon) + abs(dlat)) + noise)
            order = np.argsort(score)[::-1]          # pool is sorted best-first
            W = [W[j] for j in order]
            score = [score[j] for j in order]
            pools.append({"scene": f"sc{i // 4}", "gt": gt, "wps": np.stack(W, 0),
                          "comp": {"path_score": np.asarray(score)}})
        return pools

    print("[selftest] LAT_ONLY world (score sees lateral only)")
    a = run(world("lat_only", seed=1), n_boot=400)
    print("\n[selftest] SEES_BOTH world (same 7:1 scale, score sees both)")
    b = run(world("sees_both", seed=1), n_boot=400)

    assert a["lateral"]["recovery"] > 0.6, a["lateral"]["recovery"]
    assert a["longitudinal"]["recovery"] < 0.2, a["longitudinal"]["recovery"]
    print(f"\n  ok  lat_only: lateral {a['lateral']['recovery']:.1%} vs "
          f"longitudinal {a['longitudinal']['recovery']:.1%} -> blind spot found")

    assert b["longitudinal"]["recovery"] > 0.3, b["longitudinal"]["recovery"]
    print(f"  ok  sees_both: longitudinal {b['longitudinal']['recovery']:.1%} "
          f"-> NOT reported as a blind spot, despite the identical 7:1 scale")

    # ★ the control that makes the script worth trusting: the ROOM ratio -- the
    # loss a uniformly random selector takes, i.e. the geometry of the data --
    # is the same ~7:1 in both worlds, whatever the selector does. So a raw
    # "the loss is 7:1 longitudinal" reading describes nuScenes, not the score,
    # and the verdict must normalise by it. (An earlier version of this test
    # claimed the raw ratio cannot separate the worlds at all; that was wrong,
    # and the fix is to print the baseline rather than to assert the claim.)
    for name, w in (("lat_only", a), ("sees_both", b)):
        room = w["longitudinal"]["room"] / max(w["lateral"]["room"], 1e-9)
        print(f"  -- room (random selector's loss) lon:lat in {name:<10} "
              f"{room:>5.1f}:1")
        assert 4.0 < room < 12.0, (name, room)
    print("  ok  the geometry is ~7:1 in BOTH worlds, so a raw metre ratio has to "
          "be read\n      against it -- recovery does that by construction")
    print("\nself-test PASS")


def main():
    p = argparse.ArgumentParser("axis of the selection loss (0 GPU)")
    p.add_argument("csv", nargs="*")
    p.add_argument("--records_json")
    p.add_argument("--n_boot", type=int, default=5000)
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    if a.selftest:
        _selftest()
        return
    if not a.csv or not a.records_json:
        p.error("give per_sample.csv files and --records_json (or --selftest)")
    pools, skipped = load(a.csv, a.records_json)
    if not pools:
        sys.exit(f"[fatal] nothing usable -- needs got_cand_wps. skipped: {skipped}")
    if skipped:
        print(f"[load] skipped {skipped}")
    run(pools, a.n_boot)


if __name__ == "__main__":
    main()
