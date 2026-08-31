#!/usr/bin/env python
"""Constant-velocity baseline for nuScenes open-loop planning. No model, no GPU.

WHAT THIS MEASURES
------------------
"What score do you get by assuming the car keeps doing exactly what it was
doing?" Read the ego's speed from the PREVIOUS keyframe and extrapolate it for
3 s. No image, no network, no learning -- pure inertia.

WHY IT BELONGS IN THE TABLE
---------------------------
BEV-Planner (CVPR'24) built its critique of this benchmark on exactly this row:
on nuScenes open-loop, constant-velocity extrapolation is competitive with
published SOTA. It is also a STRONGER model-free baseline than the mean-
trajectory prior already in `eval_nuscenes.py` -- the mean prior does not know
how fast this particular car is going; this one does.

sec.1.10(c) already measured the neighbourhood of the answer from the other side:
feeding our model the true first step drops avgL2@3s 3.5557 -> 1.44. So EXPECT
THIS BASELINE TO BE STRONG, possibly stronger than the finetuned model. That is
not a defect in the method -- this project deliberately does not consume ego
status (sec.7.1), which is what makes the BEV-Planner shortcut unavailable to it.
Report the row; interpret it as a statement about the benchmark.

HOW v0 IS DERIVED (and the one approximation in it)
--------------------------------------------------
`measure_causal_ego.py` established the causal estimator: the previous adjacent
keyframe's first-step displacement IS the ego motion over the preceding 0.5 s.
It needs no devkit and no map -- only a records json whose keyframes are
consecutive.

  !! FRAME CAVEAT. Each record's `waypoints` live in THAT record's ego frame. The
  predecessor's displacement is therefore expressed in the predecessor's frame,
  and converting it to the current frame needs a yaw delta we do not have here.
  `--mode straight` (default) sidesteps this completely by using only the
  LONGITUDINAL magnitude and extrapolating along the current heading -- which is
  also what "constant velocity" means in the papers that report this row.
  `--mode xy_raw` uses the raw 2-D vector without any rotation correction; it is
  APPROXIMATE and is reported with a warning, never as the headline.

WHAT IT DOES NOT DO
-------------------
No yaw/curvature model, no map, no obstacle avoidance. A record whose scene has
no adjacent predecessor cannot be scored and is EXCLUDED -- coverage is printed
and every comparison is made on the covered subset only.

READING RULE (fixed here, before the numbers arrive)
----------------------------------------------------
  * Judge on avgL2@3s (ADE) over the covered subset, scene-clustered.
  * Coverage below 95% makes this a subset result, not a table row: the printed
    verdict says so and names the number.
  * This baseline BEATING the model is an expected outcome, not a bug report.
    See the note above before rewriting any conclusion around it.

Usage
-----
  python analysis/eval_constant_velocity.py \
      --records      ./data/nuscenes_records/nuscenes_v1.0-trainval_val.json \
      --eval_records ./data/nuscenes_records/nuscenes_val_scenespread.json \
      --output_dir   ./results/const_velocity \
      --ref          ./results/base_ckpt/incumbent_cont2_ep1

  # with the UniAD/ST-P3 collision rate as well
  python analysis/eval_constant_velocity.py ... \
      --collision_json ./data/nuscenes_records/nuscenes_collision_v1.0-trainval_val.json

  python analysis/eval_constant_velocity.py --selftest

Writes `per_sample.csv` in the SAME schema `eval_nuscenes.py` uses, so
`compare_base_ckpts.py --ref <model_run> <this_run>` works with no glue.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analyze_got_csv import _mean, cluster_bootstrap_ci, wilcoxon_p  # noqa: E402

TS_RE = re.compile(r"__CAM_FRONT__(\d+)\.jpg")
HZ_IDX = {"1s": 1, "2s": 3, "3s": 5}   # eval_nuscenes.py:325 -- 0.5 s spacing
DT_LO, DT_HI = 0.45, 0.55              # what counts as an adjacent keyframe
COVERAGE_FLOOR = 0.95                  # below this it is a subset, not a row
CSV_COLS = ["L2@1s", "L2@2s", "L2@3s", "avgL2@1s", "avgL2@2s", "avgL2@3s",
            "command", "gt", "pred", "sample_token", "scene", "status"]


# --------------------------------------------------------------------------- #
# core
# --------------------------------------------------------------------------- #
def timestamp(rec):
    m = TS_RE.search(rec["images"][0])
    if m is None:
        raise SystemExit(f"cannot parse timestamp from {rec['images'][0]!r}")
    return int(m.group(1))


def diagnose_coverage(records, eval_recs, v0_table, dt_lo=DT_LO, dt_hi=DT_HI):
    """Why did records miss? Separate the three causes -- they need different fixes.

    Counting misses alone cannot tell them apart, and they are not the same
    problem: a scene start is a fact about the data, a dt outside the window is a
    knob, and a token absent from the source is the wrong json. Same shape of
    failure as `probe_future_frames.py`'s FAIL_NO_FUTURE_FRAMES vs FAIL_IMAGES_ABSENT.
    """
    by_scene = defaultdict(list)
    for r in records:
        by_scene[r["scene"]].append(r)
    for seq in by_scene.values():
        seq.sort(key=timestamp)
    pos = {r["sample_token"]: (r["scene"], i)
           for seq in by_scene.values() for i, r in enumerate(seq)}

    causes = defaultdict(int)
    gaps = []
    for rec in eval_recs:
        tok = rec["sample_token"]
        if tok in v0_table:
            continue
        if tok not in pos:
            causes["absent_from_source"] += 1
            continue
        scene, i = pos[tok]
        if i == 0:
            causes["scene_start"] += 1
            continue
        seq = by_scene[scene]
        dt = (timestamp(seq[i]) - timestamp(seq[i - 1])) / 1e6
        causes["dt_out_of_window"] += 1
        gaps.append(dt)

    print(f"\n[cv][diag] why {sum(causes.values())} records have no v0:")
    for k in ("scene_start", "dt_out_of_window", "absent_from_source"):
        if causes[k]:
            print(f"    {k:<20} {causes[k]:>4}")
    if gaps:
        g = sorted(gaps)
        q = lambda f: g[min(len(g) - 1, int(round(f * (len(g) - 1))))]
        print(f"    predecessor dt (s): min {g[0]:.3f}  p25 {q(.25):.3f}  "
              f"median {q(.50):.3f}  p75 {q(.75):.3f}  max {g[-1]:.3f}")
        near = sum(1 for d in g if 0.3 < d < 0.8)
        if near:
            print(f"    -> {near} of them sit within 0.3-0.8 s: keyframe JITTER, not a gap. "
                  f"Re-run with --dt_lo 0.3 --dt_hi 0.8 to recover them.")
        else:
            print(f"    -> none are near 0.5 s: these are real gaps in the source json, "
                  f"not a window setting. Widening --dt_lo/--dt_hi will not help.")
    if causes["absent_from_source"]:
        print(f"    -> {causes['absent_from_source']} eval tokens are NOT in --records at all. "
              f"Wrong json, or the two were built from different splits.")
    return dict(causes)


def build_v0_table(records, dt_lo=DT_LO, dt_hi=DT_HI):
    """sample_token -> (vx, vy) of the PRECEDING 0.5 s, in the predecessor's frame.

    Only adjacent keyframes qualify. A record that opens a scene, or whose
    predecessor is more than one keyframe away, gets no entry.
    """
    by_scene = defaultdict(list)
    for r in records:
        by_scene[r["scene"]].append(r)
    for seq in by_scene.values():
        seq.sort(key=timestamp)

    v0 = {}
    for seq in by_scene.values():
        for a, b in zip(seq, seq[1:]):
            if not dt_lo < (timestamp(b) - timestamp(a)) / 1e6 < dt_hi:
                continue
            wp = a["waypoints"][0]
            v0[b["sample_token"]] = (float(wp[0]), float(wp[1]))
    return v0


def cv_trajectory(v0, time_horizon, mode="straight"):
    """Extrapolate v0 for `time_horizon` steps of 0.5 s. -> list[[x, y]]"""
    vx, vy = v0
    if mode == "straight":
        # longitudinal magnitude carried along the current heading; the lateral
        # component is dropped rather than rotated, see the FRAME CAVEAT.
        return [[vx * (i + 1), 0.0] for i in range(time_horizon)]
    if mode == "xy_raw":
        return [[vx * (i + 1), vy * (i + 1)] for i in range(time_horizon)]
    raise SystemExit(f"unknown --mode {mode!r}")


def l2_metrics(pred, gt, hz_idx=HZ_IDX):
    """Byte-for-byte the contract of eval_nuscenes.l2_metrics."""
    per_step = [math.dist(p, g) for p, g in zip(pred, gt)]
    out = {}
    for label, idx in hz_idx.items():
        out[f"L2@{label}"] = per_step[idx]
        out[f"avgL2@{label}"] = sum(per_step[:idx + 1]) / (idx + 1)
    return out, per_step


def horizon_avg(m):
    """The `Avg.` column UniAD/VAD tables print: mean of the three horizons."""
    return sum(m[f"L2@{h}"] for h in ("1s", "2s", "3s")) / 3.0


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #
def run(args):
    records = json.load(open(args.records, encoding="utf-8"))
    eval_recs = json.load(open(args.eval_records, encoding="utf-8"))

    v0_table = build_v0_table(records, args.dt_lo, args.dt_hi)
    print(f"[cv] v0 source : {os.path.basename(args.records)} -> "
          f"{len(v0_table)} records have an adjacent predecessor")
    if not v0_table:
        raise SystemExit(
            "[cv] [fatal] no adjacent keyframes in --records. That json keeps only a "
            "sparse slice (the eval set is ~5.5 s apart); pass the FULL val json.")

    coll_boxes = coll_cfg = None
    if args.collision_json:
        from got_drive.collision_metric import CollisionConfig, trajectory_collisions
        coll_boxes = {r["sample_token"]: r["agent_boxes"]
                      for r in json.load(open(args.collision_json, encoding="utf-8"))}
        coll_cfg = CollisionConfig(uniad_parity=True)
        print(f"[cv] collision : uniad parity, boxes for "
              f"{sum(1 for r in eval_recs if r['sample_token'] in coll_boxes)}"
              f"/{len(eval_recs)} records, gt_mask=True")

    rows, metrics, colls, scenes = [], [], [], defaultdict(list)
    n_missing = 0

    for rec in eval_recs:
        tok = rec["sample_token"]
        gt = [list(map(float, w)) for w in rec["waypoints"]]
        if len(gt) < args.time_horizon:
            continue
        gt = gt[:args.time_horizon]

        v0 = v0_table.get(tok)
        if v0 is None:
            n_missing += 1
            rows.append({**{c: "" for c in CSV_COLS}, "sample_token": tok,
                         "scene": rec.get("scene", "?"), "command": rec.get("command", ""),
                         "gt": json.dumps(gt), "status": "no_predecessor"})
            continue

        pred = cv_trajectory(v0, args.time_horizon, args.mode)
        m, _ = l2_metrics(pred, gt)
        metrics.append(m)
        scenes[rec.get("scene", "?")].append(m["avgL2@3s"])

        if coll_boxes is not None and tok in coll_boxes:
            from got_drive.collision_metric import trajectory_collisions
            cm, _ = trajectory_collisions(pred, coll_boxes[tok], HZ_IDX, coll_cfg,
                                          gt_traj=gt)
            colls.append(cm)

        rows.append({
            **{k: round(v, 4) for k, v in m.items()},
            "command": rec.get("command", ""),
            "gt": json.dumps([[round(c, 4) for c in w] for w in gt]),
            "pred": json.dumps([[round(c, 4) for c in w] for w in pred]),
            "sample_token": tok, "scene": rec.get("scene", "?"), "status": "ok",
        })

    n_eval, n_total = len(metrics), len(eval_recs)
    coverage = n_eval / n_total if n_total else 0.0
    if not n_eval:
        raise SystemExit("[cv] [fatal] no eval record had a predecessor -- nothing to score.")

    # ---------------- report ----------------
    print(f"\n=== constant-velocity baseline (mode={args.mode}) ===")
    print(f"  covered {n_eval}/{n_total} records ({coverage:.1%}), "
          f"{len(scenes)} scenes; {n_missing} had no adjacent predecessor")
    causes = diagnose_coverage(records, eval_recs, v0_table, args.dt_lo, args.dt_hi)
    # A record that OPENS its scene has no predecessor in any source and no
    # setting recovers it. Judging absolute coverage against a flat threshold
    # therefore condemns a run that is already complete. Score against what is
    # structurally attainable, and report both.
    n_unattainable = causes.get("scene_start", 0) + causes.get("absent_from_source", 0)
    attainable = n_total - n_unattainable
    cov_att = n_eval / attainable if attainable else 0.0
    print(f"  attainable {n_eval}/{attainable} ({cov_att:.1%}) once the "
          f"{n_unattainable} records with no possible predecessor are set aside")
    if args.mode == "xy_raw":
        print("  [warn] --mode xy_raw applies NO yaw correction between frames. "
              "Approximate; do not use as the headline row.")

    summary = {
        "mode": args.mode, "n_eval_records": n_total, "n_evaluated": n_eval,
        "n_no_predecessor": n_missing, "coverage": round(coverage, 4), "coverage_attainable": round(cov_att, 4),
        "n_unattainable": n_unattainable,
        "n_scenes": len(scenes), "dt_window": [args.dt_lo, args.dt_hi],
    }
    for k in ("L2@1s", "L2@2s", "L2@3s", "avgL2@1s", "avgL2@2s", "avgL2@3s"):
        summary[k] = round(_mean([m[k] for m in metrics]), 4)
    summary["L2_horizon_avg"] = round(_mean([horizon_avg(m) for m in metrics]), 4)

    print(f"  L2@1s {summary['L2@1s']:.4f}   L2@2s {summary['L2@2s']:.4f}   "
          f"L2@3s (FDE) {summary['L2@3s']:.4f}   Avg. {summary['L2_horizon_avg']:.4f}")
    print(f"  avgL2@3s (ADE) {summary['avgL2@3s']:.4f}")

    if colls:
        summary["collision"] = {"n_evaluated": len(colls)}
        for k in colls[0]:
            summary["collision"][f"{k}_pct"] = round(100.0 * _mean([c[k] for c in colls]), 3)
        u = [summary["collision"][f"coll@{h}_pct"] for h in ("1s", "2s", "3s")]
        p = [summary["collision"][f"meanColl@{h}_pct"] for h in ("1s", "2s", "3s")]
        summary["collision"]["coll_avg_pct"] = round(sum(u) / 3, 3)
        summary["collision"]["meanColl_avg_pct"] = round(sum(p) / 3, 3)
        print(f"  Coll  UniAD {u}  Avg. {sum(u)/3:.3f}%")
        print(f"        ST-P3 {p}  Avg. {sum(p)/3:.3f}%")

    # ---------------- paired vs a model run ----------------
    if args.ref:
        summary["vs_ref"] = paired_vs_ref(args.ref, rows, n_boot=args.n_boot)
    if args.ref_collision_csv and colls:
        covered = {r["sample_token"] for r in rows if r["status"] == "ok"}
        summary["ref_collision_same_subset"] = ref_collision_on_subset(
            args.ref_collision_csv, covered)

    # ---------------- verdict, branched on what was just measured ----------------
    print("\n--- how to read this run ---")
    if cov_att < COVERAGE_FLOOR:
        print(f"  attainable coverage {cov_att:.1%} < {COVERAGE_FLOOR:.0%}: records that "
              f"COULD have a v0 are still missing one. Check the dt window above "
              f"before quoting this run.")
    elif coverage < 1.0:
        print(f"  every record that can have a v0 has one ({n_eval}/{attainable}). "
              f"The {n_unattainable} excluded records open their scene, so this row is "
              f"defined on {n_eval}/{n_total} by construction -- state that in the caption, "
              f"and read the paired delta below, which is computed on the same records "
              f"for both arms.")
    else:
        print(f"  coverage 100%: usable as a table row on all {n_eval} records.")
    if "vs_ref" in summary and summary["vs_ref"]:
        d = summary["vs_ref"]["mean_diff"]
        lo, hi = summary["vs_ref"]["ci_sc"]
        if lo <= 0.0 <= hi:
            print(f"  vs ref: {d:+.4f} m with ci_sc [{lo:+.4f}, {hi:+.4f}] spanning 0 -- "
                  f"no detectable difference between inertia and the model.")
        elif d < 0:
            print(f"  vs ref: constant velocity is BETTER by {-d:.4f} m "
                  f"(ci_sc [{lo:+.4f}, {hi:+.4f}]). Expected -- see the module docstring: "
                  f"this is a statement about the benchmark, not a defect in the model.")
        else:
            print(f"  vs ref: the model is better by {d:.4f} m "
                  f"(ci_sc [{lo:+.4f}, {hi:+.4f}]).")

    # ---------------- write ----------------
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "per_sample.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS)
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nper-sample -> {csv_path}")
    return summary


def ref_collision_on_subset(csv_path, covered_tokens):
    """Model collision restricted to the records this baseline could score.

    Without this the two collision columns sit on different samples -- the model's
    on all 600, this baseline's on the 450 that have a predecessor. The gap here is
    large enough that it probably survives, but sec.11.4 is explicit that the sample
    is checked BEFORE the statistic, not after it looks convincing.

    Reads a headline-style csv (eval_got_nuscenes.py), which carries per-record
    `base_coll@*` / `base_meanColl@*`. eval_nuscenes.py csvs have no collision
    columns and cannot be used here.
    """
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if rows and "base_coll@1s" not in rows[0]:
        print(f"[cv] [warn] {csv_path} has no base_coll@* columns -- that is an "
              f"eval_nuscenes csv, not a headline csv. Skipping.")
        return None

    keys = [f"base_{p}@{h}" for p in ("coll", "meanColl") for h in ("1s", "2s", "3s")]
    acc = defaultdict(list)
    for r in rows:
        if r.get("base_status") != "ok" or r["sample_token"] not in covered_tokens:
            continue
        for k in keys:
            if r.get(k) not in (None, ""):
                acc[k].append(float(r[k]))
    if not acc:
        print("[cv] [warn] no overlap between --ref_collision_csv and the covered records")
        return None

    out = {"n": len(acc[keys[0]])}
    for k in keys:
        out[k.replace("base_", "") + "_pct"] = round(100.0 * _mean(acc[k]), 3)
    u = [out[f"coll@{h}_pct"] for h in ("1s", "2s", "3s")]
    m = [out[f"meanColl@{h}_pct"] for h in ("1s", "2s", "3s")]
    out["coll_avg_pct"] = round(sum(u) / 3, 3)
    out["meanColl_avg_pct"] = round(sum(m) / 3, 3)
    print(f"\n  model collision on the SAME {out['n']} records")
    print(f"    UniAD {u}  Avg. {out['coll_avg_pct']:.3f}%")
    print(f"    ST-P3 {m}  Avg. {out['meanColl_avg_pct']:.3f}%")
    return out


def paired_vs_ref(ref_dir, rows, key="avgL2@3s", n_boot=10000):
    """Scene-clustered paired comparison against an eval_nuscenes-style run."""
    path = ref_dir if ref_dir.endswith(".csv") else os.path.join(ref_dir, "per_sample.csv")
    if not os.path.exists(path):
        print(f"[cv] [warn] --ref {path} not found; skipping the paired comparison")
        return None

    ref = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("status") == "ok":
                ref[r["sample_token"]] = (r.get("scene", "?"), float(r[key]))

    mine = {r["sample_token"]: (r["scene"], float(r[key]))
            for r in rows if r["status"] == "ok"}
    common = sorted(set(ref) & set(mine))
    if not common:
        print("[cv] [warn] no overlapping sample_tokens with --ref")
        return None

    clusters, diffs = defaultdict(list), []
    for t in common:
        d = mine[t][1] - ref[t][1]      # negative = constant velocity is BETTER
        diffs.append(d)
        clusters[ref[t][0]].append(d)
    lo, hi = cluster_bootstrap_ci(clusters, n_boot=n_boot)
    p_sc = wilcoxon_p([_mean(v) for v in clusters.values()])
    win = sum(1 for d in diffs if d < 0) / len(diffs)

    print(f"\n  paired vs {os.path.basename(os.path.normpath(ref_dir))} "
          f"on {len(common)} records / {len(clusters)} scenes")
    print(f"    ref {key:<10} {_mean([ref[t][1] for t in common]):.4f}")
    print(f"    cv  {key:<10} {_mean([mine[t][1] for t in common]):.4f}")
    print(f"    delta {_mean(diffs):+.4f}  ci_sc [{lo:+.4f}, {hi:+.4f}]  "
          f"p_sc {p_sc:.4f}  cv_win {win:.3f}")
    return {"n": len(common), "n_scenes": len(clusters), "key": key,
            "ref_mean": round(_mean([ref[t][1] for t in common]), 4),
            "cv_mean": round(_mean([mine[t][1] for t in common]), 4),
            "mean_diff": round(_mean(diffs), 4), "ci_sc": [round(lo, 4), round(hi, 4)],
            "p_sc": round(p_sc, 4), "cv_win_rate": round(win, 4)}


# --------------------------------------------------------------------------- #
# self-test -- synthetic worlds where the answer is known in advance
# --------------------------------------------------------------------------- #
def _synth(scene, tok, t_us, wps):
    return {"scene": scene, "sample_token": tok,
            "images": [f"x/n000-0000+0000__CAM_FRONT__{t_us}.jpg"],
            "waypoints": wps, "command": "straight"}


def selftest():
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))

    # world 1 -- exactly constant velocity: the baseline must be ~perfect
    v, recs = 2.0, []
    for i in range(4):
        recs.append(_synth("s1", f"t{i}", 1_000_000 + i * 500_000,
                           [[v * (k + 1), 0.0] for k in range(6)]))
    v0 = build_v0_table(recs)
    check("world1 adjacency", len(v0) == 3, f"got {len(v0)} predecessors / 3 expected")
    pred = cv_trajectory(v0["t1"], 6)
    m, _ = l2_metrics(pred, recs[1]["waypoints"])
    check("world1 constant velocity -> ~0 error", m["avgL2@3s"] < 1e-9,
          f"avgL2@3s={m['avgL2@3s']:.2e}")

    # world 2 -- constant acceleration: inertia must UNDER-predict, and the
    # error must grow with the horizon
    recs = []
    for i in range(4):
        v_i = 2.0 + i        # speed rises each keyframe
        recs.append(_synth("s2", f"a{i}", 1_000_000 + i * 500_000,
                           [[v_i * (k + 1), 0.0] for k in range(6)]))
    v0 = build_v0_table(recs)
    m, per = l2_metrics(cv_trajectory(v0["a2"], 6), recs[2]["waypoints"])
    check("world2 acceleration -> nonzero error", m["avgL2@3s"] > 0.5,
          f"avgL2@3s={m['avgL2@3s']:.4f}")
    check("world2 error grows with horizon", all(x < y for x, y in zip(per, per[1:])),
          f"per_step={[round(x, 2) for x in per]}")

    # world 3 -- keyframes 5.5 s apart (the eval set's own spacing): no
    # predecessor may be inferred, which is the trap the docstring warns about
    recs = [_synth("s3", f"g{i}", 1_000_000 + i * 5_500_000,
                   [[2.0 * (k + 1), 0.0] for k in range(6)]) for i in range(4)]
    check("world3 sparse keyframes -> no adjacency", len(build_v0_table(recs)) == 0)

    # world 4 -- turning: the two modes must actually differ, and straight-mode
    # must carry a lateral error (this is the FRAME CAVEAT being visible)
    turn = [[2.0 * (k + 1), 0.4 * (k + 1)] for k in range(6)]
    recs = [_synth("s4", f"c{i}", 1_000_000 + i * 500_000, turn) for i in range(3)]
    v0 = build_v0_table(recs)
    ms, _ = l2_metrics(cv_trajectory(v0["c1"], 6, "straight"), turn)
    mx, _ = l2_metrics(cv_trajectory(v0["c1"], 6, "xy_raw"), turn)
    check("world4 modes differ on a turn", abs(ms["avgL2@3s"] - mx["avgL2@3s"]) > 1e-6,
          f"straight={ms['avgL2@3s']:.4f} xy_raw={mx['avgL2@3s']:.4f}")
    check("world4 straight-mode carries the lateral error", ms["avgL2@3s"] > 0.5,
          f"avgL2@3s={ms['avgL2@3s']:.4f}")

    # horizon-average column
    check("Avg. column = mean of the three horizons",
          abs(horizon_avg({"L2@1s": 1.0, "L2@2s": 2.0, "L2@3s": 6.0}) - 3.0) < 1e-12)

    print("\nselftest:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser("constant-velocity baseline (CPU only, no torch)")
    p.add_argument("--records", help="FULL val records json (consecutive keyframes) -- "
                                     "the v0 source. NOT the scene-spread eval set.")
    p.add_argument("--eval_records", help="the 600-record scene-spread eval set")
    p.add_argument("--output_dir", default="./results/const_velocity")
    p.add_argument("--ref", default=None,
                   help="an eval_nuscenes run dir to compare against, paired on sample_token")
    p.add_argument("--ref_collision_csv", default=None,
                   help="headline-style per_sample.csv (eval_got_nuscenes.py) whose "
                        "base_coll@* columns are re-averaged over THIS run's covered "
                        "records, so the two collision columns share a sample")
    p.add_argument("--collision_json", default=None,
                   help="obstacle boxes from preprocess_nuscenes_collision.py")
    p.add_argument("--mode", choices=["straight", "xy_raw"], default="straight")
    p.add_argument("--dt_lo", type=float, default=DT_LO,
                   help="min seconds between a record and its predecessor to count as adjacent")
    p.add_argument("--dt_hi", type=float, default=DT_HI,
                   help="max seconds; widen both if the coverage diagnostic reports jitter")
    p.add_argument("--time_horizon", type=int, default=6)
    p.add_argument("--n_boot", type=int, default=10000)
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args()

    if args.selftest:
        sys.exit(selftest())
    if not args.records or not args.eval_records:
        p.error("--records and --eval_records are required (or pass --selftest)")
    run(args)


if __name__ == "__main__":
    main()
