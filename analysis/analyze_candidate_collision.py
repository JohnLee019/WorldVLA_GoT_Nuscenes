#!/usr/bin/env python
"""Can a COLLISION signal separate the candidates? Measured, not argued. 0 GPU.

The proposal this answers
-------------------------
"Keep the generator at one front camera, but let the SELECTOR see more of
nuScenes -- multi-camera 3D boxes, so candidates that would hit something get
scored down." The architecture is right (the diagnosis says selection is the
bottleneck: minADE_C 2.977 against greedy 3.556). The question is whether the
signal has anything to separate.

There is already a reason to doubt it. On the selected trajectory, coll@1s and
coll@2s are IDENTICAL for GoT and greedy (0.167% = 1 record in 600, and 2.167%
= 13), and coll@3s differs by 1.3 records. If collisions are that rare, a
collision score is CONSTANT across the pool on almost every record, and a
constant cannot rank anything.

But that is the rate on the SELECTED trajectory, which is not the same question.
What matters is whether, WITHIN one record, some candidates collide and others
do not. That has never been measured, and it is measurable for free: the
obstacle boxes are already preprocessed, and got_cand_wps now logs every
candidate's waypoints.

★ Note what the boxes are: nuScenes GROUND-TRUTH 3D annotations. Using them in
the METRIC is standard (UniAD does it). Using them in the SELECTOR is using the
ground truth at inference time -- a real deployment has a detector, not labels.
So a positive result here is an ORACLE CEILING ("with perfect perception, how
much of the gap is reachable?"), not a method. Report it that way. That is
still worth knowing: it upper-bounds the whole multi-camera direction, and if
the ceiling is near zero the direction is closed without building anything.

What it reports
---------------
  A  discriminability   fraction of records where candidates DIFFER in collision
                        status. This is the signal's ceiling: on the rest it is
                        constant and can change nothing.
  B  does it point the right way   among discriminating records, are the
                        collision-free candidates actually the lower-L2 ones?
  C  redundancy gate    correlation with kinematic, and the head-to-head win
                        rate when the two disagree. The project's own criterion
                        (check_signal_redundancy): a win-rate CI covering 0.5
                        means the signal adds nothing on top of what is there.
  D  counterfactual     the exact L2 you would get selecting "collision-free
                        first, then kinematic", next to the current score,
                        greedy and the oracle. Exact, not estimated -- the
                        candidates' true errors are logged.
  E  ★ does it reach the loss   the worst 5% of records carry 46% of the
                        positive loss against greedy. If those are not the
                        records where collisions discriminate, the signal
                        cannot touch the loss even if everything above is
                        favourable.

Usage
-----
    python analyze_candidate_collision.py \
      --csv results/fusion/final_top3/per_sample.csv \
      --collision_json ./data/nuscenes_records/nuscenes_collision_v1.0-trainval_val.json \
      --records_json ./data/nuscenes_records/nuscenes_val_scenespread.json

    python analyze_candidate_collision.py --selftest
"""
# --- 리포 루트를 import 경로에 넣는다 -------------------------------------
# 이 파일은 2026-08-21에 루트에서 이 폴더로 옮겨졌다. 파이썬은 sys.path[0]에
# *스크립트가 있는 폴더*를 넣으므로, 이 두 줄이 없으면 `python analysis/x.py`가
# got_drive / model / xllmx 를 못 찾고 ModuleNotFoundError로 죽는다.
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
# -------------------------------------------------------------------------

import argparse
import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np

from analyze_got_csv import _f, _list, cluster_bootstrap_ci, spearman
from got_drive.collision_metric import CollisionConfig, trajectory_collisions
from learn_reranker import _nested

KEY = "avgL2@3s"
HZ = {"1s": 1, "2s": 3, "3s": 5}


def load(csv_paths, collision_json, records_json, parity="uniad", gt_mask=True):
    with open(collision_json) as f:
        boxes = {r["sample_token"]: r["agent_boxes"] for r in json.load(f)}
    gts = {}
    if records_json:
        with open(records_json) as f:
            gts = {r["sample_token"]: np.asarray(r["waypoints"], float)
                   for r in json.load(f)}
    cfg = CollisionConfig(uniad_parity=(parity == "uniad"))

    pools, skipped = [], defaultdict(int)
    for path in csv_paths:
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("got_status") != "ok" or r.get("base_status") != "ok":
                    skipped["arm_failed"] += 1
                    continue
                tok = r.get("sample_token", "")
                wps, vals = _nested(r, "got_cand_wps"), _list(r, "got_cand_vals")
                if wps is None or not vals or len(wps) != len(vals):
                    skipped["no_wps"] += 1
                    continue
                if tok not in boxes:
                    skipped["no_boxes"] += 1
                    continue
                base = _f(r, f"base_{KEY}")
                got = _f(r, f"got_{KEY}")
                if base is None or got is None:
                    skipped["no_l2"] += 1
                    continue
                gt = gts.get(tok)
                gt_h = None
                if gt is not None and gt.shape[0] >= wps[0].shape[0]:
                    gt_h = gt[: wps[0].shape[0]]
                # per-candidate collision, GT-masked exactly as the eval does
                coll = []
                for w in wps:
                    cm, _ = trajectory_collisions(
                        w, boxes[tok], HZ, cfg,
                        gt_traj=gt_h if (gt_mask and gt_h is not None) else None)
                    coll.append(cm)
                kin = _list(r, "got_cand_kin")
                pools.append({
                    "scene": r.get("scene", "?"), "token": tok,
                    "command": r.get("command", "?"),
                    "wps": wps, "vals": np.asarray(vals, float),
                    "got": got, "base": base,
                    "kin": np.asarray(kin, float) if kin and len(kin) == len(vals) else None,
                    "coll": coll,
                })
    return pools, dict(skipped), cfg


def _flag(pool, key):
    """Per-candidate collision flag (1 = collides) for one horizon key."""
    return np.array([c.get(key, 0.0) for c in pool["coll"]], dtype=float)


def report(pools, hz_keys=("cumColl@3s", "coll@3s", "coll@1s")):
    n = len(pools)
    scenes = len({p["scene"] for p in pools})
    print(f"\n{'=' * 78}\nCAN A COLLISION SIGNAL SEPARATE THE CANDIDATES?\n{'=' * 78}")
    print(f"\n  {n} record-plans / {scenes} scenes, "
          f"{np.mean([len(p['vals']) for p in pools]):.2f} candidates each")
    print("  ! boxes are nuScenes GROUND TRUTH -- everything below is an ORACLE")
    print("    ceiling for the multi-camera direction, not an implementable method.")

    for key in hz_keys:
        flags = [_flag(p, key) for p in pools]
        any_coll = np.array([f.max() > 0 for f in flags])
        splits = np.array([0 < f.sum() < len(f) for f in flags])
        print(f"\n{'-' * 78}\n  [{key}]")
        print(f"    records with ANY colliding candidate   {any_coll.mean():6.1%}"
              f"  ({any_coll.sum()}/{n})")
        print(f"    ★ records where candidates DIFFER      {splits.mean():6.1%}"
              f"  ({splits.sum()}/{n})   <- the signal's ceiling")
        if splits.sum() == 0:
            print("    the flag is constant within every pool: it cannot rank "
                  "anything at this horizon.")
            continue

        # B. among discriminating records, do the collision-free ones have lower L2?
        sub = [p for p, s in zip(pools, splits) if s]
        rhos, gains = [], []
        for p in sub:
            f = _flag(p, key)
            # "higher = better" convention: 1 for collision-free
            rhos.append(spearman(list(1.0 - f), list(-p["vals"])))
            gains.append(float(p["vals"][f > 0].mean() - p["vals"][f == 0].mean()))
        rhos = [r for r in rhos if r == r]
        print(f"    rho(collision-free, -L2) on those      {np.mean(rhos):+.4f}"
              f"   (n={len(rhos)})")
        print(f"    mean L2: colliding - clean             {np.mean(gains):+.4f} m"
              f"   (>0 = colliding candidates really are worse)")

        # C. redundancy against kinematic, project gate criterion
        have_kin = [p for p in sub if p["kin"] is not None]
        if have_kin:
            disagree, wins = 0, 0
            scenes_d = defaultdict(list)
            for p in have_kin:
                f = _flag(p, key)
                j_coll = int(np.argmax((1.0 - f) * 1e6 - p["vals"] * 0))  # first clean
                clean = np.where(f == 0)[0]
                if clean.size == 0:
                    continue
                j_kin = int(np.argmax(p["kin"]))
                # the collision rule only acts when kinematic's pick collides
                if f[j_kin] == 0:
                    continue
                disagree += 1
                # among clean candidates the rule falls back to kinematic
                j_new = int(clean[np.argmax(p["kin"][clean])])
                w = 1.0 if p["vals"][j_new] < p["vals"][j_kin] else 0.0
                wins += w
                scenes_d[p["scene"]].append(w)
            if disagree:
                lo, hi = cluster_bootstrap_ci(scenes_d)
                verdict = ("adds nothing (CI covers 0.5)" if lo <= 0.5 <= hi
                           else "carries independent signal")
                print(f"    disagrees with kinematic on          {disagree} records "
                      f"({disagree / n:.1%})")
                print(f"    ★ win rate when it overrides         {wins / disagree:.3f}"
                      f"  CI [{lo:.3f}, {hi:.3f}]  -> {verdict}")
            else:
                print("    never overrides kinematic (its pick never collides) "
                      "-> adds nothing")

    # D. exact counterfactual with the strongest horizon flag
    key = "cumColl@3s"
    print(f"\n{'-' * 78}\n  [D] EXACT COUNTERFACTUAL  (rule: prefer collision-free, "
          f"then kinematic; flag={key})")
    have_kin = [p for p in pools if p["kin"] is not None]
    if not have_kin:
        print("    got_cand_kin missing; skipped")
    else:
        cur = float(np.mean([p["got"] for p in have_kin]))
        greedy = float(np.mean([p["base"] for p in have_kin]))
        orc = float(np.mean([p["vals"].min() for p in have_kin]))
        kin_only = float(np.mean([p["vals"][int(np.argmax(p["kin"]))] for p in have_kin]))
        picks = []
        for p in have_kin:
            f = _flag(p, key)
            clean = np.where(f == 0)[0]
            idx = clean if clean.size else np.arange(len(f))
            picks.append(p["vals"][int(idx[np.argmax(p["kin"][idx])])])
        rule = float(np.mean(picks))
        print(f"    GoT (current score)      {cur:.4f}")
        print(f"    kinematic alone          {kin_only:.4f}")
        print(f"    collision-gated kinematic{rule:>10.4f}   "
              f"({rule - kin_only:+.4f} vs kinematic, {rule - greedy:+.4f} vs greedy)")
        print(f"    greedy                   {greedy:.4f}")
        print(f"    oracle (minADE_C)        {orc:.4f}")

    # E. does the signal reach where the loss actually is?
    print(f"\n{'-' * 78}\n  [E] DOES IT REACH THE LOSS?")
    dev = np.array([p["got"] - p["base"] for p in pools])
    order = np.argsort(-dev)
    for frac in (0.05, 0.10, 0.20):
        k = max(1, int(round(frac * n)))
        worst = order[:k]
        share = dev[worst][dev[worst] > 0].sum() / max(dev[dev > 0].sum(), 1e-9)
        splits = np.array([0 < _flag(pools[i], "cumColl@3s").sum() < len(pools[i]["vals"])
                           for i in worst])
        allsp = np.array([0 < _flag(p, "cumColl@3s").sum() < len(p["vals"])
                          for p in pools])
        print(f"    worst {frac:>4.0%} of records ({k:3d}) carry {share:5.1%} of the "
              f"loss;  candidates differ in collision on {splits.mean():5.1%} of them "
              f"(overall {allsp.mean():5.1%})")
    print("\n    If the worst records are no more collision-discriminating than")
    print("    average, the signal cannot touch the loss even when everything")
    print("    above looks favourable -- the loss simply is not collisions.")


def _selftest():
    cfg = CollisionConfig(uniad_parity=True)
    # one obstacle straight ahead: a trajectory driving into it must collide,
    # one swerving wide must not -> the flag DOES discriminate here
    T = 6
    boxes = [[[12.0, 0.0, 4.0, 2.0, 0.0]] for _ in range(T)]
    into = np.stack([np.arange(1, T + 1) * 3.0, np.zeros(T)], 1)
    wide = np.stack([np.arange(1, T + 1) * 3.0, np.full(T, 8.0)], 1)
    a, _ = trajectory_collisions(into, boxes, HZ, cfg)
    b, _ = trajectory_collisions(wide, boxes, HZ, cfg)
    assert a["cumColl@3s"] == 1.0, a
    assert b["cumColl@3s"] == 0.0, b
    print("  ok  collision flag separates a head-on path from a wide one")

    pool = {"scene": "s0", "token": "t", "command": "straight",
            "wps": [into, wide], "vals": np.array([9.0, 1.0]),
            "got": 9.0, "base": 5.0, "kin": np.array([1.0, 0.0]),
            "coll": [a, b]}
    f = _flag(pool, "cumColl@3s")
    assert 0 < f.sum() < len(f), f
    # the rule must override kinematic here: kinematic prefers the colliding one
    clean = np.where(f == 0)[0]
    j = int(clean[np.argmax(pool["kin"][clean])])
    assert pool["vals"][j] == 1.0, "collision gate must reject the colliding pick"
    print("  ok  the gate overrides kinematic when kinematic's pick collides")

    # a pool where NO candidate collides must be reported as non-discriminating
    pool2 = dict(pool, wps=[wide, wide], coll=[b, b])
    f2 = _flag(pool2, "cumColl@3s")
    assert not (0 < f2.sum() < len(f2)), f2
    print("  ok  a pool with no collisions is correctly non-discriminating")

    print("\nself-test PASS")


def main():
    p = argparse.ArgumentParser(
        "ceiling of a collision-based selector, from logged candidates (0 GPU)")
    p.add_argument("--csv", nargs="+", help="per_sample.csv with got_cand_wps")
    p.add_argument("--collision_json",
                   default="./data/nuscenes_records/nuscenes_collision_v1.0-trainval_val.json")
    p.add_argument("--records_json", default=None,
                   help="records json, for the GT-collision mask (UniAD parity). "
                        "Without it the rates come out biased HIGH.")
    p.add_argument("--collision_parity", choices=["uniad", "ours"], default="uniad")
    p.add_argument("--no_gt_mask", action="store_true", default=False)
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    if a.selftest:
        _selftest()
        return
    if not a.csv:
        p.error("--csv is required (or --selftest)")
    for f in [*a.csv, a.collision_json]:
        if not os.path.exists(f):
            sys.exit(f"[fatal] missing {f}\n  per_sample.csv is written only when "
                     f"the eval finishes -- wait for the run to complete.")
    if not a.records_json:
        print("[warn] no --records_json: the GT-collision mask is OFF, so every "
              "rate below is biased HIGH versus the eval's own numbers.")

    pools, skipped, cfg = load(a.csv, a.collision_json, a.records_json,
                               a.collision_parity, not a.no_gt_mask)
    if not pools:
        sys.exit(f"[fatal] no usable records. skipped: {skipped}")
    print(f"[load] {len(pools)} record-plans"
          + (f"   skipped {skipped}" if skipped else "")
          + f"   (parity={a.collision_parity}, yaw={cfg.apply_yaw}, "
            f"x_offset={cfg.ego_x_offset}, gt_mask={not a.no_gt_mask})")
    report(pools)


if __name__ == "__main__":
    main()
