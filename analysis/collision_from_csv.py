#!/usr/bin/env python
"""Collision rate for any eval run, computed from its per_sample.csv. CPU only.

WHY THIS EXISTS
---------------
`eval_nuscenes.py` computes L2 and nothing else, so every run made with it has an
empty collision column -- the raw-backbone arms, the `_cont3` checkpoints, the
constant-velocity baseline. The published collision numbers all came from
`eval_got_nuscenes.py`, which is the only script wired to `--collision_json`.

But the collision metric needs nothing that is missing: the predicted trajectory,
the ground truth and the sample_token are all already in `per_sample.csv`, and
`got_drive/collision_metric.py` imports no torch. So the column can be filled
retroactively for runs that are already on disk, without touching a GPU.

WHAT IT DOES NOT DO
-------------------
It does not re-decode anything. It scores the trajectory the run already wrote,
so it inherits that run's decoding, seed and crop exactly -- which is what makes
the number comparable to the run's own L2.

PARITY
------
Defaults to `--parity uniad`, matching the rest of the project: axis-aligned ego
box, +0.5 m forward shift, GT-collision masking, vehicle-only obstacles. The three
outputs are different numbers on purpose (sec.7.3):
    coll@t      UniAD convention  -- the published headline
    meanColl@t  ST-P3 convention
    cumColl@t   ours; no public table uses it. Reported, never quoted.

Usage
-----
  python analysis/collision_from_csv.py \\
      --collision_json ./data/nuscenes_records/nuscenes_collision_v1.0-trainval_val.json \\
      ./results/base_ckpt/raw_lumina_constrained \\
      ./results/base_ckpt/finetuned_constrained \\
      ./results/const_velocity

  python analysis/collision_from_csv.py --selftest
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HZ_IDX = {"1s": 1, "2s": 3, "3s": 5}


def _mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def load_rows(run_dir):
    path = run_dir if run_dir.endswith(".csv") else os.path.join(run_dir, "per_sample.csv")
    if not os.path.exists(path):
        sys.exit(f"[fatal] {path} does not exist")
    with open(path, newline="", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r.get("status") == "ok"], path


def score(run_dir, boxes, cfg, gt_mask=True):
    from got_drive.collision_metric import trajectory_collisions

    rows, path = load_rows(run_dir)
    if rows and not rows[0].get("pred"):
        print(f"  [warn] {path} has no `pred` column -- that is a headline csv "
              f"(eval_got_nuscenes writes got_wps/base_wps instead). Skipping.")
        return None

    colls, n_no_boxes = [], 0
    for r in rows:
        tok = r["sample_token"]
        if tok not in boxes:
            n_no_boxes += 1
            continue
        pred = ast.literal_eval(r["pred"])
        gt = ast.literal_eval(r["gt"])
        cm, _ = trajectory_collisions(pred, boxes[tok], HZ_IDX, cfg,
                                      gt_traj=gt if gt_mask else None)
        colls.append(cm)

    if not colls:
        print(f"  [warn] no scorable record in {path} "
              f"({n_no_boxes} tokens had no obstacle boxes). Skipping.")
        return None

    out = {"n_rows": len(rows), "n_scored": len(colls), "n_no_boxes": n_no_boxes}
    for k in colls[0]:
        out[f"{k}_pct"] = round(100.0 * _mean([c[k] for c in colls]), 3)
    for pref in ("coll", "meanColl", "cumColl"):
        out[f"{pref}_avg_pct"] = round(
            sum(out[f"{pref}@{h}_pct"] for h in ("1s", "2s", "3s")) / 3, 3)
    return out


def report(name, m):
    print(f"\n=== {name}   scored {m['n_scored']}/{m['n_rows']} rows"
          + (f", {m['n_no_boxes']} without obstacle boxes" if m["n_no_boxes"] else ""))
    for pref, label in (("coll", "UniAD"), ("meanColl", "ST-P3"), ("cumColl", "ours ")):
        vals = [m[f"{pref}@{h}_pct"] for h in ("1s", "2s", "3s")]
        line = f"  {label}  @1s {vals[0]:>7.3f}  @2s {vals[1]:>7.3f}  @3s {vals[2]:>7.3f}   Avg. {m[f'{pref}_avg_pct']:>7.3f} %"
        print(line + ("   <- not in any public table" if pref == "cumColl" else ""))
    if m["n_scored"] < m["n_rows"]:
        frac = m["n_scored"] / m["n_rows"]
        print(f"  ! this run is scored on {frac:.1%} of its own rows; quote it as "
              f"{m['n_scored']}/{m['n_rows']}, not as a full-set number.")


# --------------------------------------------------------------------------- #
def selftest():
    from got_drive.collision_metric import CollisionConfig, trajectory_collisions
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))

    cfg = CollisionConfig(uniad_parity=True)
    T = 6
    straight = [[4.0 * (i + 1), 0.0] for i in range(T)]

    # world 1 -- an obstacle parked exactly on the path at every step
    blocking = [[[12.0, 0.0, 4.5, 2.0, 0.0]] for _ in range(T)]
    m, per = trajectory_collisions(straight, blocking, HZ_IDX, cfg)
    check("world1 obstacle on the path -> collides", any(per), f"per_step={list(map(int, per))}")

    # world 2 -- the same obstacle pushed far to the side
    clear = [[[12.0, 40.0, 4.5, 2.0, 0.0]] for _ in range(T)]
    m2, per2 = trajectory_collisions(straight, clear, HZ_IDX, cfg)
    check("world2 obstacle far aside -> no collision", not any(per2))

    # world 3 -- GT masking: when the GT itself hits the box, the step is excluded
    m3, per3 = trajectory_collisions(straight, blocking, HZ_IDX, cfg, gt_traj=straight)
    check("world3 GT masking zeroes a step the GT also hits", not any(per3),
          "pred == gt, so every colliding step is masked")

    # world 4 -- the three conventions are allowed to differ, never to be swapped
    keys = set(m.keys())
    check("world4 all three conventions emitted",
          {f"{p}@3s" for p in ("coll", "meanColl", "cumColl")} <= keys)
    check("world4 cumColl >= coll at 3s", m["cumColl@3s"] >= m["coll@3s"],
          f"cum={m['cumColl@3s']} coll={m['coll@3s']}")

    print("\nselftest:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    p = argparse.ArgumentParser("collision rate from an existing per_sample.csv (CPU only)")
    p.add_argument("runs", nargs="*", help="run dirs (or csv paths) written by eval_nuscenes.py")
    p.add_argument("--collision_json", help="obstacle boxes from preprocess_nuscenes_collision.py")
    p.add_argument("--parity", choices=["uniad", "ours"], default="uniad")
    p.add_argument("--no_gt_mask", action="store_true", default=False,
                   help="drop UniAD's GT-collision masking. Rates come out biased HIGH "
                        "versus every published number; for diagnosis only.")
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args()

    if args.selftest:
        sys.exit(selftest())
    if not args.runs or not args.collision_json:
        p.error("give at least one run dir and --collision_json (or pass --selftest)")

    from got_drive.collision_metric import CollisionConfig
    cfg = CollisionConfig(uniad_parity=(args.parity == "uniad"))
    with open(args.collision_json, encoding="utf-8") as f:
        boxes = {r["sample_token"]: r["agent_boxes"] for r in json.load(f)}
    print(f"[coll] {len(boxes)} records have obstacle boxes | parity={args.parity} | "
          f"gt_mask={not args.no_gt_mask} | ego {cfg.ego_length}x{cfg.ego_width} m, "
          f"grid {cfg.resolution} m, yaw={cfg.apply_yaw}, x_offset={cfg.ego_x_offset}")

    for run in args.runs:
        m = score(run, boxes, cfg, gt_mask=not args.no_gt_mask)
        if m:
            report(os.path.basename(os.path.normpath(run)), m)
            with open(os.path.join(run, "collision.json"), "w", encoding="utf-8") as f:
                json.dump({"parity": args.parity, "gt_mask": not args.no_gt_mask, **m}, f, indent=2)


if __name__ == "__main__":
    main()
