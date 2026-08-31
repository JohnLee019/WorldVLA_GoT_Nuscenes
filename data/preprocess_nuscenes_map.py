#!/usr/bin/env python
"""Drivable-area masks per eval record, for the DAC metric. CPU only.

WHAT IT PRODUCES
----------------
One boolean (nx, ny) mask per `sample_token`, on the SAME grid the collision
metric uses (`got_drive.collision_metric.CollisionConfig`: +-50 m, 0.5 m cells,
index 0 = x/forward, index 1 = y/left, cell i covering [x0 + i*r, x0 + (i+1)*r)).
Sharing the grid is the point -- DAC and collision then sit on identical geometry
and `analysis/pdms_components_nuscenes.py` can consume both without resampling.

NO MAP EXPANSION NEEDED
-----------------------
nuScenes' BASE download already ships the drivable surface as four binary rasters
in `maps/*.png` (10 px/m), which is exactly what DAC needs. The v1.3 map expansion
adds VECTOR layers -- lanes, stop lines, crosswalks -- none of which DAC uses.
PROJECT_HANDOFF's note that DAC is blocked on the expansion is wrong for this
metric; it is right for anything lane-level.

WHY THIS QUERIES POINTS INSTEAD OF RASTERISING A PATCH
------------------------------------------------------
The devkit can rasterise a rotated patch, but then the canvas' row/column order
and flip convention have to be matched to ours by hand, and getting it silently
backwards produces a mask that looks plausible and scores nonsense. Instead we
transform our own grid-cell centres ego -> global (a transform we own and can
check) and hand the global metres to `MapMask.is_on_mask`, which owns
global -> pixel. Neither side guesses the other's convention.

THE GATE
--------
A wrong ego->global transform cannot be spotted by looking at a mask. So this
refuses to write anything until it has measured the one property that must hold:
**the human driver stays on the drivable area**. GT trajectories are scored with
the same footprint logic DAC will use, and if fewer than `--min_gt_dac` of them
come back compliant, the transform is wrong and the tool exits instead of
emitting masks that would silently corrupt every DAC number downstream.

Usage
-----
  python data/preprocess_nuscenes_map.py \\
      --dataroot ../data/nuscenes --version v1.0-trainval \\
      --records_json ./data/nuscenes_records/nuscenes_val_scenespread.json \\
      --out ./data/nuscenes_records/nuscenes_drivable_val.npz

  python data/preprocess_nuscenes_map.py --selftest
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "analysis"))

DEFAULT_X = (-50.0, 50.0)
DEFAULT_Y = (-50.0, 50.0)
DEFAULT_RES = 0.5


def grid_centres(x_bound, y_bound, res):
    """(nx, ny, 2) ego-frame centres, matching `_rasterize_box`'s cell indexing."""
    nx = int(round((x_bound[1] - x_bound[0]) / res))
    ny = int(round((y_bound[1] - y_bound[0]) / res))
    xs = x_bound[0] + (np.arange(nx) + 0.5) * res
    ys = y_bound[0] + (np.arange(ny) + 0.5) * res
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    return np.stack([gx, gy], axis=-1)


def ego_to_global(pts_ego, translation, yaw):
    """(..., 2) ego metres -> global metres. x forward, y left, yaw about +z."""
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    return pts_ego @ R.T + np.asarray(translation[:2])


def yaw_from_quaternion(q):
    """nuScenes stores [w, x, y, z]; DAC only needs the heading about +z."""
    w, x, y, z = q
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


# --------------------------------------------------------------------------- #
def build(args):
    from nuscenes.nuscenes import NuScenes
    from got_drive.collision_metric import CollisionConfig
    from pdms_components_nuscenes import dac_violation

    cfg = CollisionConfig(uniad_parity=True)
    if (list(cfg.x_bound) != list(args.x_bound) or list(cfg.y_bound) != list(args.y_bound)
            or abs(cfg.resolution - args.resolution) > 1e-9):
        print(f"[map][warn] you are building on a grid that differs from the collision "
              f"metric's ({cfg.x_bound}, {cfg.y_bound}, {cfg.resolution}). "
              f"pdms_components_nuscenes.py will refuse the mismatch.")

    records = json.load(open(args.records_json, encoding="utf-8"))
    if args.limit:
        records = records[: args.limit]
    print(f"[map] {len(records)} records from {os.path.basename(args.records_json)}")

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    centres = grid_centres(args.x_bound, args.y_bound, args.resolution)
    flat = centres.reshape(-1, 2)
    print(f"[map] grid {centres.shape[0]}x{centres.shape[1]} @ {args.resolution} m "
          f"= {flat.shape[0]} point queries per record")

    masks, gt_ok, n_missing = {}, [], 0
    for n, rec in enumerate(records):
        tok = rec["sample_token"]
        try:
            sample = nusc.get("sample", tok)
            sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
            pose = nusc.get("ego_pose", sd["ego_pose_token"])
            log = nusc.get("log", nusc.get("scene", sample["scene_token"])["log_token"])
            mask_obj = nusc.get("map", log["map_token"])["mask"]
        except KeyError:
            n_missing += 1
            continue

        yaw = yaw_from_quaternion(pose["rotation"])
        g = ego_to_global(flat, pose["translation"], yaw)
        on = mask_obj.is_on_mask(g[:, 0], g[:, 1])
        m = np.asarray(on, dtype=bool).reshape(centres.shape[:2])
        masks[tok] = m

        gt = rec.get("waypoints")
        if gt:
            bad, _ = dac_violation(gt, m, cfg)
            gt_ok.append(0.0 if bad else 1.0)

        if (n + 1) % 100 == 0:
            print(f"[map]   {n + 1}/{len(records)}")

    if not masks:
        sys.exit("[map][fatal] no record produced a mask")
    if n_missing:
        print(f"[map][warn] {n_missing} records had no resolvable ego_pose/map")

    # ---- the gate -----------------------------------------------------------
    frac = float(np.mean(gt_ok)) if gt_ok else 0.0
    print(f"\n[map][gate] GT trajectories inside the drivable area: {frac:.1%} "
          f"({int(sum(gt_ok))}/{len(gt_ok)})")
    if frac < args.min_gt_dac:
        sys.exit(
            f"[map][fatal] the human driver should stay on the road, and only {frac:.1%} of "
            f"GT trajectories do (threshold {args.min_gt_dac:.0%}).\n"
            f"  That is not a property of the data -- it means the ego->global transform is "
            f"wrong, so every mask here would silently corrupt DAC.\n"
            f"  Check, in this order: (1) is LIDAR_TOP the frame the waypoints were built "
            f"in (data/preprocess_nuscenes.py), (2) is the quaternion order [w,x,y,z], "
            f"(3) do the waypoints mean x=forward y=left.\n"
            f"  Nothing was written.")
    print(f"[map][gate] PASS -- the transform reproduces the one property that must hold.")

    meta = {"resolution": args.resolution, "x_bound": list(args.x_bound),
            "y_bound": list(args.y_bound), "gt_dac_frac": round(frac, 4),
            "n_records": len(masks), "source": "nuScenes base maps/*.png (no expansion)"}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, __meta__=json.dumps(meta),
                        **{k: v.astype(np.uint8) for k, v in masks.items()})
    size = os.path.getsize(args.out) / 1e6
    print(f"\n[map] wrote {len(masks)} masks -> {args.out} ({size:.1f} MB)")
    print(f"[map] now: python analysis/pdms_components_nuscenes.py "
          f"--drivable_masks {args.out} --collision_json <...> <run dirs>")


# --------------------------------------------------------------------------- #
def selftest():
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))

    c = grid_centres((-50.0, 50.0), (-50.0, 50.0), 0.5)
    check("grid shape matches the collision grid", c.shape == (200, 200, 2), f"{c.shape}")
    check("cell 0 centre is half a cell inside the lower bound",
          abs(c[0, 0, 0] + 49.75) < 1e-9 and abs(c[0, 0, 1] + 49.75) < 1e-9, f"{c[0, 0]}")
    check("index 0 walks +x, index 1 walks +y",
          c[1, 0, 0] > c[0, 0, 0] and c[0, 1, 1] > c[0, 0, 1])

    # identity pose leaves points untouched
    p = np.array([[1.0, 2.0], [-3.0, 4.0]])
    check("zero yaw at the origin is the identity",
          np.allclose(ego_to_global(p, [0.0, 0.0, 0.0], 0.0), p))
    # +90 deg maps forward(+x) onto global +y
    r = ego_to_global(np.array([[1.0, 0.0]]), [0.0, 0.0, 0.0], np.pi / 2)
    check("yaw +90 deg sends ego-forward to global +y", np.allclose(r, [[0.0, 1.0]], atol=1e-9),
          f"{r}")
    # translation is applied after rotation
    r2 = ego_to_global(np.array([[1.0, 0.0]]), [10.0, -5.0, 0.0], np.pi)
    check("translation applies after rotation", np.allclose(r2, [[9.0, -5.0]], atol=1e-9), f"{r2}")

    # quaternion -> yaw, [w, x, y, z]
    check("identity quaternion -> yaw 0", abs(yaw_from_quaternion([1, 0, 0, 0])) < 1e-12)
    h = np.sqrt(0.5)
    check("90 deg about +z -> yaw pi/2",
          abs(yaw_from_quaternion([h, 0, 0, h]) - np.pi / 2) < 1e-9,
          f"{yaw_from_quaternion([h, 0, 0, h]):.6f}")
    check("180 deg about +z -> |yaw| pi",
          abs(abs(yaw_from_quaternion([0, 0, 0, 1])) - np.pi) < 1e-9)

    # round trip: a point placed by the transform lands where the grid says it should
    yaw = 0.7
    t = [123.0, -45.0, 0.0]
    ego_pt = np.array([[10.0, 3.0]])
    g = ego_to_global(ego_pt, t, yaw)
    back = ego_to_global(g - np.array(t[:2]) + np.array([0.0, 0.0]), [0.0, 0.0, 0.0], -yaw)
    check("ego->global->ego round trip", np.allclose(back, ego_pt, atol=1e-9), f"{back}")

    print("\nselftest:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    p = argparse.ArgumentParser("build drivable-area masks for DAC (CPU only)")
    p.add_argument("--dataroot")
    p.add_argument("--version", default="v1.0-trainval")
    p.add_argument("--records_json")
    p.add_argument("--out", default="./data/nuscenes_records/nuscenes_drivable_val.npz")
    p.add_argument("--x_bound", type=float, nargs=2, default=list(DEFAULT_X))
    p.add_argument("--y_bound", type=float, nargs=2, default=list(DEFAULT_Y))
    p.add_argument("--resolution", type=float, default=DEFAULT_RES)
    p.add_argument("--min_gt_dac", type=float, default=0.90,
                   help="refuse to write masks if fewer GT trajectories than this stay "
                        "drivable -- that means the transform is wrong, not the data")
    p.add_argument("--limit", type=int, default=0, help="debug only; NOT a sample (sec.11.4)")
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args()

    if args.selftest:
        sys.exit(selftest())
    if not args.dataroot or not args.records_json:
        p.error("--dataroot and --records_json are required (or pass --selftest)")
    build(args)


if __name__ == "__main__":
    main()
