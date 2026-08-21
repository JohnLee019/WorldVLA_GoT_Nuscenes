"""
Preprocess nuScenes into OBSTACLE-BOX records for the collision-rate metric.

The planning records (preprocess_nuscenes.py) hold only the ego's future
waypoints. The UniAD/ST-P3 collision metric additionally needs, for every future
timestep, the boxes of the OTHER agents -- expressed in the SAME frame the
planner predicts in (the t0 ego frame). Open-loop assumption: other agents keep
their ground-truth motion regardless of what the ego plans.

Each emitted record:
    {
        "sample_token": str,               # the t0 keyframe (joins the planning record)
        "scene": str,
        "agent_boxes": [                   # length n_future; index t = waypoint t
            [[x, y, L, W, yaw], ...],      # obstacle boxes at that timestep, ego_t0 frame
            ...
        ],
    }

Geometry reuses preprocess_nuscenes.py's helpers, so the ego frame definition
cannot drift from the trajectory records. Annotation boxes are global
(centre + size + rotation); we map centres with global_to_ego and subtract the
t0 ego yaw from each box yaw, giving boxes in the t0 ego frame.

★ CATEGORY CHOICE MATTERS. --categories decides what counts as an obstacle and
directly changes the reported collision rate. Default follows the common
vehicle+VRU convention; pin it to the repo you compare against.

Usage
-----
    python -m data.preprocess_nuscenes_collision \
        --dataroot /data/nuscenes \
        --version v1.0-trainval \
        --out_dir ./data/nuscenes_records \
        --split val --n_future 6 --future_stride 1
"""

import argparse
import json
import os

import numpy as np
from pyquaternion import Quaternion

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes

from data.preprocess_nuscenes import (
    sample_ego_pose,
    global_to_ego,
    ordered_samples,
)

# What counts as an obstacle. nuScenes category names are prefixed
# (vehicle.*, human.pedestrian.*, ...); matching is by prefix.
DEFAULT_CATEGORIES = ("vehicle.", "human.pedestrian.")


def _yaw_from_quaternion(q: Quaternion) -> float:
    """Heading of a box in the global XY plane."""
    return float(np.arctan2(*q.rotation_matrix[1::-1, 0]))


def boxes_in_ego_frame(nusc, sample, ego_t, ego_q, categories):
    """All obstacle boxes of `sample`, expressed in the (ego_t, ego_q) frame.

    Returns a list of [x, y, L, W, yaw]. nuScenes box size is (width, length,
    height); we emit (length, width) to match the metric's convention.
    """
    ego_yaw = _yaw_from_quaternion(ego_q)
    out = []
    for ann_token in sample["anns"]:
        ann = nusc.get("sample_annotation", ann_token)
        if not ann["category_name"].startswith(tuple(categories)):
            continue
        centre = global_to_ego(np.array([ann["translation"]]), ego_t, ego_q)[0]
        w, l, _h = ann["size"]
        yaw = _yaw_from_quaternion(Quaternion(ann["rotation"])) - ego_yaw
        out.append([round(float(centre[0]), 3), round(float(centre[1]), 3),
                    round(float(l), 3), round(float(w), 3), round(float(yaw), 4)])
    return out


def build_collision_records(nusc, scene_names, n_future, future_stride,
                            ref_sensor, categories):
    """One record per keyframe that has n_future future keyframes ahead."""
    records = []
    name_to_scene = {sc["name"]: sc for sc in nusc.scene}

    for name in scene_names:
        if name not in name_to_scene:
            continue
        samples = ordered_samples(nusc, name_to_scene[name])
        poses = [sample_ego_pose(nusc, s, ref_sensor) for s in samples]

        last_valid = len(samples) - n_future * future_stride
        for i in range(max(0, last_valid)):
            ego_t, ego_q = poses[i]           # t0 ego frame: the planner's frame
            agent_boxes = []
            for k in range(n_future):
                fut = samples[i + (k + 1) * future_stride]
                agent_boxes.append(
                    boxes_in_ego_frame(nusc, fut, ego_t, ego_q, categories))

            records.append({
                "sample_token": samples[i]["token"],
                "scene": name,
                "agent_boxes": agent_boxes,
            })

    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataroot", required=True, help="nuScenes root (samples/, v1.0-*/)")
    ap.add_argument("--version", default="v1.0-trainval")
    ap.add_argument("--out_dir", default="./data/nuscenes_records")
    ap.add_argument("--split", default="val", choices=["train", "val"])
    ap.add_argument("--n_future", type=int, default=6, help="must match the planning records")
    ap.add_argument("--future_stride", type=int, default=1)
    ap.add_argument("--ref_sensor", default="LIDAR_TOP")
    ap.add_argument("--categories", nargs="+", default=list(DEFAULT_CATEGORIES),
                    help="category-name PREFIXES counted as obstacles. Changes the "
                         "collision rate -- pin to the paper/repo you compare against.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=True)

    splits = create_splits_scenes()
    if args.version == "v1.0-mini":
        nusc_split = {"train": "mini_train", "val": "mini_val"}[args.split]
    else:
        nusc_split = args.split
    scene_names = splits[nusc_split]

    records = build_collision_records(
        nusc, scene_names, args.n_future, args.future_stride,
        args.ref_sensor, args.categories)

    out_path = os.path.join(
        args.out_dir, f"nuscenes_collision_{args.version}_{args.split}.json")
    with open(out_path, "w") as f:
        json.dump(records, f)

    n_boxes = [len(b) for r in records for b in r["agent_boxes"]]
    print(f"[{args.split}] scenes={len(scene_names)}  records={len(records)}  -> {out_path}")
    if n_boxes:
        print(f"    categories={args.categories}")
        print(f"    boxes/timestep: mean={np.mean(n_boxes):.1f} max={max(n_boxes)} "
              f"(zero-box timesteps: {sum(1 for n in n_boxes if n == 0)}/{len(n_boxes)})")


if __name__ == "__main__":
    main()
