"""
Preprocess nuScenes into WORLD-MODEL training records.

The action model (preprocess_nuscenes.py) learns image -> waypoints. The world
model learns the complementary map:

    (current CAM_FRONT frame, a short action) -> the CAM_FRONT frame that the
    ego actually observes after executing that action.

Because nuScenes is logged video, the "future frame" is free supervision: for a
keyframe j it is simply the CAM_FRONT image of a later keyframe
j + action_len * future_stride. The action is the first `action_len` future
waypoints expressed in keyframe j's ego frame -- exactly the granularity a GoT
segment commits before the Context Update predicts the next observation.

Each emitted record:
    {
        "sample_token": str,          # the CURRENT keyframe
        "scene": str,
        "cur_image":    abs_path,     # CAM_FRONT at j
        "future_image": abs_path,     # CAM_FRONT at j + action_len*future_stride
        "action": [[x, y], ...],      # action_len waypoints, ego_j frame, metres
    }

The action range is NOT renormalised here; WM training reuses the action model's
nuscenes_norm.json so the <|action|> tokenization is identical across the two
models. Waypoint geometry is computed with the exact helpers from
preprocess_nuscenes.py, so the two pipelines cannot silently drift apart.

Usage
-----
    python -m data.preprocess_nuscenes_wm \
        --dataroot /data/nuscenes \
        --version v1.0-mini \
        --out_dir ./data/nuscenes_wm_records \
        --action_len 2 --future_stride 1
"""

import argparse
import json
import os
from collections import Counter

import numpy as np

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes

# reuse the action pipeline's geometry so the two never diverge
from data.preprocess_nuscenes import (
    sample_ego_pose,
    global_to_ego,
    ordered_samples,
    derive_command,
)

CAM = "CAM_FRONT"


def build_wm_records(nusc, scene_names, action_len, future_stride, ref_sensor):
    """One record per keyframe that has `action_len` future keyframes ahead."""
    records = []
    n_skipped = 0
    name_to_scene = {sc["name"]: sc for sc in nusc.scene}

    for name in scene_names:
        if name not in name_to_scene:
            continue
        scene = name_to_scene[name]
        samples = ordered_samples(nusc, scene)
        poses = [sample_ego_pose(nusc, s, ref_sensor) for s in samples]

        # target frame is action_len*future_stride keyframes ahead
        last_valid = len(samples) - action_len * future_stride
        for j in range(max(0, last_valid)):
            ego_t, ego_q = poses[j]

            # first action_len future ego positions, in keyframe j's ego frame
            fut_global = np.array(
                [poses[j + (k + 1) * future_stride][0] for k in range(action_len)]
            )  # (action_len, 3)
            action = global_to_ego(fut_global, ego_t, ego_q)[:, :2]  # (action_len, 2)

            cur_s = samples[j]
            fut_s = samples[j + action_len * future_stride]
            if CAM not in cur_s["data"] or CAM not in fut_s["data"]:
                n_skipped += 1
                continue

            cur_sd = nusc.get("sample_data", cur_s["data"][CAM])
            fut_sd = nusc.get("sample_data", fut_s["data"][CAM])
            cur_p = os.path.abspath(os.path.join(nusc.dataroot, cur_sd["filename"]))
            fut_p = os.path.abspath(os.path.join(nusc.dataroot, fut_sd["filename"]))
            # partial download: need BOTH current and future frame on disk
            if not (os.path.exists(cur_p) and os.path.exists(fut_p)):
                n_skipped += 1
                continue

            records.append({
                "sample_token": cur_s["token"],
                "scene": name,
                "cur_image": cur_p,
                "future_image": fut_p,
                "action": action.astype(np.float32).round(4).tolist(),
                "command": derive_command(action),  # coarse tag, for inspection only
            })

    return records, n_skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataroot", required=True, help="nuScenes root (samples/, sweeps/, v1.0-*/)")
    ap.add_argument("--version", default="v1.0-mini")
    ap.add_argument("--out_dir", default="./data/nuscenes_wm_records")
    ap.add_argument("--action_len", type=int, default=2,
                    help="waypoints given as the action (== GoT segment length)")
    ap.add_argument("--future_stride", type=int, default=1,
                    help="keyframe stride per waypoint (1 keyframe = 0.5 s)")
    ap.add_argument("--ref_sensor", default="LIDAR_TOP")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=True)

    splits = create_splits_scenes()
    if args.version == "v1.0-mini":
        split_map = {"train": "mini_train", "val": "mini_val"}
    else:
        split_map = {"train": "train", "val": "val"}

    for out_split, nusc_split in split_map.items():
        scene_names = splits[nusc_split]
        records, n_skipped = build_wm_records(
            nusc, scene_names, args.action_len, args.future_stride, args.ref_sensor
        )
        out_path = os.path.join(args.out_dir, f"nuscenes_wm_{args.version}_{out_split}.json")
        with open(out_path, "w") as f:
            json.dump(records, f)
        print(f"[{out_split}] scenes={len(scene_names)}  records={len(records)}  "
              f"skipped={n_skipped}  -> {out_path}")
        if records:
            print(f"    action_len={args.action_len} horizon={args.action_len*args.future_stride*0.5:.1f}s  "
                  f"command={dict(Counter(r['command'] for r in records))}")


if __name__ == "__main__":
    main()
