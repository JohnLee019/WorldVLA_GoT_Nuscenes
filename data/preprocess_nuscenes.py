"""
Preprocess nuScenes into the record format consumed by this VLA pipeline.

For every keyframe (2 Hz `sample`) we emit one record:
    - input  : CAM_FRONT image path (optionally more cameras)
    - output : the ego vehicle's future trajectory, expressed as `n_future`
               waypoints in the *current* ego frame (x = forward, y = left),
               sampled every `future_stride` keyframes (0.5 s each).

This mirrors the standard nuScenes open-loop planning convention
(UniAD / VAD / BEV-Planner): 3 s horizon, 0.5 s interval -> 6 waypoints.

The output json is a list of dicts:
    {
        "sample_token": str,
        "scene": str,
        "images": [abs_path, ...],       # ordered as `--cameras`
        "waypoints": [[x, y], ...],      # length == n_future, ego frame, metres
        "command": "left" | "right" | "straight",
    }

We do NOT tokenize here. Image VQGAN tokenization happens at train time inside
the item_processor, so this file only needs numpy + the nuscenes devkit.

Usage
-----
    python -m data.preprocess_nuscenes \
        --dataroot /data/nuscenes \
        --version v1.0-mini \
        --out_dir ./data/nuscenes_records \
        --n_future 6 --future_stride 1

Install (on the training server):
    pip install nuscenes-devkit
"""

import argparse
import json
import os
from collections import Counter

import numpy as np
from pyquaternion import Quaternion

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes


# ---------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------
def sample_ego_pose(nusc, sample, ref_sensor="LIDAR_TOP"):
    """Return (translation[3], Quaternion) of the ego pose that is treated as
    the canonical pose of this keyframe. Following the planning literature we
    use the LIDAR_TOP sample_data's ego_pose as the reference frame."""
    sd = nusc.get("sample_data", sample["data"][ref_sensor])
    ego = nusc.get("ego_pose", sd["ego_pose_token"])
    return np.array(ego["translation"], dtype=np.float64), Quaternion(ego["rotation"])


def global_to_ego(points_global, ego_t, ego_q):
    """Transform global (x, y, z) points into the ego frame defined by
    (ego_t, ego_q). ego_q is the ego->global rotation, so we apply its inverse.
    Returns array of same shape."""
    points_global = np.asarray(points_global, dtype=np.float64)
    r_inv = ego_q.inverse.rotation_matrix  # global -> ego
    return (r_inv @ (points_global - ego_t).T).T


def derive_command(waypoints, lateral_thresh=2.0):
    """Coarse high-level command from the final lateral offset (nuScenes/VAD
    style). y > 0 is left in the ego frame."""
    final_y = float(waypoints[-1][1])
    if final_y > lateral_thresh:
        return "left"
    if final_y < -lateral_thresh:
        return "right"
    return "straight"


# ---------------------------------------------------------------------------
# core
# ---------------------------------------------------------------------------
def ordered_samples(nusc, scene):
    """All keyframe sample dicts of a scene, in temporal order."""
    out = []
    tok = scene["first_sample_token"]
    while tok:
        s = nusc.get("sample", tok)
        out.append(s)
        tok = s["next"]
    return out


def build_records(nusc, scene_names, cameras, n_future, future_stride, ref_sensor):
    records = []
    n_skipped = 0
    name_to_scene = {sc["name"]: sc for sc in nusc.scene}

    for name in scene_names:
        if name not in name_to_scene:
            continue
        scene = name_to_scene[name]
        samples = ordered_samples(nusc, scene)
        poses = [sample_ego_pose(nusc, s, ref_sensor) for s in samples]

        # last index that still has n_future waypoints ahead of it
        last_valid = len(samples) - n_future * future_stride
        for i in range(max(0, last_valid)):
            ego_t, ego_q = poses[i]

            fut_global = np.array(
                [poses[i + (k + 1) * future_stride][0] for k in range(n_future)]
            )  # (n_future, 3) global translations
            wp = global_to_ego(fut_global, ego_t, ego_q)[:, :2]  # (n_future, 2)

            img_paths = []
            ok = True
            for cam in cameras:
                if cam not in samples[i]["data"]:
                    ok = False
                    break
                sd = nusc.get("sample_data", samples[i]["data"][cam])
                p = os.path.abspath(os.path.join(nusc.dataroot, sd["filename"]))
                # Metadata covers all 850 scenes, but a partial keyframe-blob
                # download only has some on disk. Skip samples whose image file is
                # missing so records never reference files training would fail to
                # open. With a full download nothing is skipped.
                if not os.path.exists(p):
                    ok = False
                    break
                img_paths.append(p)
            if not ok:
                n_skipped += 1
                continue

            records.append(
                {
                    "sample_token": samples[i]["token"],
                    "scene": name,
                    "images": img_paths,
                    "waypoints": wp.astype(np.float32).round(4).tolist(),
                    "command": derive_command(wp),
                }
            )

    return records, n_skipped


def compute_norm_stats(records, pad=1.25):
    """Derive the waypoint normalization range used by norm_action.

    IMPORTANT: we deliberately do NOT use the raw per-split min/max. The
    discretization range decides what the action tokenizer is able to represent
    at all, and a small or biased split will silently make valid manoeuvres
    unrepresentable. Concretely, nuScenes v1.0-mini's 8 train scenes contain
    zero right turns (y_min ~= -1.1), while its val scenes are ~29% right turns
    (y down to -7.7): normalizing with the train min/max would clip every val
    right turn to the lower bound.

    So:
      * y (lateral) is forced SYMMETRIC -- driving is left/right symmetric even
        when a given split happens not to be.
      * x (forward) keeps 0 as its floor (reverse is not in these records) and
        is padded on top.
    Raw statistics are still reported under "raw_stats" for inspection.
    """
    all_wp = np.array([wp for r in records for wp in r["waypoints"]], dtype=np.float64)
    raw_min, raw_max = all_wp.min(axis=0), all_wp.max(axis=0)

    x_lo = min(0.0, float(raw_min[0])) - 1.0
    x_hi = float(raw_max[0]) * pad
    y_abs = max(abs(float(raw_min[1])), abs(float(raw_max[1]))) * pad

    return {
        # what item_processor.norm_action_nuscenes consumes
        "min": [round(x_lo, 4), round(-y_abs, 4)],
        "max": [round(x_hi, 4), round(y_abs, 4)],
        "raw_stats": {
            "min": raw_min.round(4).tolist(),
            "max": raw_max.round(4).tolist(),
            "mean": all_wp.mean(axis=0).round(4).tolist(),
            "std": all_wp.std(axis=0).round(4).tolist(),
            "p1": np.percentile(all_wp, 1, axis=0).round(4).tolist(),
            "p99": np.percentile(all_wp, 99, axis=0).round(4).tolist(),
        },
        "n_waypoint_vectors": int(all_wp.shape[0]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataroot", required=True, help="nuScenes root (contains samples/, sweeps/, v1.0-mini/)")
    ap.add_argument("--version", default="v1.0-mini")
    ap.add_argument("--out_dir", default="./data/nuscenes_records")
    ap.add_argument("--cameras", nargs="+", default=["CAM_FRONT"],
                    help="camera channels used as input, in order")
    ap.add_argument("--n_future", type=int, default=6, help="number of future waypoints")
    ap.add_argument("--future_stride", type=int, default=1,
                    help="keyframe stride between waypoints (1 keyframe = 0.5 s)")
    ap.add_argument("--ref_sensor", default="LIDAR_TOP", help="sensor whose ego_pose defines the frame")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=True)

    # official scene splits (mini_train: 8 scenes, mini_val: 2 scenes)
    splits = create_splits_scenes()
    if args.version == "v1.0-mini":
        split_map = {"train": "mini_train", "val": "mini_val"}
    else:
        split_map = {"train": "train", "val": "val"}

    norm_stats = None
    for out_split, nusc_split in split_map.items():
        scene_names = splits[nusc_split]
        records, n_skipped = build_records(
            nusc, scene_names, args.cameras, args.n_future, args.future_stride, args.ref_sensor
        )
        out_path = os.path.join(args.out_dir, f"nuscenes_{args.version}_{out_split}.json")
        with open(out_path, "w") as f:
            json.dump(records, f)
        print(f"[{out_split}] scenes={len(scene_names)}  records={len(records)}  "
              f"skipped={n_skipped}  -> {out_path}")

        # Manoeuvre balance: a split with (near-)zero turns in one direction is
        # a red flag -- see the note in compute_norm_stats().
        if records:
            counts = Counter(r["command"] for r in records)
            wp = np.array([r["waypoints"] for r in records])
            print(f"    command={dict(counts)}  "
                  f"x=[{wp[:, :, 0].min():.2f}, {wp[:, :, 0].max():.2f}]  "
                  f"y=[{wp[:, :, 1].min():.2f}, {wp[:, :, 1].max():.2f}]")
            for direction in ("left", "right"):
                if counts.get(direction, 0) == 0:
                    print(f"    WARNING: no '{direction}' turns in this split "
                          f"-- do not fit the normalization range to it")

        if out_split == "train":
            norm_stats = compute_norm_stats(records)

    # normalization stats (from the train split) for item_processor.norm_action
    norm_stats["n_future"] = args.n_future
    norm_stats["action_dim"] = 2
    norm_stats["cameras"] = args.cameras
    norm_path = os.path.join(args.out_dir, "nuscenes_norm.json")
    with open(norm_path, "w") as f:
        json.dump(norm_stats, f, indent=2)
    print("\n=== norm stats (paste into item_processor.norm_action_nuscenes) ===")
    print(json.dumps(norm_stats, indent=2))
    print(f"saved -> {norm_path}")


if __name__ == "__main__":
    main()
