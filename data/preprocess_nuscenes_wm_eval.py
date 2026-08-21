"""
Preprocess nuScenes into WORLD-MODEL-EVAL frame records.

This is the extra data the offline WM-image eval needs on top of the existing
planning records (PROJECT_HANDOFF §7.4). The planning records
(preprocess_nuscenes.py) already give, per keyframe j: the t0 CAM_FRONT image,
the 6 GT waypoints (ego_j frame) and the command -- everything the planner and
the GT-action floor need. The ONE thing missing is the *real future frames* to
compare the WM's predictions against.

The eval is teacher-forced per segment (§2③②): for each segment it feeds the WM
the REAL frame at the segment start (so per-segment errors do not compound) and
compares the WM's 1 s-ahead prediction to the REAL frame at the segment end. With
n_segments=3, segment_len=2 waypoints (= 1 s at future_stride=1), the segment
boundaries fall on keyframe offsets 0, 2, 4, 6 from j -- i.e. 0 s, 1 s, 2 s, 3 s.

So each emitted record is just the aligned chain of CAM_FRONT frame paths at
those boundaries, keyed by sample_token so the eval can join it to the planning
record:

    {
        "sample_token": str,          # the t0 keyframe (== planning record key)
        "scene": str,
        "frames": [p@j, p@j+2, p@j+4, p@j+6],   # boundary CAM_FRONT paths, in order
        "frame_offsets": [0, 2, 4, 6],           # keyframe offsets, for clarity
    }

frames[s]   is the teacher-forcing anchor (real frame) at the start of segment s.
frames[s+1] is the real target the WM prediction for segment s is scored against.

No geometry/actions are stored here on purpose: both the plan's and the GT's
per-segment actions are derived in the eval from their (6,2) t0-frame
trajectories with a single shared re-basing helper, so plan and GT use an
identical action convention (see eval_wm_image_nuscenes.seg_local_actions). Ego
poses are therefore not needed.

Usage
-----
    python -m data.preprocess_nuscenes_wm_eval \
        --dataroot /data/nuscenes \
        --version v1.0-trainval \
        --out_dir ./data/nuscenes_wm_records \
        --split val \
        --n_segments 3 --segment_len 2 --future_stride 1

Emits nuscenes_wm_eval_<version>_<split>.json. Records whose t0 CAM_FRONT is
missing on disk, or which do not have all boundary frames downloaded, are
skipped (a partial keyframe-blob download only has some frames) -- so the eval
never references a file it cannot open. With a full download nothing is skipped.
"""

import argparse
import json
import os

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes

from data.preprocess_nuscenes import ordered_samples

CAM = "CAM_FRONT"


def build_eval_frame_records(nusc, scene_names, boundary_offsets):
    """One record per keyframe that has every boundary frame ahead of it.

    boundary_offsets: sorted keyframe offsets from j to collect, e.g. [0,2,4,6].
    """
    records = []
    n_skipped = 0
    max_offset = boundary_offsets[-1]
    name_to_scene = {sc["name"]: sc for sc in nusc.scene}

    for name in scene_names:
        if name not in name_to_scene:
            continue
        samples = ordered_samples(nusc, name_to_scene[name])

        last_valid = len(samples) - max_offset      # need j+max_offset to exist
        for j in range(max(0, last_valid)):
            frame_paths = []
            ok = True
            for off in boundary_offsets:
                s = samples[j + off]
                if CAM not in s["data"]:
                    ok = False
                    break
                sd = nusc.get("sample_data", s["data"][CAM])
                p = os.path.abspath(os.path.join(nusc.dataroot, sd["filename"]))
                # partial download: need every boundary frame on disk
                if not os.path.exists(p):
                    ok = False
                    break
                frame_paths.append(p)
            if not ok:
                n_skipped += 1
                continue

            records.append({
                "sample_token": samples[j]["token"],
                "scene": name,
                "frames": frame_paths,
                "frame_offsets": list(boundary_offsets),
            })

    return records, n_skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataroot", required=True, help="nuScenes root (samples/, sweeps/, v1.0-*/)")
    ap.add_argument("--version", default="v1.0-trainval")
    ap.add_argument("--out_dir", default="./data/nuscenes_wm_records")
    ap.add_argument("--split", default="val", choices=["train", "val"],
                    help="which planning split to build eval frames for (val is the eval split)")
    ap.add_argument("--n_segments", type=int, default=3,
                    help="GoT segments (must match the eval); boundaries = n_segments+1 frames")
    ap.add_argument("--segment_len", type=int, default=2,
                    help="waypoints per segment (== keyframes per segment at future_stride=1)")
    ap.add_argument("--future_stride", type=int, default=1,
                    help="keyframe stride per waypoint (1 keyframe = 0.5 s)")
    ap.add_argument("--ref_sensor", default="LIDAR_TOP",
                    help="kept for parity with the other preprocessors (unused: frames only)")
    args = ap.parse_args()

    # segment boundaries in keyframe offsets: 0, seg, 2*seg, ... n_segments*seg
    step = args.segment_len * args.future_stride
    boundary_offsets = [s * step for s in range(args.n_segments + 1)]

    os.makedirs(args.out_dir, exist_ok=True)
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=True)

    splits = create_splits_scenes()
    if args.version == "v1.0-mini":
        nusc_split = {"train": "mini_train", "val": "mini_val"}[args.split]
    else:
        nusc_split = args.split
    scene_names = splits[nusc_split]

    records, n_skipped = build_eval_frame_records(nusc, scene_names, boundary_offsets)
    out_path = os.path.join(args.out_dir, f"nuscenes_wm_eval_{args.version}_{args.split}.json")
    with open(out_path, "w") as f:
        json.dump(records, f)

    horizon_s = boundary_offsets[-1] * 0.5 * args.future_stride / max(args.future_stride, 1)
    print(f"[{args.split}] scenes={len(scene_names)}  records={len(records)}  "
          f"skipped={n_skipped}  -> {out_path}")
    print(f"    boundary keyframe offsets={boundary_offsets} "
          f"(segments={args.n_segments} x {args.segment_len}wp, "
          f"horizon={boundary_offsets[-1]*0.5:.1f}s)")


if __name__ == "__main__":
    main()
