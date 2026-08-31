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
# ego status -- CAUSAL ONLY
# ---------------------------------------------------------------------------
# ★★The single rule of this section: every pose read here is at a timestamp
# STRICTLY BEFORE the keyframe. If a future pose ever leaks in, the L2 gain is
# fabricated and the whole arm is worthless. The `prev` chain of a sample_data
# only ever walks backwards, which is why it is used instead of indexing
# `poses[i+1]`.
#
# ★Window lengths are MEASURED, not assumed (`analysis/measure_ego_window.py`,
# 5,170 keyframes / 150 scenes). Error against the GT first step, m per 0.5 s:
#
#     0.05 s  0.0792     <- floor; one LIDAR sweep, effectively instantaneous
#     0.10 s  0.0805        only 1.8% off the floor, but averages two sweeps
#     0.50 s  0.1209
#     1.00 s  0.1696
#
# So velocity wants a SHORT window. Two consequences worth recording:
#   * the gain is in std(err) (0.3384 -> 0.2202 m/s), not bias (-0.014 -> -0.007).
#     A short window wins because it sits closer in time to the target, not
#     because it corrects a lag. The residual is genuine future acceleration.
#   * 0.05 and 0.10 being within 1.8% means pose differencing has bottomed out.
#     That is why the CAN bus expansion (measured 50 Hz speed) was NOT downloaded:
#     it can only improve an instrument that is already at its floor.
#
# Acceleration and yaw rate get a LONGER window on purpose: differencing over
# 0.1 s is noise-dominated for both.
V_WINDOW = 0.10    # velocity: short, near-instantaneous
A_WINDOW = 0.50    # accel: difference of two short-window velocities this far apart
YAW_WINDOW = 0.50  # yaw rate: rotation is slow, a short baseline is noise

STATE_DIM = 4  # [v_x, v_y, a_x, yaw_rate]
STATE_KEYS = ["v_x", "v_y", "a_x", "yaw_rate"]


def _pose_of_sd(nusc, sd):
    ego = nusc.get("ego_pose", sd["ego_pose_token"])
    return np.array(ego["translation"], dtype=np.float64), Quaternion(ego["rotation"])


def _walk_back(nusc, sd, target_dt=0.5, min_dt=0.2):
    """Follow `prev` (strictly into the past) until ~`target_dt` s have elapsed.

    Returns (sample_data, dt_seconds) or None when the chain is too short --
    which happens at the very start of a scene. Sweeps carry ego_pose in the
    METADATA, so this works even on a keyframes-only blob download.
    """
    t0 = sd["timestamp"]
    cur, best = sd, None
    while cur["prev"]:
        cur = nusc.get("sample_data", cur["prev"])
        dt = (t0 - cur["timestamp"]) / 1e6
        # The `prev` chain is supposed to be monotonically into the past. Assert
        # it rather than trust it: a non-positive dt here would mean a future
        # pose is about to be used as "ego status", which is the one failure
        # mode that silently invalidates every number the arm produces.
        assert dt > 0, f"non-causal prev link: dt={dt}s at {cur['token']}"
        best = (cur, dt)
        if dt >= target_dt:
            break
    if best is None or best[1] < min_dt:
        return None
    return best


def _wrap_pi(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def ego_state_causal(nusc, sample, ref_sensor="LIDAR_TOP"):
    """[v_x, v_y, a_x, yaw_rate] in the CURRENT ego frame, from PAST poses only.

    Returns (state, valid). `valid` is 0 at the start of a scene where the
    `prev` chain cannot reach back far enough; the record is still emitted (with
    a zero state) so that the record population -- and therefore the frozen
    600-record eval set -- is byte-identical to the stateless build.
    """
    sd0 = nusc.get("sample_data", sample["data"][ref_sensor])
    t0, q0 = _pose_of_sd(nusc, sd0)

    def velocity_at(sd):
        """Short-window velocity in that sample_data's own ego frame, or None."""
        ta, qa = _pose_of_sd(nusc, sd)
        back = _walk_back(nusc, sd, target_dt=V_WINDOW, min_dt=0.04)
        if back is None:
            return None
        sd_prev, dt = back
        t_prev, _ = _pose_of_sd(nusc, sd_prev)
        # global_to_ego gives where the PAST pose sits relative to now, so negate
        # it to get the travelled displacement, then divide by the real elapsed dt.
        return -global_to_ego(t_prev[None, :], ta, qa)[0][:2] / dt

    v = velocity_at(sd0)
    if v is None:
        return [0.0] * STATE_DIM, 0

    # Yaw rate over its own, longer window.
    yaw_rate = 0.0
    back_yaw = _walk_back(nusc, sd0, target_dt=YAW_WINDOW)
    if back_yaw is not None:
        sd_y, dt_y = back_yaw
        _, q_y = _pose_of_sd(nusc, sd_y)
        yaw_rate = _wrap_pi(q0.yaw_pitch_roll[0] - q_y.yaw_pitch_roll[0]) / dt_y

    # Longitudinal accel = how the short-window velocity changed over A_WINDOW.
    # Both velocities use the SAME estimator, so its bias cancels in the difference.
    a_x = 0.0
    back_a = _walk_back(nusc, sd0, target_dt=A_WINDOW)
    if back_a is not None:
        sd_a, dt_a = back_a
        v_then = velocity_at(sd_a)
        if v_then is not None:
            a_x = float(v[0] - v_then[0]) / dt_a

    return [round(float(x), 4) for x in (v[0], v[1], a_x, yaw_rate)], 1


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

            state, state_valid = ego_state_causal(nusc, samples[i], ref_sensor)

            records.append(
                {
                    "sample_token": samples[i]["token"],
                    "scene": name,
                    "images": img_paths,
                    "waypoints": wp.astype(np.float32).round(4).tolist(),
                    "command": derive_command(wp),
                    "state": state,
                    "state_valid": state_valid,
                }
            )

    return records, n_skipped


def compute_state_norm_stats(records, pad=1.25):
    """Derive the ego-status normalization range used by norm_state.

    ★This is NOT optional plumbing. `FlexARItemProcessor_Action` inherits a
    `norm_state` whose min/max are hardcoded LIBERO robot-arm values; feeding
    driving speeds through it clips every record to +-1 and produces a null that
    looks like "ego status does not help". The nuScenes item processor overrides
    `norm_state` with what this function writes.

    Same convention as the waypoints: signed quantities are forced SYMMETRIC
    (driving is left/right symmetric even when a split is not) and v_x keeps 0
    as its floor. Ranges are padded so a faster val record is representable
    rather than silently clipped.
    """
    valid = [r["state"] for r in records if r.get("state_valid")]
    if not valid:
        return {}
    st = np.array(valid, dtype=np.float64)
    lo, hi = st.min(axis=0), st.max(axis=0)

    smin, smax = [], []
    for j in range(st.shape[1]):
        if STATE_KEYS[j] == "v_x":                       # forward speed: 0 floor
            smin.append(min(0.0, float(lo[j])) - 1.0)
            smax.append(float(hi[j]) * pad)
        else:                                            # signed: symmetric
            a = max(abs(float(lo[j])), abs(float(hi[j]))) * pad
            smin.append(-a)
            smax.append(a)

    return {
        # what FlexARItemProcessor_Action_NuScenes.norm_state consumes
        "state_keys": STATE_KEYS,
        "state_min": [round(v, 4) for v in smin],
        "state_max": [round(v, 4) for v in smax],
        "state_raw_stats": {
            "min": lo.round(4).tolist(),
            "max": hi.round(4).tolist(),
            "mean": st.mean(axis=0).round(4).tolist(),
            "std": st.std(axis=0).round(4).tolist(),
        },
        "n_state_invalid": int(sum(1 for r in records if not r.get("state_valid"))),
    }


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
        **compute_state_norm_stats(records, pad=pad),
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
    ap.add_argument("--inherit_wp_norm", default=None, metavar="NORM_JSON",
                    help="take the waypoint min/max from an existing norm json instead of "
                         "fitting them here. Use this whenever the records feed a checkpoint "
                         "that already exists: the action grid must not move while some other "
                         "variable (e.g. ego status) is being changed. State ranges are still "
                         "fitted from this build.")
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
    records_train = []
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

            st = np.array([r["state"] for r in records if r["state_valid"]])
            n_bad = sum(1 for r in records if not r["state_valid"])
            print(f"    state: valid={len(st)}/{len(records)} invalid={n_bad} "
                  f"(scene starts, emitted with a zero state -- records are NOT dropped)")
            if len(st):
                for j, k in enumerate(STATE_KEYS):
                    print(f"      {k:<9} mean {st[:, j].mean():+8.4f}  "
                          f"[{st[:, j].min():+8.4f}, {st[:, j].max():+8.4f}]")
                # v_x is the axis the whole arm rests on: it should look like a
                # speed in m/s, not a per-0.5s displacement and not a garbage
                # scale. nuScenes urban driving sits around 4-6 m/s on average.
                mean_v = float(st[:, 0].mean())
                if not 1.0 < mean_v < 15.0:
                    print(f"    WARNING: mean v_x = {mean_v:.3f} m/s is outside the "
                          f"plausible 1-15 m/s band -- check the dt units")

        if out_split == "train":
            norm_stats = compute_norm_stats(records)
            records_train = records

    # normalization stats (from the train split) for item_processor.norm_action
    if args.inherit_wp_norm:
        # ★Adding a channel must not also move the action grid underneath it.
        #
        # An existing checkpoint decodes with the waypoint min/max it was trained
        # on, so a run that changes BOTH the input (ego status) and the 255-bin
        # grid cannot attribute its result to either. Inheriting keeps the new arm
        # one variable away from the incumbent.
        #
        # This is also how the shipped `data/nuscenes_records/nuscenes_norm.json`
        # stays reproducible: it was fitted on v1.0-mini (its raw y_min of -1.1484
        # is mini's "no right turns" signature, see compute_norm_stats) and never
        # regenerated for trainval. The forced-symmetric y rule kept that harmless
        # -- +-15.1637 still covers trainval's [-11.11, 13.32] -- and its 0.2253 m
        # bins are FINER than a trainval-fitted grid's 0.2855 m, so re-fitting
        # would coarsen the quantization (RMS 0.065 -> 0.082 m) for no gain.
        with open(args.inherit_wp_norm) as f:
            inherited = json.load(f)
        fitted_min, fitted_max = norm_stats["min"], norm_stats["max"]
        norm_stats["min"], norm_stats["max"] = inherited["min"], inherited["max"]
        norm_stats["wp_norm_inherited_from"] = args.inherit_wp_norm
        norm_stats["wp_norm_fitted_here"] = {"min": fitted_min, "max": fitted_max}
        print(f"\n[wp norm] inherited from {args.inherit_wp_norm}: "
              f"min={inherited['min']} max={inherited['max']}")
        print(f"[wp norm] (this split would have fitted min={fitted_min} max={fitted_max})")

        # Inheriting a range that does not cover this split silently clips
        # manoeuvres, so say how much rather than let it pass unnoticed.
        all_wp = np.array([wp for r in records_train for wp in r["waypoints"]])
        lo = np.array(inherited["min"]); hi = np.array(inherited["max"])
        n_clip = int(((all_wp < lo) | (all_wp > hi)).any(axis=1).sum())
        print(f"[wp norm] {n_clip}/{len(all_wp)} train waypoint vectors "
              f"({100*n_clip/len(all_wp):.4f}%) fall outside the inherited range "
              f"and will be clipped")

    norm_stats["n_future"] = args.n_future
    norm_stats["action_dim"] = 2
    norm_stats["cameras"] = args.cameras
    norm_stats["version"] = args.version
    norm_path = os.path.join(args.out_dir, "nuscenes_norm.json")

    # ★★The norm filename carries no version, but the RECORD filenames do -- so a
    # `--version v1.0-mini` run into a directory holding trainval records leaves
    # the records untouched and silently replaces the action grid underneath
    # them. Both defaults point that way (`--version v1.0-mini`,
    # `--out_dir ./data/nuscenes_records`), so one argument-less run is enough.
    #
    # This actually happened and cost a session to find: mini's grid (y +-15.1637)
    # replaced trainval's (y +-16.6451), and every later eval decoded on the wrong
    # grid. It surfaced as the incumbent scoring avgL2@3s 3.7889 instead of its
    # frozen 3.5557 -- a plausible-looking number, not a crash. The tell was the
    # quantisation step of the near-constant lateral coordinate: 0.119 (= 2*15.1637/255)
    # where the stored run had 0.131 (= 2*16.6451/255).
    if os.path.exists(norm_path):
        try:
            with open(norm_path) as f:
                prev = json.load(f)
        except (json.JSONDecodeError, OSError):
            prev = {}
        prev_version = prev.get("version")
        if prev_version is not None and prev_version != args.version:
            raise SystemExit(
                f"REFUSING to overwrite {norm_path}\n"
                f"  it was fitted on '{prev_version}', this run is '{args.version}'\n"
                f"  existing: min={prev.get('min')} max={prev.get('max')}\n"
                f"  this run: min={norm_stats['min']} max={norm_stats['max']}\n"
                f"Checkpoints trained against the existing grid decode with it. Write\n"
                f"somewhere else with --out_dir, or delete the file if you really mean\n"
                f"to re-fit (and then re-evaluate everything that used it)."
            )
        if prev_version is None and prev.get("min") is not None \
                and list(prev["min"]) != list(norm_stats["min"]):
            # Pre-dates the version field, so we cannot tell which split fitted it.
            print(f"\n[warn] {norm_path} exists with a DIFFERENT grid and no version tag:\n"
                  f"       existing min={prev.get('min')} max={prev.get('max')}\n"
                  f"       this run min={norm_stats['min']} max={norm_stats['max']}\n"
                  f"       overwriting. If a checkpoint was trained on the existing grid,\n"
                  f"       its numbers will move -- check against a frozen result.")

    with open(norm_path, "w") as f:
        json.dump(norm_stats, f, indent=2)
    print("\n=== norm stats (paste into item_processor.norm_action_nuscenes) ===")
    print(json.dumps(norm_stats, indent=2))
    print(f"saved -> {norm_path}")


if __name__ == "__main__":
    main()
