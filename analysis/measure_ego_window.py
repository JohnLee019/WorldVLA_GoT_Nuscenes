"""Is a SHORTER causal window a better ego-status estimator? (CPU, no GPU)

`data/preprocess_nuscenes.ego_state_causal` estimates v0 as the displacement over
the past ~0.5 s. That is the AVERAGE speed over [t-0.5, t], but the quantity the
model needs to match is the average over [t, t+0.5]. Under roughly linear
acceleration those differ by about a*0.5 -- a systematic LAG, not noise. An
instantaneous v(t) should remove half of it.

This measures that directly, by sweeping the window length against the same
target sec.1.15 used. It reads only nuScenes METADATA (sweep ego_poses), which is
already on disk, so it answers "is the CAN bus expansion worth downloading?"
before downloading it:

  * a short window clearly wins  -> CAN bus (measured 50 Hz speed) is worth it,
                                    and preprocess should use a shorter target_dt
  * no window beats ~0.5 s       -> the residual is future acceleration, which no
                                    instrument can observe. Do not download.

The bias/noise split in the output is the part to read: a window that lowers MAE
by cutting BIAS is measuring the same thing better, while one that only lowers
noise is averaging more. Only the first translates into the sec.1.15 sigma.

Usage
-----
  python analysis/measure_ego_window.py --dataroot ../data/nuscenes \
      --version v1.0-trainval --scenes 150
"""
import argparse
import sys

import numpy as np

sys.path.insert(0, ".")
from data.preprocess_nuscenes import (  # noqa: E402
    _pose_of_sd, _walk_back, global_to_ego, ordered_samples,
)

WINDOWS = [0.05, 0.10, 0.25, 0.50, 1.00]


def v_lon_over(nusc, sample, target_dt, ref_sensor):
    """Causal longitudinal speed from a window of about `target_dt` seconds."""
    sd0 = nusc.get("sample_data", sample["data"][ref_sensor])
    t0, q0 = _pose_of_sd(nusc, sd0)
    back = _walk_back(nusc, sd0, target_dt=target_dt, min_dt=min(0.04, target_dt * 0.8))
    if back is None:
        return None
    sd1, dt = back
    t1, _ = _pose_of_sd(nusc, sd1)
    return float((-global_to_ego(t1[None, :], t0, q0)[0][:2] / dt)[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataroot", required=True)
    ap.add_argument("--version", default="v1.0-trainval")
    ap.add_argument("--split", default="val")
    ap.add_argument("--ref_sensor", default="LIDAR_TOP")
    ap.add_argument("--scenes", type=int, default=150, help="cap for a quick pass")
    args = ap.parse_args()

    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.splits import create_splits_scenes

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=True)
    split = args.split if args.version != "v1.0-mini" else f"mini_{args.split}"
    names = create_splits_scenes()[split][: args.scenes]
    by_name = {sc["name"]: sc for sc in nusc.scene}

    # target = the GT first step, i.e. mean speed over [t, t+0.5] -- exactly what
    # sec.1.15 rescaled trajectories to match.
    rows = {w: [] for w in WINDOWS}
    targets = []
    for name in names:
        if name not in by_name:
            continue
        samples = ordered_samples(nusc, by_name[name])
        poses = [_pose_of_sd(nusc, nusc.get("sample_data", s["data"][args.ref_sensor]))
                 for s in samples]
        for i in range(len(samples) - 1):
            t0, q0 = poses[i]
            t1, _ = poses[i + 1]
            dt = (samples[i + 1]["timestamp"] - samples[i]["timestamp"]) / 1e6
            if not 0.45 < dt < 0.55:
                continue
            tgt = float(global_to_ego(t1[None, :], t0, q0)[0][0] / dt)
            vs = {w: v_lon_over(nusc, samples[i], w, args.ref_sensor) for w in WINDOWS}
            if any(v is None for v in vs.values()):
                continue
            targets.append(tgt)
            for w in WINDOWS:
                rows[w].append(vs[w])

    targets = np.array(targets)
    if len(targets) == 0:
        raise SystemExit("no usable keyframe pairs -- check --dataroot / --split")
    print(f"\n{len(targets)} keyframes over {len(names)} scenes  "
          f"(target = GT first-step speed, mean {targets.mean():.3f} m/s)\n")

    print(f"{'window':>8}{'MAE m/s':>10}{'bias':>9}{'std(err)':>10}"
          f"{'MAE @0.5s':>12}{'vs 0.50s':>10}")
    base = None
    for w in WINDOWS:
        err = np.array(rows[w]) - targets
        mae = float(np.abs(err).mean())
        # the sec.1.15 sigma lives in metres per 0.5 s, not m/s
        mae_step = mae * 0.5
        if w == 0.50:
            base = mae
        print(f"{w:>8.2f}{mae:>10.4f}{err.mean():>+9.4f}{err.std():>10.4f}"
              f"{mae_step:>12.4f}", end="")
        print(f"{(mae - base) / base:>+9.1%}" if base else "")
    print("\nsec.1.15 used the 0.50 s window: MAE 0.1560 m per 0.5 s "
          "(= 0.3120 m/s), sigma 0.2377 m")
    print("Read BIAS, not just MAE: a shorter window should cut the systematic lag.")
    print("If nothing beats 0.50 s the residual is future acceleration -- "
          "unobservable, so the CAN bus expansion buys nothing.")


if __name__ == "__main__":
    main()
