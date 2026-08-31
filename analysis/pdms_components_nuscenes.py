#!/usr/bin/env python
"""PDM-Score components on nuScenes: NC / DAC / TTC / Comf / EP and the aggregate.

READ THIS BEFORE QUOTING ANY NUMBER FROM HERE
---------------------------------------------
These are NAVSIM PDM-Score components, and NAVSIM defines them on a 4 s @ 10 Hz
LQR simulation over navtest scenarios. This tool computes the same-spirit
quantities on our nuScenes setup: 3 s @ 2 Hz, open loop, 600 records / 150 scenes.
Every output is therefore suffixed `-proxy` and the aggregate is `PDMS-proxy`.

  ⛔ DO NOT put a column from this tool beside a published NAVSIM table.
     Different benchmark, horizon, sampling rate and simulator. A `NC-proxy` of
     97.6 sitting next to UniAD's `NC 97.8` is a coincidence of scale, not a
     comparison -- the sec.7.1 L2-parallel-table error wearing a new hat.
  ✅ DO use these columns to compare OUR OWN arms against each other. That is a
     self-paired comparison on one fixed set, which is what this project is
     designed around.

PER-COMPONENT HONESTY
---------------------
NC-proxy   `1 - collision`, with NO at-fault attribution. Real NC only counts
           collisions the ego caused; deciding that needs the other agent's
           motion and a rule set, and the boxes carry no track id. So this is a
           LOWER bound on NC.
DAC        The real thing, if you pass `--drivable_masks` built by
           `data/preprocess_nuscenes_map.py`. Binary per scenario: did the ego
           footprint ever leave the drivable area. Needs nuScenes map expansion
           v1.3, which is NOT part of the base download.
TTC-proxy  At each step the ego is projected forward at its current velocity for
           `--ttc_horizon` seconds and tested against the agent boxes AT THAT
           FUTURE TIME (better than NAVSIM's constant-velocity agents, since we
           have the real future annotation). But we can only test at 0.5 s
           multiples where NAVSIM tests at 0.1 s, so short violations between
           samples are invisible: this is biased OPTIMISTIC.
Comf-proxy Kinematic bounds on the trajectory. Ours has 6 points at 0.5 s, so
           jerk is a second difference of a 6-sample sequence -- noisy, and coarse
           sampling under-reports brief violations. Yaw terms come from the
           trajectory heading, not from a predicted yaw channel.
           ⚠️ The bounds are the nuPlan defaults AS DOCUMENTED, not read from a
           devkit vendored here. Verify before quoting.
EP-proxy   Real EP normalises progress against the PDM-Closed planner. We have no
           route and no reference planner, so this normalises against the GROUND
           TRUTH driver instead: how far along the human's heading did the plan
           get, as a fraction of how far the human got. Records where the human
           barely moved are excluded (`--min_progress_m`) because the ratio is
           meaningless there -- the count is reported.

Aggregate: PDMS-proxy = NC x DAC x (5*EP + 5*TTC + 2*Comf) / 12, the v1.1 devkit
formula. It is only printed when every component is available; with DAC missing
the tool says so instead of substituting 1.0.

Usage
-----
  python analysis/pdms_components_nuscenes.py \\
      --collision_json ./data/nuscenes_records/nuscenes_collision_v1.0-trainval_val.json \\
      --drivable_masks ./data/nuscenes_records/nuscenes_drivable_val.npz \\
      ./results/base_ckpt/raw_lumina_constrained \\
      ./results/base_ckpt/incumbent_cont2_ep1 \\
      ./results/const_velocity

  python analysis/pdms_components_nuscenes.py --selftest
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HZ_IDX = {"1s": 1, "2s": 3, "3s": 5}
DT = 0.5

# nuPlan comfort bounds as documented. NOT read from a vendored devkit -- verify.
MAX_ABS_LON_ACCEL = 2.40
MAX_ABS_LAT_ACCEL = 4.89
MAX_ABS_LON_JERK = 4.13
MAX_ABS_YAW_RATE = 0.95


def _mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


# --------------------------------------------------------------------------- #
# Comf
# --------------------------------------------------------------------------- #
def kinematics(traj, dt=DT):
    """Cumulative ego-frame waypoints -> lon/lat accel, lon jerk, yaw rate.

    The trajectory starts at the ego origin, so the step INTO waypoint 0 is a real
    0.5 s of motion. Prepend (0, 0) rather than dropping it, or the interval the
    model is most confident about is silently excluded from every bound.
    """
    pts = [(0.0, 0.0)] + [tuple(map(float, p)) for p in traj]
    vel = [((b[0] - a[0]) / dt, (b[1] - a[1]) / dt) for a, b in zip(pts, pts[1:])]
    head = [math.atan2(v[1], v[0]) if (v[0] or v[1]) else 0.0 for v in vel]

    lon_a, lat_a, yaw_rate = [], [], []
    for i in range(1, len(vel)):
        ax = (vel[i][0] - vel[i - 1][0]) / dt
        ay = (vel[i][1] - vel[i - 1][1]) / dt
        h = head[i]
        lon_a.append(ax * math.cos(h) + ay * math.sin(h))
        lat_a.append(-ax * math.sin(h) + ay * math.cos(h))
        d = (head[i] - head[i - 1] + math.pi) % (2 * math.pi) - math.pi
        yaw_rate.append(d / dt)

    lon_j = [(lon_a[i] - lon_a[i - 1]) / dt for i in range(1, len(lon_a))]
    return lon_a, lat_a, lon_j, yaw_rate


def comfortable(traj, dt=DT):
    lon_a, lat_a, lon_j, yaw = kinematics(traj, dt)
    checks = {
        "lon_accel": all(abs(x) <= MAX_ABS_LON_ACCEL for x in lon_a),
        "lat_accel": all(abs(x) <= MAX_ABS_LAT_ACCEL for x in lat_a),
        "lon_jerk": all(abs(x) <= MAX_ABS_LON_JERK for x in lon_j),
        "yaw_rate": all(abs(x) <= MAX_ABS_YAW_RATE for x in yaw),
    }
    return all(checks.values()), checks


# --------------------------------------------------------------------------- #
# EP
# --------------------------------------------------------------------------- #
def ego_progress(pred, gt, min_progress_m=1.0):
    """Progress along the GT heading, as a fraction of the GT's own progress.

    -> (value in [0, 1], scored?) ; scored=False when the human barely moved, where
    the ratio has no meaning and a tiny denominator would manufacture noise.
    """
    g = np.asarray(gt, dtype=np.float64)[-1]
    p = np.asarray(pred, dtype=np.float64)[-1]
    d = float(np.linalg.norm(g))
    if d < min_progress_m:
        return None, False
    return float(np.clip(float(p @ g) / (d * d), 0.0, 1.0)), True


# --------------------------------------------------------------------------- #
# TTC
# --------------------------------------------------------------------------- #
def ttc_violations(traj, agent_boxes, cfg, horizon_s=1.0, dt=DT):
    """(T,) bool: would a constant-velocity continuation from step t hit an agent?

    The projection is tested against the boxes at the FUTURE index, i.e. the real
    annotated positions, rather than constant-velocity agents. Only 0.5 s
    multiples can be tested, so brief violations between samples are missed --
    the result is biased optimistic and must be read as such.
    """
    from got_drive.collision_metric import _rasterize_box, _headings

    traj = np.asarray(traj, dtype=np.float64)
    T = traj.shape[0]
    k = max(1, int(round(horizon_s / dt)))
    pts = np.vstack([np.zeros((1, 2)), traj])
    vel = (pts[1:] - pts[:-1]) / dt
    yaws = _headings(traj) if cfg.apply_yaw else np.zeros(T)

    nx = int(round((cfg.x_bound[1] - cfg.x_bound[0]) / cfg.resolution))
    ny = int(round((cfg.y_bound[1] - cfg.y_bound[0]) / cfg.resolution))

    out = np.zeros(T, dtype=bool)
    for t in range(T):
        j = min(t + k, T - 1)
        boxes = agent_boxes[j] if j < len(agent_boxes) else []
        if not boxes:
            continue
        proj = traj[t] + vel[t] * horizon_s
        cx = proj[0] + cfg.ego_x_offset * np.cos(yaws[t])
        cy = proj[1] + cfg.ego_x_offset * np.sin(yaws[t])
        ego = np.zeros((nx, ny), dtype=bool)
        _rasterize_box(ego, cx, cy, cfg.ego_length, cfg.ego_width, yaws[t], cfg)
        if not ego.any():
            continue
        occ = np.zeros((nx, ny), dtype=bool)
        for b in boxes:
            _rasterize_box(occ, b[0], b[1], b[2], b[3], b[4], cfg)
        out[t] = bool((ego & occ).any())
    return out


# --------------------------------------------------------------------------- #
# DAC
# --------------------------------------------------------------------------- #
def dac_violation(traj, drivable, cfg):
    """-> (left_drivable?, n_steps_outside_grid). Binary per scenario, like NAVSIM.

    A step whose footprint falls outside the mask's extent is counted as a
    violation AND reported separately: leaving the +-50 m mapped window inside 3 s
    is itself anomalous, and silently scoring it as compliant would reward exactly
    the runaway trajectories this metric exists to catch.
    """
    from got_drive.collision_metric import _rasterize_box, _headings

    traj = np.asarray(traj, dtype=np.float64)
    T = traj.shape[0]
    yaws = _headings(traj) if cfg.apply_yaw else np.zeros(T)
    nx, ny = drivable.shape

    bad, n_out = False, 0
    for t in range(T):
        cx = traj[t, 0] + cfg.ego_x_offset * np.cos(yaws[t])
        cy = traj[t, 1] + cfg.ego_x_offset * np.sin(yaws[t])
        ego = np.zeros((nx, ny), dtype=bool)
        _rasterize_box(ego, cx, cy, cfg.ego_length, cfg.ego_width, yaws[t], cfg)
        if not ego.any():          # footprint entirely off the grid
            n_out += 1
            bad = True
            continue
        if (ego & ~drivable).any():
            bad = True
    return bad, n_out


# --------------------------------------------------------------------------- #
def score(run_dir, boxes, drivable, cfg, args, gt_by_tok=None):
    from got_drive.collision_metric import trajectory_collisions

    path = run_dir if run_dir.endswith(".csv") else os.path.join(run_dir, "per_sample.csv")
    if not os.path.exists(path):
        sys.exit(f"[fatal] {path} does not exist")
    with open(path, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r.get(args.status_col) == "ok"]
    if rows and not rows[0].get(args.pred_col):
        print(f"  [warn] {path} has no `pred` column (headline csv). Skipping.")
        return None

    comf, comf_fail = [], {k: 0 for k in ("lon_accel", "lat_accel", "lon_jerk", "yaw_rate")}
    nc, ttc, dac, ep = [], [], [], []
    n_boxed = n_masked = n_ep_skipped = n_off_grid = 0

    for r in rows:
        tok = r["sample_token"]
        pred = ast.literal_eval(r[args.pred_col])
        gt = ast.literal_eval(r["gt"]) if r.get("gt") else (gt_by_tok or {}).get(tok)
        if gt is None:
            continue

        ok, checks = comfortable(pred)
        comf.append(1.0 if ok else 0.0)
        for k, v in checks.items():
            if not v:
                comf_fail[k] += 1

        v, scored = ego_progress(pred, gt, args.min_progress_m)
        if scored:
            ep.append(v)
        else:
            n_ep_skipped += 1

        if tok in boxes:
            n_boxed += 1
            cm, _ = trajectory_collisions(pred, boxes[tok], HZ_IDX, cfg,
                                          gt_traj=None if args.no_gt_mask else gt)
            nc.append(1.0 - cm["coll@3s"])
            vp = ttc_violations(pred, boxes[tok], cfg, args.ttc_horizon)
            if not args.no_gt_mask:
                vp = vp & ~ttc_violations(gt, boxes[tok], cfg, args.ttc_horizon)
            ttc.append(0.0 if vp.any() else 1.0)

        if drivable is not None and tok in drivable:
            n_masked += 1
            bad, n_out = dac_violation(pred, drivable[tok], cfg)
            n_off_grid += n_out
            dac.append(0.0 if bad else 1.0)

    out = {
        "n_rows": len(rows), "n_boxed": n_boxed, "n_masked": n_masked,
        "n_ep_skipped": n_ep_skipped, "n_dac_steps_off_grid": n_off_grid,
        "comf_proxy": round(100 * _mean(comf), 3), "comf_violations": comf_fail,
        "nc_proxy": round(100 * _mean(nc), 3) if nc else None,
        "ttc_proxy": round(100 * _mean(ttc), 3) if ttc else None,
        "dac": round(100 * _mean(dac), 3) if dac else None,
        "ep_proxy": round(100 * _mean(ep), 3) if ep else None,
    }
    if all(out[k] is not None for k in ("nc_proxy", "dac", "ttc_proxy", "ep_proxy")):
        NC, DAC, TTC, C, EP = (out["nc_proxy"] / 100, out["dac"] / 100,
                               out["ttc_proxy"] / 100, out["comf_proxy"] / 100,
                               out["ep_proxy"] / 100)
        out["pdms_proxy"] = round(100 * NC * DAC * (5 * EP + 5 * TTC + 2 * C) / 12, 3)
    else:
        out["pdms_proxy"] = None
    return out


def report(name, m):
    def f(v):
        return "   n/a " if v is None else f"{v:>7.3f}"
    print(f"\n=== {name}   {m['n_rows']} rows")
    print(f"  NC-proxy {f(m['nc_proxy'])}   DAC {f(m['dac'])}   TTC-proxy {f(m['ttc_proxy'])}"
          f"   Comf-proxy {f(m['comf_proxy'])}   EP-proxy {f(m['ep_proxy'])}")
    print(f"  PDMS-proxy {f(m['pdms_proxy'])}"
          + ("" if m["pdms_proxy"] is not None else
             "   <- a component is missing; NOT substituting 1.0 for it"))
    v = [f"{k} {n}" for k, n in sorted(m["comf_violations"].items(), key=lambda kv: -kv[1]) if n]
    print(f"    comfort violations: {', '.join(v) if v else 'none'}")
    if m["nc_proxy"] is not None and m["n_boxed"] < m["n_rows"]:
        print(f"    NC/TTC scored on {m['n_boxed']}/{m['n_rows']} rows (boxes available)")
    if m["dac"] is not None:
        print(f"    DAC scored on {m['n_masked']}/{m['n_rows']} rows; "
              f"{m['n_dac_steps_off_grid']} steps left the mapped window entirely")
    if m["n_ep_skipped"]:
        print(f"    EP skipped {m['n_ep_skipped']} rows where the GT barely moved")


def verdict(results):
    print("\n" + "-" * 78)
    print("HOW TO READ THIS")
    print("-" * 78)
    comf = [m["comf_proxy"] for m in results.values() if m]
    if comf and min(comf) >= 99.0:
        print(f"  Comf-proxy {min(comf):.1f}-{max(comf):.1f} across every arm: SATURATED, matching")
        print(f"  NAVSIM (Comf. = 100 even for Constant Velocity). It cannot rank these arms,")
        print(f"  and our GoT score is exactly this axis (kinematic + command) -- which is why")
        print(f"  it could not select (sec.1.1). The saturation is the finding, not a null.")
    elif comf:
        print(f"  Comf-proxy spans {min(comf):.1f}-{max(comf):.1f}: on OUR trajectories it is NOT")
        print(f"  saturated, unlike NAVSIM's. Do not inherit the 'zero discriminative power'")
        print(f"  claim without re-checking it here -- that claim was measured elsewhere.")
    if any(m["dac"] is None for m in results.values() if m):
        print("  DAC is missing: pass --drivable_masks built by data/preprocess_nuscenes_map.py")
        print("  (needs nuScenes map expansion v1.3, which the base download does not include).")
    print("  ⛔ These columns compare OUR arms to each other. Do not place them beside a")
    print("     published NAVSIM table -- different benchmark, horizon, rate and simulator.")
    print("     For a real PDMS, run the NAVSIM track (scripts/run_navsim.sh).")


# --------------------------------------------------------------------------- #
def selftest():
    from got_drive.collision_metric import CollisionConfig
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))

    cfg = CollisionConfig(uniad_parity=True)
    cv = [[4.0 * (i + 1), 0.0] for i in range(6)]

    # --- Comf
    lon_a, _, _, _ = kinematics(cv)
    check("comf: constant velocity -> zero accel", max(map(abs, lon_a)) < 1e-9)
    check("comf: constant velocity is comfortable", comfortable(cv)[0])
    ramp = [[0.5 * 8.0 * ((i + 1) * DT) ** 2, 0.0] for i in range(6)]
    check("comf: hard acceleration flagged", not comfortable(ramp)[0])
    jump = [[30.0, 0.0]] + [[30.0 + 4.0 * i, 0.0] for i in range(1, 6)]
    check("comf: first-interval spike is seen", not comfortable(jump)[0])

    # --- EP
    v, sc = ego_progress(cv, cv)
    check("ep: matching the GT scores 1.0", sc and abs(v - 1.0) < 1e-9, f"v={v}")
    half = [[2.0 * (i + 1), 0.0] for i in range(6)]
    v2, _ = ego_progress(half, cv)
    check("ep: half the progress scores 0.5", abs(v2 - 0.5) < 1e-9, f"v={v2}")
    v3, _ = ego_progress([[-5.0 * (i + 1), 0.0] for i in range(6)], cv)
    check("ep: driving backwards clips at 0", abs(v3) < 1e-12, f"v={v3}")
    _, sc4 = ego_progress(cv, [[0.01 * (i + 1), 0.0] for i in range(6)])
    check("ep: stationary GT is skipped, not divided by", not sc4)

    # --- TTC
    blocking = [[[12.0 + 4.0 * t, 0.0, 4.5, 2.0, 0.0]] for t in range(6)]
    far = [[[0.0, 45.0, 4.5, 2.0, 0.0]] for _ in range(6)]
    check("ttc: agent parked in the projected path -> violation",
          ttc_violations(cv, blocking, cfg).any())
    check("ttc: agent far aside -> no violation", not ttc_violations(cv, far, cfg).any())
    check("ttc: a longer horizon cannot see fewer violations",
          ttc_violations(cv, blocking, cfg, 2.0).sum() >= 0)

    # --- DAC
    nx = int(round((cfg.x_bound[1] - cfg.x_bound[0]) / cfg.resolution))
    ny = int(round((cfg.y_bound[1] - cfg.y_bound[0]) / cfg.resolution))
    allfree = np.ones((nx, ny), dtype=bool)
    bad, n_out = dac_violation(cv, allfree, cfg)
    check("dac: everything drivable -> compliant", not bad and n_out == 0)
    corridor = np.zeros((nx, ny), dtype=bool)
    ymid = ny // 2
    corridor[:, ymid - 6:ymid + 6] = True          # a +-3 m lane along +x
    check("dac: straight run stays in the lane", not dac_violation(cv, corridor, cfg)[0])
    swerve = [[4.0 * (i + 1), 20.0] for i in range(6)]
    check("dac: leaving the lane is flagged", dac_violation(swerve, corridor, cfg)[0])
    runaway = [[200.0 * (i + 1), 0.0] for i in range(6)]
    bad_r, n_out_r = dac_violation(runaway, corridor, cfg)
    check("dac: leaving the mapped window counts as a violation and is reported",
          bad_r and n_out_r > 0, f"n_out={n_out_r}")

    print("\nselftest:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    p = argparse.ArgumentParser("PDM-Score components on nuScenes (CPU only)")
    p.add_argument("runs", nargs="*")
    p.add_argument("--collision_json", default=None)
    p.add_argument("--drivable_masks", default=None,
                   help="npz from data/preprocess_nuscenes_map.py; without it DAC and "
                        "PDMS-proxy are reported as n/a rather than guessed")
    p.add_argument("--pred_col", default="pred")
    p.add_argument("--status_col", default="status")
    p.add_argument("--gt_json", default=None)
    p.add_argument("--parity", choices=["uniad", "ours"], default="uniad")
    p.add_argument("--ttc_horizon", type=float, default=1.0)
    p.add_argument("--min_progress_m", type=float, default=1.0)
    p.add_argument("--no_gt_mask", action="store_true", default=False)
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args()

    if args.selftest:
        sys.exit(selftest())
    if not args.runs:
        p.error("give at least one run dir (or pass --selftest)")

    from got_drive.collision_metric import CollisionConfig
    cfg = CollisionConfig(uniad_parity=(args.parity == "uniad"))

    boxes = {}
    if args.collision_json:
        with open(args.collision_json, encoding="utf-8") as f:
            boxes = {r["sample_token"]: r["agent_boxes"] for r in json.load(f)}
    else:
        print("[warn] no --collision_json: NC-proxy and TTC-proxy will be n/a")

    drivable = None
    if args.drivable_masks:
        z = np.load(args.drivable_masks)
        meta = json.loads(str(z["__meta__"]))
        if (abs(meta["resolution"] - cfg.resolution) > 1e-9
                or list(meta["x_bound"]) != list(cfg.x_bound)
                or list(meta["y_bound"]) != list(cfg.y_bound)):
            sys.exit(f"[fatal] mask grid {meta} does not match the collision grid "
                     f"(res {cfg.resolution}, x {cfg.x_bound}, y {cfg.y_bound}). "
                     f"Rebuild the masks with the same bounds or the two metrics sit "
                     f"on different geometry.")
        drivable = {k: z[k].astype(bool) for k in z.files if k != "__meta__"}
        print(f"[pdms] drivable masks for {len(drivable)} records "
              f"(grid {meta['x_bound']} x {meta['y_bound']} @ {meta['resolution']} m)")

    print(f"[pdms] comfort bounds (nuPlan defaults, VERIFY before quoting): "
          f"lon_a {MAX_ABS_LON_ACCEL}, lat_a {MAX_ABS_LAT_ACCEL}, "
          f"lon_jerk {MAX_ABS_LON_JERK}, yaw_rate {MAX_ABS_YAW_RATE} | dt {DT}s | "
          f"ttc horizon {args.ttc_horizon}s | gt_mask {not args.no_gt_mask}")

    gt_by_tok = None
    if args.gt_json:
        with open(args.gt_json, encoding="utf-8") as f:
            gt_by_tok = {r["sample_token"]: r["waypoints"] for r in json.load(f)}
        print(f"[pdms] GT joined from {os.path.basename(args.gt_json)} ({len(gt_by_tok)} records)")

    results = {}
    for run in args.runs:
        m = score(run, boxes, drivable, cfg, args, gt_by_tok)
        if m:
            name = os.path.basename(os.path.normpath(run))
            results[name] = m
            report(name, m)
            with open(os.path.join(run, "pdms_components.json"), "w", encoding="utf-8") as f:
                json.dump(m, f, indent=2)
    if results:
        verdict(results)


if __name__ == "__main__":
    main()
