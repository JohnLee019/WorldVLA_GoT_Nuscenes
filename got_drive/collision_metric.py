"""
UniAD / ST-P3-style open-loop COLLISION-RATE metric for nuScenes planning.

The second standard metric UniAD reports alongside L2. Given a predicted ego
trajectory and the ground-truth boxes of the OTHER agents at each future
timestep (open-loop assumption: other agents follow their GT motion), it asks:
does the ego footprint, driven along the predicted path, overlap any agent?

Method (matches ST-P3/UniAD's grid-occupancy overlap)
-----------------------------------------------------
Everything is expressed in the ego frame at t0 (the frame the planner predicts
in). At each future timestep t:
  * rasterize the ego box (fixed size, placed at the predicted waypoint with the
    heading implied by the trajectory) into a BEV occupancy grid,
  * rasterize every agent box at that timestep into the same grid,
  * collision iff the two occupancies share any cell.
This reproduces UniAD's 0.5 m grid-quantized overlap (not an analytic polygon
test), so the numbers line up with the occupancy-based metric.

Reported per horizon -- all THREE aggregations, because published tables differ
and the same trajectories give different numbers under each:
  * coll@1s/2s/3s     -- collision at exactly that timestep (UniAD's "uniad"
                         evaluation_strategy: `value[i]`)
  * meanColl@1s/2s/3s -- mean of the per-step rates over 0..t (UniAD's "stp3"
                         strategy: `value[:i+1].mean()`, what ST-P3/VAD report)
  * cumColl@1s/2s/3s  -- collision at ANY step up to t (a strict latch; OURS, no
                         published table uses this -- keep it as a diagnostic)

★ Verified against UniAD's official PlanningMetric
  (projects/mmdet3d_plugin/uniad/dense_heads/planning_head_plugin/planning_metrics.py).
  Matching already: ego 4.084 x 1.85 m, +-50 m bounds, 0.5 m grid.
  Differing by design, switched by `uniad_parity=True`:
    1. UniAD does NOT rotate the ego box -- its footprint is axis-aligned at
       every waypoint regardless of heading (a known limitation, criticised by
       BEV-Planner). Ours orients the box along the trajectory heading, which is
       physically right but NOT comparable to published numbers.
    2. UniAD shifts the box +0.5 m forward (`pts` are built from -H/2+0.5 and
       H/2+0.5), i.e. the waypoint is a rear-axle-ish reference, not the centre.
    3. UniAD masks out steps where the GT trajectory ITSELF collides
       (`m1 = logical_and(m1, logical_not(gt_box_coll))`) -- annotation and
       box-size artifacts. Pass `gt_traj=` to enable; without it our rates are
       biased HIGH relative to any published number.
    4. UniAD rasterises vehicles only; ST-P3/VAD add pedestrians. That is set
       upstream by preprocess_nuscenes_collision.py --categories, not here.

  Report `uniad_parity=True` as the headline for standards compliance, and the
  yaw-aware default as a stricter secondary. Do not mix them in one table.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class CollisionConfig:
    # ST-P3/UniAD default ego box (metres). PIN to the repo you compare against.
    ego_length: float = 4.084
    ego_width: float = 1.85
    x_bound: Tuple[float, float] = (-50.0, 50.0)   # forward range (ego +x)
    y_bound: Tuple[float, float] = (-50.0, 50.0)   # lateral range (ego +y, left)
    resolution: float = 0.5                        # BEV cell size (m) -> 200x200

    # --- ego-footprint placement (see the module note) -----------------------
    # uniad_parity=True reproduces UniAD's PlanningMetric exactly: axis-aligned
    # box, shifted +0.5 m forward. Set it INSTEAD of the two fields below; it
    # overrides them in __post_init__ so parity cannot be half-applied.
    uniad_parity: bool = False
    apply_yaw: bool = True          # orient the ego box along the trajectory heading
    ego_x_offset: float = 0.0       # forward shift of the box centre from the waypoint

    def __post_init__(self):
        if self.uniad_parity:
            self.apply_yaw = False
            self.ego_x_offset = 0.5


# ──────────────────────────────────────────────────────────────────────────
# oriented-box rasterization into a BEV occupancy grid
# ──────────────────────────────────────────────────────────────────────────

def _box_corners(cx, cy, L, W, yaw):
    """4 corners of an oriented rectangle (forward=+x local, left=+y local)."""
    dx, dy = L / 2.0, W / 2.0
    local = np.array([[dx, dy], [dx, -dy], [-dx, -dy], [-dx, dy]])
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    return local @ R.T + np.array([cx, cy])


def _rasterize_box(occ, cx, cy, L, W, yaw, cfg: CollisionConfig):
    """Mark grid cells whose CENTER lies inside the oriented box as True.

    Only iterates over the box's local AABB (clipped to the grid), so cost is
    proportional to the box area, not the whole grid.
    """
    x0, x1 = cfg.x_bound
    y0, y1 = cfg.y_bound
    r = cfg.resolution

    corners = _box_corners(cx, cy, L, W, yaw)
    cxmin, cymin = corners.min(0)
    cxmax, cymax = corners.max(0)
    ix0 = max(0, int(np.floor((cxmin - x0) / r)))
    ix1 = min(occ.shape[0] - 1, int(np.ceil((cxmax - x0) / r)))
    iy0 = max(0, int(np.floor((cymin - y0) / r)))
    iy1 = min(occ.shape[1] - 1, int(np.ceil((cymax - y0) / r)))
    if ix0 > ix1 or iy0 > iy1:
        return

    c, s = np.cos(yaw), np.sin(yaw)          # world->local uses R(-yaw)
    hl, hw = L / 2.0, W / 2.0
    for ix in range(ix0, ix1 + 1):
        wx = x0 + (ix + 0.5) * r
        for iy in range(iy0, iy1 + 1):
            wy = y0 + (iy + 0.5) * r
            dx, dy = wx - cx, wy - cy
            lx = c * dx + s * dy             # box-local x
            ly = -s * dx + c * dy            # box-local y
            if abs(lx) <= hl and abs(ly) <= hw:
                occ[ix, iy] = True


# ──────────────────────────────────────────────────────────────────────────
# trajectory collision
# ──────────────────────────────────────────────────────────────────────────

def _headings(traj):
    """Per-waypoint heading (rad) from finite differences; first from origin."""
    traj = np.asarray(traj, dtype=np.float64)
    prev = np.vstack([[0.0, 0.0], traj[:-1]])
    d = traj - prev
    return np.arctan2(d[:, 1], d[:, 0])


def _per_step_collisions(traj, agent_boxes: List[List], cfg: CollisionConfig) -> np.ndarray:
    """(T,) bool: does the ego footprint driven along `traj` overlap an agent?"""
    traj = np.asarray(traj, dtype=np.float64)
    T = traj.shape[0]
    yaws = _headings(traj) if cfg.apply_yaw else np.zeros(T)

    nx = int(round((cfg.x_bound[1] - cfg.x_bound[0]) / cfg.resolution))
    ny = int(round((cfg.y_bound[1] - cfg.y_bound[0]) / cfg.resolution))

    per_step = np.zeros(T, dtype=bool)
    for t in range(T):
        boxes_t = agent_boxes[t] if t < len(agent_boxes) else []
        if not boxes_t:
            continue
        # the box reference point sits `ego_x_offset` ahead of the waypoint,
        # along the heading (== plain +x when apply_yaw is off, as in UniAD)
        cx = traj[t, 0] + cfg.ego_x_offset * np.cos(yaws[t])
        cy = traj[t, 1] + cfg.ego_x_offset * np.sin(yaws[t])
        ego = np.zeros((nx, ny), dtype=bool)
        _rasterize_box(ego, cx, cy, cfg.ego_length, cfg.ego_width, yaws[t], cfg)
        if not ego.any():
            continue                                    # ego outside BEV range
        agents = np.zeros((nx, ny), dtype=bool)
        for b in boxes_t:
            x, y, L, W, yaw = (float(v) for v in b)
            _rasterize_box(agents, x, y, L, W, yaw, cfg)
        per_step[t] = bool((ego & agents).any())
    return per_step


def trajectory_collisions(
    pred_traj,
    agent_boxes: List[List],
    hz_idx: Dict[str, int],
    cfg: Optional[CollisionConfig] = None,
    gt_traj=None,
):
    """Collision of `pred_traj` against `agent_boxes`, per horizon.

    pred_traj   : (T, 2) ego waypoints in the t0 ego frame (metres).
    agent_boxes : length-T list; agent_boxes[t] is a list of [x, y, L, W, yaw]
                  obstacle boxes at timestep t (SAME t0 ego frame, metres/rad).
    hz_idx      : {"1s": 1, "2s": 3, "3s": 5} label -> waypoint index.
    gt_traj     : (T, 2) GROUND-TRUTH waypoints. When given, steps at which the
                  GT trajectory itself collides are masked out of the prediction's
                  collision curve -- UniAD's correction for annotation/box-size
                  artifacts. Omit it and the rates come out biased HIGH versus
                  every published number.

    Returns (metrics_dict, per_step_bool) with coll@<h> / meanColl@<h> /
    cumColl@<h> (see the module note for which published table uses which).
    per_step_bool is the (T,) collision curve, already GT-masked.
    """
    cfg = cfg or CollisionConfig()
    per_step = _per_step_collisions(pred_traj, agent_boxes, cfg)
    if gt_traj is not None:
        per_step = per_step & ~_per_step_collisions(gt_traj, agent_boxes, cfg)

    out = {}
    for label, idx in hz_idx.items():
        out[f"coll@{label}"] = float(per_step[idx])
        out[f"meanColl@{label}"] = float(per_step[: idx + 1].mean())
        out[f"cumColl@{label}"] = float(per_step[: idx + 1].any())
    return out, per_step


# ──────────────────────────────────────────────────────────────────────────
# self-test: pure numpy, no nuScenes / GPU
# ──────────────────────────────────────────────────────────────────────────

def _selftest():
    cfg = CollisionConfig()
    hz = {"1s": 1, "2s": 3, "3s": 5}

    # ego drives straight forward: waypoints every 0.5 s
    traj = np.array([[2, 0], [4, 0], [6, 0], [8, 0], [10, 0], [12, 0]], dtype=float)

    # (a) an agent box sitting exactly on the ego at 2 s (idx 3, pos (8,0))
    boxes_hit = [[] for _ in range(6)]
    boxes_hit[3] = [[8.0, 0.0, 4.5, 2.0, 0.0]]        # overlaps ego at (8,0)
    m, ps = trajectory_collisions(traj, boxes_hit, hz, cfg)
    assert ps[3] and m["coll@2s"] == 1.0, f"expected collision at 2s, got {m}"
    assert m["coll@1s"] == 0.0, "no collision at 1s"
    assert m["cumColl@3s"] == 1.0 and m["cumColl@2s"] == 1.0, "cumulative should latch"
    assert m["cumColl@1s"] == 0.0, "no cumulative collision by 1s"
    # meanColl = per-step rate averaged over 0..t (UniAD "stp3" strategy).
    # per_step = [F,F,F,T,F,F] -> @1s: 0/2, @2s: 1/4, @3s: 1/6
    assert m["meanColl@1s"] == 0.0, m
    assert abs(m["meanColl@2s"] - 0.25) < 1e-12, m
    assert abs(m["meanColl@3s"] - 1.0 / 6.0) < 1e-12, m

    # (b) agent far to the side -> no collision anywhere
    boxes_far = [[[8.0, 30.0, 4.5, 2.0, 0.0]] for _ in range(6)]
    m2, ps2 = trajectory_collisions(traj, boxes_far, hz, cfg)
    assert not ps2.any(), f"expected no collision, got {ps2}"

    # (c) agent in an adjacent lane (y=3.5), ego width 1.85 -> gap, no overlap
    boxes_lane = [[[8.0, 3.5, 4.5, 2.0, 0.0]] for _ in range(6)]
    m3, ps3 = trajectory_collisions(traj, boxes_lane, hz, cfg)
    assert not ps3.any(), f"adjacent lane must not collide, got {ps3}"

    # (d) rotation sanity: a long box across the path at 3 s (idx5, (12,0))
    boxes_rot = [[] for _ in range(6)]
    boxes_rot[5] = [[12.0, 0.0, 8.0, 2.0, np.pi / 2]]  # length across lateral
    m4, ps4 = trajectory_collisions(traj, boxes_rot, hz, cfg)
    assert ps4[5] and m4["coll@3s"] == 1.0, f"rotated box should hit at 3s, got {m4}"

    # (e) corners/local-frame sanity
    corners = _box_corners(0, 0, 4.0, 2.0, 0.0)
    assert np.allclose(sorted(corners[:, 0]), [-2, -2, 2, 2]), corners

    # ---- UniAD parity -----------------------------------------------------
    uniad = CollisionConfig(uniad_parity=True)
    assert uniad.apply_yaw is False and uniad.ego_x_offset == 0.5, uniad

    # (f) +0.5 m forward shift: ego at (8,0) spans x in [5.958, 10.042] centred,
    #     [6.458, 10.542] shifted. A thin agent at x in [10.1, 10.5] is reachable
    #     only by the shifted box (grid cell centre 10.25).
    boxes_off = [[] for _ in range(6)]
    boxes_off[3] = [[10.3, 0.0, 0.4, 2.0, 0.0]]
    _, ps_c = trajectory_collisions(traj, boxes_off, hz, cfg)
    _, ps_u = trajectory_collisions(traj, boxes_off, hz, uniad)
    assert not ps_c.any(), f"centred box must miss, got {ps_c}"
    assert ps_u[3], f"UniAD's +0.5 m shifted box must hit, got {ps_u}"

    # (g) yaw: on a 45-degree diagonal path, a point off the diagonal corner is
    #     inside the ORIENTED box but outside the axis-aligned one UniAD uses.
    diag = np.array([[2, 2], [4, 4], [6, 6], [8, 8], [10, 10], [12, 12]], dtype=float)
    boxes_diag = [[] for _ in range(6)]
    boxes_diag[3] = [[9.25, 9.25, 0.4, 0.4, 0.0]]
    _, ps_yaw = trajectory_collisions(diag, boxes_diag, hz, cfg)
    _, ps_noyaw = trajectory_collisions(diag, boxes_diag, hz, uniad)
    assert ps_yaw[3], f"oriented ego box should hit, got {ps_yaw}"
    assert not ps_noyaw.any(), f"axis-aligned (UniAD) box should miss, got {ps_noyaw}"

    # ---- GT-collision masking (UniAD's correction) ------------------------
    # (h) GT drives the same path -> it collides too -> the prediction's hit is
    #     discounted entirely.
    m_gt, ps_gt = trajectory_collisions(traj, boxes_hit, hz, cfg, gt_traj=traj)
    assert not ps_gt.any(), f"GT-colliding step must be masked out, got {ps_gt}"
    assert m_gt["coll@2s"] == 0.0 and m_gt["cumColl@3s"] == 0.0, m_gt

    # (i) GT drives far away -> it never collides -> the prediction's hit stands.
    gt_far = traj + np.array([0.0, 20.0])
    m_keep, ps_keep = trajectory_collisions(traj, boxes_hit, hz, cfg, gt_traj=gt_far)
    assert ps_keep[3] and m_keep["coll@2s"] == 1.0, m_keep

    # (j) partial masking: GT collides at step 3 only, prediction at 3 and 4.
    boxes_two = [[] for _ in range(6)]
    boxes_two[3] = [[8.0, 0.0, 4.5, 2.0, 0.0]]
    boxes_two[4] = [[10.0, 0.0, 4.5, 2.0, 0.0]]
    gt_only3 = traj.copy()
    gt_only3[4] = [10.0, 20.0]          # GT swerves away exactly at step 4
    _, ps_part = trajectory_collisions(traj, boxes_two, hz, cfg, gt_traj=gt_only3)
    assert not ps_part[3] and ps_part[4], f"only step 3 should be masked, got {ps_part}"

    print("collision_metric self-test: OK")


if __name__ == "__main__":
    _selftest()
