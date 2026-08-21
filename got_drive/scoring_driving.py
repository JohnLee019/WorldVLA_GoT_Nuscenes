"""
Driving-appropriate candidate scores for open-loop GoT (no world model).

The LIBERO scores are unusable for driving (see got_vla_v2/scoring/): they need
`env.step`, or reward *large* pixel motion (a manipulation prior that is exactly
backwards for driving), or assign every candidate the same value. Instead we
score each candidate trajectory *directly*, from the trajectory itself plus the
cheap side information already in each record. No simulator, no world model, no
future image.

Two minimal components (both computable from the trajectory alone):

  kinematic  -- comfort/plausibility prior. Penalises acceleration, jerk and
                lateral (centripetal) acceleration, and hard-penalises
                physically implausible speed/accel. Higher = smoother = better.
                CAN prune insane candidates; CANNOT tell if a smooth trajectory
                is correct for the scene (a smooth path into a wall scores well).

  command    -- goal grounding. Rewards a final lateral offset consistent with
                the high-level navigation command (left / right / straight).
                In the nuScenes planning protocol the command is a *provided*
                navigation input (UniAD/VAD consume it), so conditioning on it
                is standard, not GT leakage. NOTE: in this repo the command is
                derived from the GT future by preprocess_nuscenes.derive_command;
                to use it honestly at test time it must come from a route/nav
                input, not from the GT waypoints.

Scores are HIGHER = better, matching the GoT convention (keep the top-b by
accumulated path score). Because the two components live in different units,
`rank_candidates` z-normalises each across the candidate set before the weighted
sum, so the weights are scale-free.

A third option, model self-likelihood ("how much does the model believe this
trajectory"), needs a teacher-forced forward pass and therefore the model on
GPU; it lives in segment_generation.py, not here, so this module stays pure
numpy and locally testable. Run `python -m got_drive.scoring_driving` for a
self-test.

Design note for segment-wise GoT
--------------------------------
Pass the CUMULATIVE trajectory decided so far (prefix + this segment), not just
the new segment, so the kinematic cost sees the junction between segments. The
ego's current pose is the origin (0, 0) at t=0; `include_origin=True` prepends
it so the first velocity reflects real ego motion. `command` is a terminal
signal (it reads the final lateral offset), so it is exact only once the
trajectory is complete; on intermediate segments it scores the trend so far.
"""

import numpy as np

# waypoints are 0.5 s apart (see eval_nuscenes)
DEFAULT_DT = 0.5
# derive_command's lateral threshold (see data/preprocess_nuscenes.py)
LATERAL_THRESH = 2.0


def _finite_diff(p, dt):
    """Row-wise finite difference along time: (N, d) -> (N-1, d)."""
    return np.diff(p, axis=0) / dt


def _with_origin(traj, include_origin):
    traj = np.asarray(traj, dtype=np.float64)
    if traj.ndim != 2:
        raise ValueError(f"trajectory must be (N, 2), got shape {traj.shape}")
    if include_origin:
        traj = np.vstack([np.zeros((1, traj.shape[1])), traj])
    return traj


def score_kinematic(traj, dt=DEFAULT_DT, include_origin=True,
                    w_accel=1.0, w_jerk=1.0, w_lat=1.0,
                    v_max=20.0, a_max=5.0, w_limit=10.0):
    """Comfort/plausibility score for one trajectory. Higher (=> 0) is smoother.

    Returns the NEGATIVE of a weighted motion cost, so a perfectly smooth,
    feasible path approaches 0 and rough/implausible paths are strongly
    negative. Units are m/s^n; combine relatively via `rank_candidates`.
    """
    traj = _with_origin(traj, include_origin)
    if traj.shape[0] < 3:
        return 0.0

    v = _finite_diff(traj, dt)              # (N-1, 2) velocity
    speed = np.linalg.norm(v, axis=1)
    a = _finite_diff(v, dt)                 # (N-2, 2) acceleration
    accel_mag = np.linalg.norm(a, axis=1)

    cost = w_accel * float(np.mean(accel_mag ** 2))

    if a.shape[0] >= 2:
        j = _finite_diff(a, dt)            # (N-3, 2) jerk
        cost += w_jerk * float(np.mean(np.linalg.norm(j, axis=1) ** 2))

    # lateral (centripetal) acceleration via |v x a| / |v|, aligning v[:-1] to a
    vv = v[:-1]
    sp = np.linalg.norm(vv, axis=1) + 1e-6
    cross = np.abs(vv[:, 0] * a[:, 1] - vv[:, 1] * a[:, 0])
    lat_acc = cross / sp
    cost += w_lat * float(np.mean(lat_acc ** 2))

    # hard-ish penalties for physically implausible speed / acceleration
    cost += w_limit * float(np.mean(np.clip(speed - v_max, 0, None) ** 2))
    cost += w_limit * float(np.mean(np.clip(accel_mag - a_max, 0, None) ** 2))

    return -cost


def score_command(traj, command, lateral_thresh=LATERAL_THRESH,
                  include_origin=False):
    """Goal-grounding score: agreement of the final lateral offset with command.

    command in {"left", "right", "straight", "unknown"}; ego frame y is +left.
    Scaled by `lateral_thresh` so the natural decision boundary sits near +/-1.
    Higher is better:
        left     -> reward large +y   (turning left)
        right    -> reward large -y   (turning right)
        straight -> reward small |y|  (staying in lane)
        unknown  -> 0 for every candidate, i.e. NO preference

    "unknown" is NAVSIM's fourth driving_command slot. Session 15 measured 363
    of them in navtrain (0.24%) after Step 2.12b had found 0% on navtest and
    the handoff concluded the builder need not handle the class -- true for
    navtest, false for navtrain. Without this branch GoT inference raises on
    those scenarios.

    Returning 0 rather than falling back to "straight" is the honest reading:
    no navigation instruction means the command term expresses no preference,
    not "go straight". A constant component drops out of the ranking because
    _zscore maps zero variance to zeros, so the other components decide -- the
    same behaviour as --w_command 0, which measured d_output -0.0002.
    """
    traj = _with_origin(traj, include_origin)
    final_y = float(traj[-1, 1])
    t = float(lateral_thresh)
    if command == "left":
        return final_y / t
    if command == "right":
        return -final_y / t
    if command == "straight":
        return -abs(final_y) / t
    if command == "unknown":
        return 0.0
    raise ValueError(f"unknown command: {command!r}")


def _zscore(x):
    x = np.asarray(x, dtype=np.float64)
    s = x.std()
    if s < 1e-8:                # all candidates equal on this component
        return np.zeros_like(x)
    return (x - x.mean()) / s


def _rank_norm(x):
    """Rank-normalise to evenly spaced [-1, 1]; ties share the average rank.

    Why this exists. score_kinematic is -(a sum of squared terms plus a 10x hard
    limit penalty), so across one candidate set it routinely spans EIGHT orders
    of magnitude -- a real pool measured on nuScenes:

        [-0.013, -25.06, -1185.94, -2416.05, -2749.44, -3783.88, -14757.40,
         -1523444.60]

    z-normalising that hands the whole standard deviation (~5e5) to the single
    worst candidate, and every remaining candidate collapses onto one point: the
    three best above, whose true errors are 0.035 / 0.170 / 1.312 m, receive
    z-scores of +0.3851 / +0.3851 / +0.3828. The term that ranks candidates best
    in raw form (Spearman +0.55 on straights, +0.81 on turns) therefore
    contributes almost nothing to the ranking that is actually used, which is a
    plausible root cause for six ablation arms coming back null: reweighting a
    component that carries no ordering cannot change the outcome.

    Ranks discard magnitude, which is safe here precisely because catastrophes
    are not this function's job -- `_feasible` / `_apply_feasibility` veto them
    absolutely, outside the normalisation. See `rank_candidates(norm=...)`.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n < 2:
        return np.zeros_like(x)
    order = np.argsort(x, kind="stable")
    r = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and x[order[j + 1]] == x[order[i]]:
            j += 1
        r[order[i:j + 1]] = 0.5 * (i + j)
        i = j + 1
    return 2.0 * r / (n - 1) - 1.0


def _rank_norm_nan_safe(x):
    """_rank_norm over the finite entries; non-finite -> neutral 0."""
    x = np.asarray(x, dtype=np.float64)
    mask = np.isfinite(x)
    out = np.zeros_like(x)
    if mask.sum() >= 2:
        out[mask] = _rank_norm(x[mask])
    return out


# Normalisation applied to each component before the weighted sum.
#   "zscore" -- historical default; see _rank_norm for how outliers destroy it
#   "rank"   -- outlier-robust, keeps only the ordering
def _normalisers(norm):
    """(plain, nan_safe) normaliser pair for `norm`. Raises on an unknown name so
    a typo in a CLI flag fails immediately instead of silently z-scoring."""
    if norm == "zscore":
        return _zscore, _zscore_nan_safe
    if norm == "rank":
        return _rank_norm, _rank_norm_nan_safe
    raise ValueError(f"unknown score normalisation: {norm!r} (want 'zscore' or 'rank')")


def _zscore_nan_safe(x):
    """z-score over the finite entries only; non-finite (nan) entries -> 0.

    Used for the world-model component: candidates whose WM prediction was
    malformed, or that were not short-listed for a (costly) WM call, get nan and
    therefore a neutral 0 contribution, so they are judged on kinematic+command
    alone rather than poisoning the mean/std of the whole set.
    """
    x = np.asarray(x, dtype=np.float64)
    mask = np.isfinite(x)
    out = np.zeros_like(x)
    if mask.sum() < 2:
        return out
    m, s = x[mask].mean(), x[mask].std()
    if s >= 1e-8:
        out[mask] = (x[mask] - m) / s
    return out


# Hard physical-feasibility limits for the veto gate. Deliberately GENEROUS:
# meant to reject only clearly-impossible candidates, not to shape comfort (that
# is score_kinematic's job). A candidate violating ANY of these is pushed below
# every feasible one BEFORE it can be selected.
FEAS_V_MAX = 30.0         # m/s   (~108 km/h); per 0.5 s step -> 15 m
FEAS_A_MAX = 12.0         # m/s^2 (allows hard braking; rejects teleport-accel)
FEAS_LAT_STEP_MAX = 4.0   # m     per-step lateral jump |dy|; rejects range-bound flings
# Subtracted from an infeasible candidate's (z-scored, O(1)) total. Large but
# finite so the accumulated path_score in the pipeline stays comparable rather
# than collapsing to -inf.
INFEASIBLE_PENALTY = 1e6


def _feasible(traj, dt=DEFAULT_DT, include_origin=True,
              v_max=FEAS_V_MAX, a_max=FEAS_A_MAX, lat_step_max=FEAS_LAT_STEP_MAX):
    """True unless the trajectory violates a hard physical limit.

    Absolute, base-quality-independent gate. High-temperature sampling can emit
    off-manifold garbage (30 m forward in 0.5 s, a lateral fling to the range
    bound). score_kinematic already penalises these, but rank_candidates
    z-normalises, compressing even a colossal penalty to ~1-2 sigma so the
    command score can still select it. This flag lets the ranker veto them
    outright instead of merely down-ranking them.
    """
    traj = _with_origin(traj, include_origin)
    if traj.shape[0] < 2:
        return True
    v = _finite_diff(traj, dt)
    if np.any(np.linalg.norm(v, axis=1) > v_max):
        return False
    if np.any(np.abs(np.diff(traj[:, 1])) > lat_step_max):
        return False
    if traj.shape[0] >= 3:
        a = _finite_diff(v, dt)
        if np.any(np.linalg.norm(a, axis=1) > a_max):
            return False
    return True


def _apply_feasibility(total, kin, cands, dt, include_origin, feas_kwargs):
    """Veto physically-infeasible candidates AFTER the z-norm, so the hard limit
    acts as an absolute cutoff the (scale-free) command score cannot overturn.

    Infeasible candidates are sunk below every feasible one AND, among
    themselves, re-ranked by raw kinematic score ALONE (least-insane first). The
    command term is intentionally dropped for them: on a turn command rewards a
    large final |y|, i.e. it actively prefers the range-bound lateral flings we
    are rejecting -- so if a fully-infeasible pool (or a beam slot left over
    after the feasible ones) must pick something, it falls back to the
    smoothest/least-violating candidate, never the command-favoured extreme.
    """
    feas_kwargs = feas_kwargs or {}
    feas = np.array([_feasible(c, dt=dt, include_origin=include_origin, **feas_kwargs)
                     for c in cands])
    if feas.all():
        return total
    out = np.array(total, dtype=np.float64, copy=True)
    out[~feas] = np.asarray(kin, dtype=np.float64)[~feas] - INFEASIBLE_PENALTY
    return out


def rank_candidates(cands, command, weights=(1.0, 1.0),
                    dt=DEFAULT_DT, include_origin=True,
                    lateral_thresh=LATERAL_THRESH, kin_kwargs=None,
                    feas_kwargs=None, norm="zscore"):
    """Rank a set of candidate (cumulative) trajectories. Higher total = better.

    Each component is normalised across the candidate set before the weighted
    sum, so `weights=(w_kin, w_cmd)` are scale-free relative weights. Physically
    infeasible candidates (see `_feasible`) are then vetoed so they can never be
    selected over a feasible one. Returns (total_scores, components) where
    components has the raw per-candidate 'kinematic' and 'command' arrays for
    logging/ablation.

    `norm` selects the normalisation: "zscore" (historical default) or "rank"
    (outlier-robust). Prefer "rank" -- score_kinematic spans several orders of
    magnitude within one pool, and a single wild candidate then owns the standard
    deviation and flattens every other candidate to the same z. See _rank_norm
    for the measured example.
    """
    kin_kwargs = kin_kwargs or {}
    nrm, _ = _normalisers(norm)
    kin = np.array([score_kinematic(c, dt=dt, include_origin=include_origin, **kin_kwargs)
                    for c in cands])
    cmd = np.array([score_command(c, command, lateral_thresh=lateral_thresh,
                                  include_origin=include_origin) for c in cands])
    total = weights[0] * nrm(kin) + weights[1] * nrm(cmd)
    total = _apply_feasibility(total, kin, cands, dt, include_origin, feas_kwargs)
    return total, {"kinematic": kin, "command": cmd}


def rank_candidates_wm(cands, command, wm_scores, weights=(1.0, 1.0, 1.0),
                       dt=DEFAULT_DT, include_origin=True,
                       lateral_thresh=LATERAL_THRESH, kin_kwargs=None,
                       feas_kwargs=None, norm="zscore"):
    """rank_candidates + a precomputed world-model plausibility component.

    wm_scores: array aligned with `cands`, higher = more plausible future
    (got_drive.world_model.score_next_frame_plausibility). Use np.nan for a
    candidate with no valid WM score (malformed frame, or not short-listed for a
    WM call in the two-stage rerank); its WM contribution is a neutral 0.

    weights = (w_kinematic, w_command, w_wm), scale-free (each component is
    z-normalised across the candidate set before the weighted sum). Returns
    (total_scores, components) with 'kinematic', 'command', 'wm' raw arrays.
    """
    kin_kwargs = kin_kwargs or {}
    nrm, nrm_nan = _normalisers(norm)
    kin = np.array([score_kinematic(c, dt=dt, include_origin=include_origin, **kin_kwargs)
                    for c in cands])
    cmd = np.array([score_command(c, command, lateral_thresh=lateral_thresh,
                                  include_origin=include_origin) for c in cands])
    wm = np.asarray(wm_scores, dtype=np.float64)
    total = (weights[0] * nrm(kin)
             + weights[1] * nrm(cmd)
             + weights[2] * nrm_nan(wm))
    total = _apply_feasibility(total, kin, cands, dt, include_origin, feas_kwargs)
    return total, {"kinematic": kin, "command": cmd, "wm": wm}


# --------------------------------------------------------------------------- #
# Local self-test (pure numpy, no GPU). Validates the two scores rank
# hand-built trajectories the way we expect.
# --------------------------------------------------------------------------- #
def _selftest():
    dt = DEFAULT_DT
    # ~8 m/s straight cruise, one waypoint every 0.5 s (x grows ~4 m/step)
    smooth_straight = np.array([[4, 0], [8, 0], [12, 0], [16, 0], [20, 0], [24, 0]], float)
    # same forward progress but jittery lateral -> high jerk / lateral accel
    jerky = np.array([[4, 0], [8, 1.5], [12, -1.5], [16, 1.5], [20, -1.5], [24, 0]], float)
    # smooth left turn (final y well past +thresh) and right turn (past -thresh)
    left_turn = np.array([[4, 0.2], [8, 0.8], [12, 1.6], [16, 2.6], [20, 3.8], [24, 5.2]], float)
    right_turn = np.array([[4, -0.2], [8, -0.8], [12, -1.6], [16, -2.6], [20, -3.8], [24, -5.2]], float)

    ok = True

    # 1) kinematic: smooth >> jerky
    ks, kj = score_kinematic(smooth_straight, dt), score_kinematic(jerky, dt)
    print(f"[kinematic] smooth={ks:.3f}  jerky={kj:.3f}   (expect smooth > jerky)")
    ok &= ks > kj

    # 2) command: correct direction wins for each command
    for cmd, cands, want_idx, names in [
        ("left",     [left_turn, smooth_straight, right_turn], 0, ["left", "straight", "right"]),
        ("right",    [left_turn, smooth_straight, right_turn], 2, ["left", "straight", "right"]),
        ("straight", [left_turn, smooth_straight, right_turn], 1, ["left", "straight", "right"]),
    ]:
        cs = [score_command(c, cmd) for c in cands]
        best = int(np.argmax(cs))
        print(f"[command={cmd:8s}] scores={[round(x,2) for x in cs]} "
              f"-> best={names[best]} (want {names[want_idx]})")
        ok &= best == want_idx

    # 3) rank_candidates: for a 'right' command, the smooth right turn should
    #    top a set that also contains a jerky (uncomfortable) right turn.
    jerky_right = np.array([[4, -0.5], [8, -0.3], [12, -2.2], [16, -2.0], [20, -4.5], [24, -5.2]], float)
    total, comp = rank_candidates([right_turn, jerky_right, left_turn, smooth_straight],
                                  command="right", weights=(1.0, 1.0))
    winner = int(np.argmax(total))
    print(f"[rank right] total={np.round(total,2).tolist()} kin={np.round(comp['kinematic'],2).tolist()} "
          f"cmd={np.round(comp['command'],2).tolist()} -> winner idx={winner} (want 0)")
    ok &= winner == 0

    # 4) rank_candidates_wm: three near-identical right turns (so kin+cmd are ~tied);
    #    idx0 gets no WM call (nan -> neutral), idx1 a plausible future, idx2 an
    #    implausible one. With WM weighted up, the winner must be idx1, and the
    #    un-scored idx0 must stay neutral (nan not poisoning the z-score).
    twin_a = right_turn
    twin_b = np.array([[4, -0.2], [8, -0.85], [12, -1.6], [16, -2.55], [20, -3.8], [24, -5.2]], float)
    twin_c = np.array([[4, -0.25], [8, -0.8], [12, -1.65], [16, -2.5], [20, -3.85], [24, -5.2]], float)
    wm_scores = [np.nan, 2.0, -2.0]     # idx0 not scored; idx1 plausible; idx2 not
    total_wm, comp_wm = rank_candidates_wm([twin_a, twin_b, twin_c], command="right",
                                           wm_scores=wm_scores, weights=(1.0, 1.0, 3.0))
    wm_winner = int(np.argmax(total_wm))
    print(f"[rank_wm right] total={np.round(total_wm,2).tolist()} "
          f"wm={comp_wm['wm'].tolist()} -> winner idx={wm_winner} (want 1)")
    ok &= wm_winner == 1

    # 5) feasibility veto (isolates the gate). With command weighted UP, a smooth
    #    candidate that drifts slightly off 'straight' would LOSE on command to an
    #    insane forward-teleport candidate that happens to end at y=0 -- exactly
    #    the reward-hacking that produced GoT's catastrophic-tail losses. The gate
    #    must veto the physically-impossible candidate so the smooth one wins.
    smooth_drift = np.array([[4, 0.3], [8, 0.6], [12, 0.9], [16, 1.2], [20, 1.5], [24, 1.8]], float)
    insane_teleport = np.array([[4, 0], [8, 0], [12, 0], [16, 0], [45, 0], [70, 0]], float)
    print(f"[feasible] smooth_drift={_feasible(smooth_drift)} "
          f"insane_teleport={_feasible(insane_teleport)}   (expect True, False)")
    ok &= _feasible(smooth_drift) and not _feasible(insane_teleport)
    tf, _ = rank_candidates([smooth_drift, insane_teleport],
                            command="straight", weights=(1.0, 5.0))
    fwin = int(np.argmax(tf))
    print(f"[feas veto] total={np.round(tf,2).tolist()} -> winner idx={fwin} (want 0, smooth)")
    ok &= fwin == 0

    # 6) all-infeasible fallback = kinematic-only (command must NOT decide). Both
    #    candidates violate limits; 'left' command loves the lateral fling (final
    #    y=16), but the fallback must pick the smoother constant-speed overshoot.
    mild_speed = np.array([[16, 0], [32, 0], [48, 0], [64, 0], [80, 0], [96, 0]], float)   # 32 m/s, smooth
    wild_lateral = np.array([[4, 0], [8, 0], [12, 0], [16, 0], [20, 0], [20, 16]], float)  # lateral teleport
    print(f"[all-infeas] feasible={[_feasible(mild_speed), _feasible(wild_lateral)]} (expect both False)")
    ok &= (not _feasible(mild_speed)) and (not _feasible(wild_lateral))
    ti, _ = rank_candidates([mild_speed, wild_lateral], command="left", weights=(1.0, 5.0))
    iwin = int(np.argmax(ti))
    print(f"[all-infeas] total={np.round(ti,1).tolist()} -> winner idx={iwin} "
          f"(want 0, least-insane; NOT the command-favoured fling)")
    ok &= iwin == 0

    # 7) normalisation: reproduce the measured pool that motivated `norm="rank"`.
    #    Under z-score, one wild candidate owns the std and the three best
    #    collapse onto the same value even though their true errors span 37x;
    #    under rank-norm they stay separated. This is the failure that plausibly
    #    made six ablation arms null, so it is pinned here rather than described.
    real_kin = np.array([-0.013, -25.06, -1185.94, -2416.05,
                         -2749.44, -3783.88, -14757.40, -1523444.60])
    z, rk = _zscore(real_kin), _rank_norm(real_kin)
    z_span = float(z[:3].max() - z[:3].min())
    rk_span = float(rk[:3].max() - rk[:3].min())
    print(f"[norm] top-3 spread of the measured pool: zscore={z_span:.5f} "
          f"rank={rk_span:.5f}   (expect zscore ~0 = non-discriminating)")
    ok &= z_span < 0.01 and rk_span > 0.5

    # rank-norm invariants: order preserved, evenly spaced, ties averaged
    mono = np.all(np.diff(_rank_norm(np.array([1.0, 2.0, 3.0, 4.0]))) > 0)
    ends = np.allclose(_rank_norm(np.array([5.0, 1.0, 3.0])), [1.0, -1.0, 0.0])
    ties = np.allclose(_rank_norm(np.array([1.0, 1.0, 2.0])), [-0.5, -0.5, 1.0])
    nan_ok = np.allclose(_rank_norm_nan_safe(np.array([np.nan, 1.0, 2.0])),
                         [0.0, -1.0, 1.0])
    print(f"[norm] monotone={mono} endpoints={ends} ties_averaged={ties} "
          f"nan_neutral={nan_ok}")
    ok &= bool(mono) and ends and ties and nan_ok

    # with rank-norm the smooth right turn must still win (case 3 must not regress)
    tr, _ = rank_candidates([right_turn, jerky_right, left_turn, smooth_straight],
                            command="right", weights=(1.0, 1.0), norm="rank")
    print(f"[norm] rank-norm winner idx={int(np.argmax(tr))} (want 0)")
    ok &= int(np.argmax(tr)) == 0

    # the feasibility veto must survive rank-norm: ranks are O(1), so without the
    # absolute veto a command-favoured impossible candidate could win again
    tv, _ = rank_candidates([smooth_drift, insane_teleport], command="straight",
                            weights=(1.0, 5.0), norm="rank")
    print(f"[norm] rank-norm + veto winner idx={int(np.argmax(tv))} (want 0, smooth)")
    ok &= int(np.argmax(tv)) == 0

    try:
        rank_candidates([right_turn, left_turn], command="right", norm="typo")
        print("[norm] unknown norm did NOT raise -> FAIL")
        ok = False
    except ValueError:
        print("[norm] unknown norm raises ValueError: True")

    print("\nSELF-TEST:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if _selftest() else 1)
