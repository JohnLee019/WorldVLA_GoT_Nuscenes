"""Locating and repairing a feasibility violation.

WHY A VIOLATION IS A FIRST-CLASS OBJECT HERE
    `scoring_driving._feasible` already knows WHICH limit a trajectory breaks and
    at which waypoint -- and throws all of it away, returning one bool that the
    ranker turns into a -1e6 penalty. sec.1.5 had to recover that information by
    hand to learn something the pipeline could have logged: **acceleration is
    71-74% of all violations and they concentrate at index 3-4, the segment 2->3
    seam.** A tree has nowhere to put that fact; a graph makes it a vertex, and
    an Improve operation consumes it.

★THE INVARIANT THAT MAKES THE IMPROVE LOOP SAFE
    `locate_violation(t) is None` must be exactly equivalent to `_feasible(t)`.
    If the two disagreed, ValidateAndImprove would either spin (repairing what
    the veto still rejects) or declare victory on a trajectory the scorer will
    veto anyway. So the checks below are a transcription of `_feasible`'s, in the
    same order, against the same imported constants -- not a re-derivation -- and
    selftest.py asserts the equivalence on random trajectories rather than
    trusting this comment.
"""

from typing import Dict, Optional

import numpy as np

from got_drive.scoring_driving import (
    DEFAULT_DT,
    FEAS_A_MAX,
    FEAS_LAT_STEP_MAX,
    FEAS_V_MAX,
    _finite_diff,
    _with_origin,
)


def locate_violation(traj, dt: float = DEFAULT_DT, include_origin: bool = True,
                     v_max: float = FEAS_V_MAX, a_max: float = FEAS_A_MAX,
                     lat_step_max: float = FEAS_LAT_STEP_MAX) -> Optional[Dict]:
    """The FIRST limit this trajectory breaks, or None if it breaks none.

    Returns {axis, index, value, limit} where `index` indexes into `traj` itself
    (origin excluded), i.e. the waypoint to repair.

    Check order mirrors `_feasible` exactly: speed, then per-step lateral jump,
    then acceleration.
    """
    t = _with_origin(traj, include_origin)
    n_pad = 1 if include_origin else 0
    if t.shape[0] < 2:
        return None

    v = _finite_diff(t, dt)                       # (N-1, 2), v[i] enters t[i+1]
    speed = np.linalg.norm(v, axis=1)
    if np.any(speed > v_max):
        i = int(np.argmax(speed > v_max))
        return {"axis": "speed", "index": max(i + 1 - n_pad, 0),
                "value": float(speed[i]), "limit": float(v_max)}

    dy = np.abs(np.diff(t[:, 1]))
    if np.any(dy > lat_step_max):
        i = int(np.argmax(dy > lat_step_max))
        return {"axis": "lateral_step", "index": max(i + 1 - n_pad, 0),
                "value": float(dy[i]), "limit": float(lat_step_max)}

    if t.shape[0] >= 3:
        a = np.linalg.norm(_finite_diff(v, dt), axis=1)   # a[i] enters t[i+2]
        if np.any(a > a_max):
            i = int(np.argmax(a > a_max))
            return {"axis": "acceleration", "index": max(i + 2 - n_pad, 0),
                    "value": float(a[i]), "limit": float(a_max)}
    return None


def repair_prefix(traj, violation: Dict, dt: float = DEFAULT_DT,
                  include_origin: bool = True) -> Optional[np.ndarray]:
    """A corrected prefix ending at the violating waypoint, or None if not repairable.

    The violating step is CLAMPED to the limit that it broke, keeping its
    direction, and everything after it is dropped for the model to regenerate.

    ★WHY CLAMPING RATHER THAN TRUNCATING. Truncating and re-generating hands the
    model back a prefix it already produced, and sec.1.5 measured what that
    achieves: prefix perturbations of avgL2 0.25-0.4 m flipped the sampled tokens
    on **0 of 600** records. Re-rolling from an unchanged prefix is a no-op with
    extra cost. Clamping actually MOVES the prefix, and because violations are
    gross by construction (a 12 m/s^2 limit is only broken by a near-teleport)
    the move is usually large enough to matter -- which the operation measures
    per record instead of assuming.
    """
    t = np.asarray(traj, dtype=np.float64)
    k = int(violation["index"])
    if k < 0 or k >= t.shape[0]:
        return None

    full = _with_origin(t, include_origin)
    n_pad = 1 if include_origin else 0
    j = k + n_pad                                  # index of the bad point in `full`
    if j < 1:
        return None
    prev = full[j - 1]
    step = full[j] - prev
    axis = violation["axis"]

    if axis == "speed":
        max_step = violation["limit"] * dt
    elif axis == "acceleration":
        # v_prev is the step that ENTERED prev; the clamp keeps |v - v_prev| <= a*dt
        v_prev = (prev - full[j - 2]) if j >= 2 else np.zeros(2)
        target = v_prev + _clip_norm(step / dt - v_prev, violation["limit"] * dt)
        fixed = prev + target * dt
        return np.vstack([t[:k], fixed[None, :]]) if k else fixed[None, :]
    elif axis == "lateral_step":
        fixed = prev + np.array([step[0],
                                 np.sign(step[1]) * min(abs(step[1]),
                                                        violation["limit"])])
        return np.vstack([t[:k], fixed[None, :]]) if k else fixed[None, :]
    else:
        return None

    fixed = prev + _clip_norm(step, max_step)
    return np.vstack([t[:k], fixed[None, :]]) if k else fixed[None, :]


def _clip_norm(vec: np.ndarray, max_norm: float) -> np.ndarray:
    """Scale `vec` down to `max_norm` if it is longer, keeping its direction."""
    n = float(np.linalg.norm(vec))
    if n <= max_norm or n == 0.0:
        return vec
    return vec * (max_norm / n)
