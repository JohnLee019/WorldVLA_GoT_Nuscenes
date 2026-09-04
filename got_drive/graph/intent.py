"""Driving intents: the hypothesis a node branches on.

WHY INTENTS AND NOT TIME SEGMENTS
    The incumbent branches on TIME (0-1 s, 1-2 s, 2-3 s), and sec.1.17 measured
    what that costs: live options per stage are 1.52 / 2.00 / 7.95, and 48% of
    records have exactly ONE option at stage 1. So real deliberation only happens
    in the last segment, while the error that dominates the loss -- longitudinal
    speed, 7.5x the lateral error (sec.1.9a) -- is decided in the first, where
    there is nothing to choose between. "Deliberation happens late, error
    accumulates early."

    Branching on a hypothesis instead moves the choice to the first decision by
    construction.

THE AXES ARE NOT A GUESS -- THEY ARE PRESCRIBED BY MEASUREMENT
    Step 2.11 measured that transplanting the current candidate generator to
    NAVSIM navtrain scores `GAP_dep` -0.0942, i.e. WORSE than a random pool, and
    named the two causes: **zero curvature diversity**, and a speed ladder
    spanning v x 0.60-1.50 when the room is at **v x 0.00-0.45**. Step 2.9/2.10
    then fixed the requirement:

        "diversity in the SPEED PROFILE, curvature as symmetric pairs near 0,
         conservative end (decelerate / stop) MANDATORY"

    with the optimum at v x 0.45-0.60 and the sign of `GAP_dep` flipping between
    v x 1.00 (+0.0295) and v x 1.25 (-0.0594) at N=9. Hence the defaults below.

    ⚠️The parameterisation (speed multiplier x constant curvature) is deliberately
    the same as `navsim_tools/vocab_agent.py`'s, so the nuScenes arm and the
    NAVSIM vocabulary sweeps stay directly comparable. Do not "improve" it into a
    different shape without re-reading Step 2.8's N-dependence -- GAP and the
    transition point are both functions of N.

    ⚠️Curvature must be SYMMETRIC pairs. The "do not do" list is explicit: the
    `k=+0.005` cell that `analyze_vgrid.py [4]` liked sits on an unexplained left
    bias (Step 2.10d), and Step 2.7e had already discarded that same bias once.

THE HARD CONSTRAINT: INTENTS MUST BE BIG ENOUGH TO MOVE THE MODEL
    Our only conditioning channel without retraining is prefix re-conditioning,
    and sec.1.5 measured its floor: prefix perturbations of avgL2 0.25-0.4 m
    flipped the sampled tokens on 0 of 600 records -- three arms landed on
    bit-identical last-segment coordinates. Below roughly a metre the model just
    emits its habitual continuation and the "intents" all realise to the same
    trajectory.

    So `MIN_SEPARATION_M` is not a style preference. `separation_report()`
    measures what the intent set ACTUALLY achieved on this record and the
    operation records it, because an intent arm whose intents did not separate is
    measuring nothing and must say so rather than returning a plausible number.
"""

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

# Multipliers on the ego's own implied first-step displacement. 0.0 is the stop
# anchor Step 2.11 found the payoff concentrated at; 0.5 sits in the measured
# optimum band (v x 0.45-0.60); 1.0 is the model's own plan, kept so the pool
# still contains what the incumbent would have produced.
DEFAULT_SPEED_SCALES: Tuple[float, ...] = (0.0, 0.5, 1.0)

# ★DEFAULT IS SPEED-ONLY, AND THAT IS A MEASUREMENT, NOT A SIMPLIFICATION.
#
# A constant-curvature arc of length L is displaced laterally by about k*L^2/2.
# Against sec.1.5's 1 m floor, over a 2 m first step and over the whole 3 s
# horizon (~12 m at 4 m/s):
#
#       k        L = 2 m (one step)      L = 12 m (full horizon)
#     0.01           0.02 m                    0.72 m
#     0.05           0.10 m                    3.60 m
#
# So NAVSIM's k = +-0.01 cannot separate anything here even if the ENTIRE
# trajectory were committed -- 0.72 m is still under the floor. Those values were
# fitted to a 4 s horizon at nuPlan speeds, where L is far larger; they do not
# transfer to a 3 s nuScenes horizon, and using them because Step 2.10 printed
# them would have produced seven intents realising to one plan.
#
# Curvature is therefore SUPPORTED but not default. To test it honestly you need
# BOTH k = +-0.05 AND a multi-step anchor (`anchor_steps` >= 2), and even then
# separation_report is what says whether it worked on this record -- not this
# comment.
DEFAULT_CURVATURES: Tuple[float, ...] = (0.0,)

# What a curvature axis needs to clear the floor, if you enable one. Kept as a
# named constant so the arithmetic above is not re-derived from memory.
USABLE_CURVATURES: Tuple[float, ...] = (-0.05, 0.0, 0.05)

# Below this, sec.1.5 says the model ignores the prefix. Reported, not enforced:
# a run that fails it is informative (it says the channel is too weak here), but
# it must never be quoted as an intent result.
MIN_SEPARATION_M: float = 1.0


@dataclass(frozen=True)
class Intent:
    """One driving hypothesis.

    A frozen value object with invariants -- which is what a dataclass is FOR.
    Note this is not in tension with `Thought.state` being a plain dict: the dict
    is the graph's storage (upstream fidelity, JSON-serialisable), and an Intent
    is a value that lives inside it. Representation and algorithm are separate
    choices; this is the algorithm one.
    """

    speed_scale: float
    curvature: float

    def __post_init__(self) -> None:
        if self.speed_scale < 0:
            raise ValueError(f"speed_scale must be >= 0, got {self.speed_scale}")
        if self.speed_scale == 0.0 and self.curvature != 0.0:
            # Curving while stopped is not a distinct plan -- every (0.0, k) is
            # the same trajectory. vocab_agent.py hit the identical degeneracy
            # ("정지 5중복") and analyze_vocab_n_dependence.py merges them before
            # sampling subsets, because otherwise duplicate candidates bias the
            # N-dependence curve. Canonicalise instead of deduplicating later.
            raise ValueError("a stopped intent must have curvature 0.0; "
                             "use Intent(0.0, 0.0)")

    @property
    def name(self) -> str:
        return f"v{self.speed_scale:.2f}_k{self.curvature:+.3f}"

    def as_dict(self) -> Dict:
        return {"speed_scale": float(self.speed_scale),
                "curvature": float(self.curvature),
                "name": self.name}


def make_intent_grid(speed_scales: Sequence[float] = DEFAULT_SPEED_SCALES,
                     curvatures: Sequence[float] = DEFAULT_CURVATURES) -> List[Intent]:
    """The intent set, with the stop degeneracy collapsed to a single entry.

    Order is deterministic (speed outer, curvature inner) because it decides the
    generator call order and therefore the RNG stream -- sec.1.4's drifting
    baseline came from exactly this kind of order dependence going unnoticed.
    """
    out: List[Intent] = []
    seen = set()
    for s in speed_scales:
        for k in curvatures:
            k_eff = 0.0 if s == 0.0 else k
            key = (float(s), float(k_eff))
            if key in seen:
                continue
            seen.add(key)
            out.append(Intent(float(s), float(k_eff)))
    return out


def anchor_waypoint(step_len: float, intent: Intent,
                    anchor_steps: int = 1) -> np.ndarray:
    """The waypoints this intent commits to, as (anchor_steps, 2) in the ego frame.

    Constant-curvature arc, `step_len * speed_scale` per step, starting at the
    origin with heading 0. `step_len` is the ego's own implied first-step
    displacement -- RealizeIntent takes it from the model's greedy plan, which is
    what lets this run on the incumbent checkpoint with no ego status. With ego
    status the same anchor is computed from a measured v0 and the intents become
    accelerations in m/s^2, so the >= 1 m floor can be hit by arithmetic instead
    of hoped for.

    `anchor_steps > 1` commits more of the horizon. It exists for the curvature
    axis: lateral displacement grows as k*L^2/2, so a one-step anchor expresses
    speed well and curvature essentially not at all (0.02 m at k = 0.01). Every
    committed step is one the model no longer chooses, so raise it only for the
    axis that needs it.
    """
    n = int(anchor_steps)
    assert n >= 1, "anchor_steps must be >= 1"
    s = float(step_len) * float(intent.speed_scale)
    k = float(intent.curvature)
    if abs(k) < 1e-9 or s == 0.0:
        return np.array([[s * (i + 1), 0.0] for i in range(n)], dtype=np.float64)
    r = 1.0 / k
    return np.array([[r * np.sin(k * s * (i + 1)),
                      r * (1.0 - np.cos(k * s * (i + 1)))] for i in range(n)],
                    dtype=np.float64)


def separation_report(anchors: Sequence[np.ndarray]) -> Dict:
    """How far apart the intents actually put the first waypoint.

    ★Returned and logged on every record, never assumed. sec.1.5 measured that a
    0.25-0.4 m prefix perturbation flips zero tokens out of 600, so an intent set
    whose `min_pairwise` sits under MIN_SEPARATION_M has not created alternatives
    -- it has created one plan under several names, and any selection result from
    it is a result about nothing.

    `min_pairwise` is the honest statistic rather than `max` or `mean`: one wide
    pair does not rescue a set where two intents coincide, because those two are
    the ones the scorer will be asked to tell apart.
    """
    pts = np.asarray([np.asarray(a, dtype=np.float64)[-1, :2] for a in anchors])
    n = len(pts)
    if n < 2:
        return {"n": n, "min_pairwise": 0.0, "max_pairwise": 0.0,
                "separated": False, "min_required": MIN_SEPARATION_M}
    d = [float(np.linalg.norm(pts[i] - pts[j]))
         for i in range(n) for j in range(i + 1, n)]
    out = {
        "n": n,
        "min_pairwise": round(min(d), 4),
        "max_pairwise": round(max(d), 4),
        # The set is only "separated" when EVERY pair clears the floor.
        "separated": bool(min(d) >= MIN_SEPARATION_M),
        "min_required": MIN_SEPARATION_M,
    }
    # Per-axis, because the two axes fail for different reasons and a single
    # number hides which one did. Longitudinal separation comes from the speed
    # multipliers and clears the floor easily; lateral comes from curvature and
    # grows as k*L^2/2, so it needs either a large k or a long committed anchor.
    lon = [abs(float(pts[i][0] - pts[j][0]))
           for i in range(n) for j in range(i + 1, n)]
    lat = [abs(float(pts[i][1] - pts[j][1]))
           for i in range(n) for j in range(i + 1, n)]
    out["max_lon_gap"] = round(max(lon), 4)
    out["max_lat_gap"] = round(max(lat), 4)
    return out
