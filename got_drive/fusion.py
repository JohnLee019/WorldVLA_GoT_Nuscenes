"""Combine candidate trajectories instead of selecting one of them.

Why this exists
---------------
Every intervention tried so far asked the score to RANK better: weight sweeps,
z-norm, kinematic-only selection, self-likelihood, a learned linear rule, a
learned nonlinear rule, a learned fallback detector. All null. The learned
ones bound the ceiling: a fitted nonlinear combination of every logged signal
reaches within-pool rho 0.524 against 0.519 for the single best component, and
selecting with it still lands 0.029 m worse than not deliberating at all.

Fusion asks a different question, and it is the only untried one that does not
depend on ranking ability:

  1. The metric is a DISTANCE (avgL2), and the point estimator that minimises
     expected distance is the geometric median of the predictive distribution,
     not its mode. Greedy decoding returns approximately the mode, so a gap
     exists that requires no ability to tell candidates apart.
  2. Measurement says GoT adds near-symmetric zero-mean noise around greedy
     (42.7% worse / 38.7% better / 18.7% identical). Median and mean are the
     operations that cancel zero-mean noise.
  3. The absorption law is an identity over `output = minADE_C + selection_gap`.
     With no selection there is no gap term, so there is nothing to absorb.

Median is the default: it is the estimator matched to the metric, and it is
markedly more robust than the mean to the classic failure of this idea --
averaging "turn left" and "turn right" into "drive into the barrier". That
failure is a real risk here and must be checked per command, not just in
aggregate.

Note the candidates are NOT independent samples: beam search shares prefixes,
so with beam_width=2 and k=4 the eight final candidates come from two distinct
first-two-segment paths and differ only in the last segment. Variance
cancellation is therefore diluted for the early waypoints -- which is exactly
why per-segment fusion (fuse_scope="segment") exists alongside final fusion.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

FUSE_MODES = ("median", "mean")


def fuse_trajectories(trajs: Sequence[np.ndarray],
                      mode: str = "median") -> Optional[np.ndarray]:
    """Componentwise combination of equal-length trajectories.

    trajs : list of (T, 2) arrays, all the same shape.
    mode  : "median" (default, matched to an L2-distance metric and robust to
            one candidate belonging to a different mode) or "mean".

    Returns (T, 2), or None when the input is empty or ragged. A single
    trajectory is returned unchanged -- fusing one candidate must be exactly
    "select that candidate", so fuse_top_m=1 is a valid no-op control arm.
    """
    if not trajs:
        return None
    arrs = [np.asarray(t, dtype=np.float64) for t in trajs]
    shape = arrs[0].shape
    if any(a.shape != shape or a.ndim != 2 for a in arrs):
        return None
    if len(arrs) == 1:
        return arrs[0].copy()
    stack = np.stack(arrs, axis=0)          # (n, T, 2)
    if mode == "mean":
        return stack.mean(axis=0)
    if mode == "median":
        # componentwise median, not the geometric (spatial) median: it is the
        # separable estimator, it is what a per-coordinate L1 fit gives, and it
        # needs no iterative solver. For 2-D waypoints the two differ only
        # slightly, and the difference is far below this project's resolution
        # (scene-clustered CI half-width ~0.031 m).
        return np.median(stack, axis=0)
    raise ValueError(f"unknown fuse mode {mode!r}; expected one of {FUSE_MODES}")


def top_m_indices(scores: Sequence[float], m: int) -> List[int]:
    """Indices of the m highest scores, best first; m<=0 means all.

    Ties keep generation order (stable sort), matching the beam's convention so
    a fused arm and a selecting arm see the same short-list on ties.
    """
    s = np.asarray(scores, dtype=np.float64)
    n = len(s)
    if n == 0:
        return []
    keep = n if m <= 0 else min(m, n)
    order = sorted(range(n), key=lambda i: (-s[i], i))
    return order[:keep]


def _selftest():
    # a single candidate must pass through untouched: fuse_top_m=1 is the
    # no-op control arm and must reproduce plain selection exactly
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    out = fuse_trajectories([a], "median")
    assert np.array_equal(out, a) and out is not a, "must copy, not alias"
    print("  ok  single candidate passes through (fuse_top_m=1 == selection)")

    # median ignores an outlier that the mean chases -- the property this
    # module is chosen for
    b = np.array([[1.0, 0.0], [2.0, 0.0]])
    c = np.array([[1.0, 0.2], [2.0, 0.2]])
    d = np.array([[1.0, 9.0], [2.0, 9.0]])          # different mode
    med = fuse_trajectories([b, c, d], "median")
    mean = fuse_trajectories([b, c, d], "mean")
    assert abs(med[0, 1] - 0.2) < 1e-12, med
    assert abs(mean[0, 1] - 3.0666666666) < 1e-6, mean
    print(f"  ok  median rejects an off-mode candidate "
          f"({med[0,1]:.2f}) where the mean chases it ({mean[0,1]:.2f})")

    # exact values on an even count (numpy averages the two middles)
    e = fuse_trajectories([np.array([[0.0, 0.0]]), np.array([[2.0, 4.0]])], "median")
    assert np.allclose(e, [[1.0, 2.0]]), e
    print("  ok  even counts average the two middle values")

    # ragged / empty input must be reported, never silently truncated
    assert fuse_trajectories([]) is None
    assert fuse_trajectories([a, np.array([[1.0, 2.0]])]) is None
    print("  ok  empty and ragged input return None")

    # fusion is a convex/interior operation: every fused coordinate lies inside
    # the candidates' range, so it can never invent a waypoint beyond the pool
    rng = np.random.RandomState(0)
    ts = [rng.randn(6, 2) for _ in range(5)]
    for mode in FUSE_MODES:
        f = fuse_trajectories(ts, mode)
        lo, hi = np.min(np.stack(ts), 0), np.max(np.stack(ts), 0)
        assert np.all(f >= lo - 1e-12) and np.all(f <= hi + 1e-12), mode
    print("  ok  fused output stays within the candidates' envelope")

    idx = top_m_indices([0.1, 0.9, 0.5, 0.9], 2)
    assert idx == [1, 3], idx                     # tie -> generation order
    assert top_m_indices([0.1, 0.9], 0) == [1, 0]
    assert top_m_indices([0.1, 0.9], 99) == [1, 0]
    assert top_m_indices([], 3) == []
    print("  ok  top_m_indices: ties stable, m<=0 and m>n mean all")

    try:
        fuse_trajectories([a, a], "geometric")
    except ValueError:
        print("  ok  unknown mode raises instead of silently averaging")
    else:
        raise AssertionError("unknown mode must raise")

    print("\nfusion self-test PASS")


if __name__ == "__main__":
    _selftest()
