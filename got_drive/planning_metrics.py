"""
Oracle / selection-quality / tail / significance metrics for the GoT planning eval.

Why these exist
---------------
The headline nuScenes numbers (L2, collision) answer "did GoT beat greedy?".
They do NOT answer "why not?", and that is the question the project is actually
stuck on (PROJECT_HANDOFF §7.3: GoT 4.88 vs baseline 4.58). GoT is a two-part
machine -- a GENERATOR (the base VLA resampled at k temperatures) and a SELECTOR
(the driving score) -- and a single L2 number cannot tell you which half failed.

Splitting them needs one extra ingredient: the candidate pool the selector chose
from (DriveGoTPipeline.final_candidates()). With it:

    minADE_C        = L2 of the BEST candidate in the pool   -> generator ceiling
    selection_gap   = L2(chosen) - L2(best)                  -> selector's loss

    minADE_C < baseline  and  gap large   -> candidates are fine, the SCORE is bad
    minADE_C ~ baseline                   -> the candidates themselves are no
                                             better than greedy (base not
                                             converged / temperatures too tame)

Everything here is pure numpy on already-generated trajectories: no extra model
calls, so it is free to compute on the same eval pass.

Also here
---------
* tail_stats     -- P50/P90/P95/max and the catastrophe rate. The feasibility
                    gate (§10) removes catastrophic candidates (27 m -> 15 m);
                    that shows up in the tail, not in the mean.
* paired_comparison -- GoT vs baseline is an exactly paired design (same record,
                    same checkpoint), and the effect is small (0.30 m), so the
                    mean alone cannot separate signal from noise. Wilcoxon
                    signed-rank (not t-test: the L2 tail is heavy) + bootstrap CI.

Self-test: `python -m got_drive.planning_metrics` (numpy only, no GPU/scipy).
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import numpy as np

# keys emitted by eval_nuscenes.l2_metrics: "L2@1s" (UniAD, exact timestep) and
# "avgL2@1s" (ST-P3/VAD, mean over 0..t). Both conventions are carried through.
_L2_PREFIXES = ("L2@", "avgL2@")


def _l2_keys(metrics: Dict[str, float]) -> List[str]:
    return [k for k in metrics if k.startswith(_L2_PREFIXES)]


# ──────────────────────────────────────────────────────────────────────────
# oracle / selection quality
# ──────────────────────────────────────────────────────────────────────────

def oracle_and_selection(
    cand_metrics: Sequence[Dict[str, float]],
    selected_idx: Optional[int],
    rank_key: str = "avgL2@3s",
) -> Dict[str, float]:
    """Generator ceiling and selector loss over one record's candidate pool.

    cand_metrics : per-candidate dicts from eval_nuscenes.l2_metrics.
    selected_idx : index of the candidate GoT actually committed to.
    rank_key     : the single scalar used to pick the oracle candidate. ONE
                   candidate is chosen and then reported across all horizons --
                   taking a per-horizon min would mix waypoints from different
                   candidates and describe a trajectory nobody could have driven.

    Emits (rounded by the caller):
      oracle_<key>          best candidate's L2 at each horizon
      minADE_C              that candidate's `rank_key` (= minADE over the pool)
      selection_gap_<key>   chosen - oracle  (0 = the score picked perfectly)
      selection_rank        rank of the chosen candidate in the true-L2 order
      selection_top1        1.0 iff the score picked the truly best candidate
      candidate_spread      std of `rank_key` over the pool; ~0 means the
                            temperatures produced no diversity, so GoT had
                            nothing to choose between (e.g. stopped scenes)
      worst_candidate       max `rank_key`; the size of what the feasibility
                            gate has to veto
    """
    if not cand_metrics:
        return {}
    keys = _l2_keys(cand_metrics[0])
    if rank_key not in cand_metrics[0]:
        rank_key = keys[-1] if keys else None
        if rank_key is None:
            return {}

    vals = np.array([m[rank_key] for m in cand_metrics], dtype=np.float64)
    oracle_idx = int(np.argmin(vals))

    out: Dict[str, float] = {
        "n_candidates": float(len(cand_metrics)),
        "minADE_C": float(vals[oracle_idx]),
        "candidate_spread": float(vals.std()),
        "worst_candidate": float(vals.max()),
    }
    for k in keys:
        out[f"oracle_{k}"] = float(cand_metrics[oracle_idx][k])

    if selected_idx is not None and 0 <= selected_idx < len(cand_metrics):
        for k in keys:
            out[f"selection_gap_{k}"] = float(
                cand_metrics[selected_idx][k] - cand_metrics[oracle_idx][k])
        # rank of the chosen candidate once the pool is sorted by TRUE L2
        order = np.argsort(vals, kind="mergesort")
        out["selection_rank"] = float(int(np.where(order == selected_idx)[0][0]))
        out["selection_top1"] = float(out["selection_rank"] == 0.0)
    return out


# ──────────────────────────────────────────────────────────────────────────
# tail
# ──────────────────────────────────────────────────────────────────────────

def tail_stats(values: Sequence[Optional[float]], catastrophe_m: float = 10.0) -> Dict[str, float]:
    """Distribution shape of one L2 key. The mean hides exactly the failure mode
    the feasibility gate targets, so report the quantiles next to it."""
    v = np.asarray([x for x in values if x is not None], dtype=np.float64)
    if v.size == 0:
        return {}
    return {
        "n": float(v.size),
        "mean": float(v.mean()),
        "p50": float(np.percentile(v, 50)),
        "p90": float(np.percentile(v, 90)),
        "p95": float(np.percentile(v, 95)),
        "max": float(v.max()),
        f"frac_gt_{catastrophe_m:g}m": float(np.mean(v > catastrophe_m)),
    }


# ──────────────────────────────────────────────────────────────────────────
# paired significance (no scipy required)
# ──────────────────────────────────────────────────────────────────────────

def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average ranks (1-based), ties shared -- scipy.stats.rankdata equivalent."""
    a = np.asarray(a, dtype=np.float64)
    order = np.argsort(a, kind="mergesort")
    sorted_a = a[order]
    ranks = np.empty(a.size, dtype=np.float64)
    i = 0
    while i < a.size:
        j = i
        while j + 1 < a.size and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def _wilcoxon_p(d: np.ndarray) -> Optional[float]:
    """Two-sided Wilcoxon signed-rank p for paired differences `d`.

    Uses scipy when importable (exact//corrected), else a tie-corrected normal
    approximation so the eval never hard-depends on scipy being installed on
    gpu-server. Zero differences are dropped (Wilcoxon convention).
    """
    d = np.asarray(d, dtype=np.float64)
    d = d[d != 0.0]
    n = d.size
    if n < 6:                       # too few non-ties for either method to mean much
        return None
    try:
        from scipy.stats import wilcoxon           # noqa: WPS433 (optional dep)
        return float(wilcoxon(d, alternative="two-sided", zero_method="wilcox").pvalue)
    except Exception:
        pass
    absd = np.abs(d)
    ranks = _rankdata(absd)
    w_plus = float(ranks[d > 0].sum())
    mu = n * (n + 1) / 4.0
    _, counts = np.unique(absd, return_counts=True)
    tie_term = float(np.sum(counts.astype(np.float64) ** 3 - counts))
    var = n * (n + 1) * (2 * n + 1) / 24.0 - tie_term / 48.0
    if var <= 0:
        return None
    z = (w_plus - mu) / math.sqrt(var)
    return float(math.erfc(abs(z) / math.sqrt(2.0)))       # = 2*(1 - Phi(|z|))


def _bootstrap_ci(d: np.ndarray, n_boot: int = 10000, seed: int = 0,
                  alpha: float = 0.05, chunk: int = 1000) -> List[float]:
    """Percentile bootstrap CI of mean(d).

    Resampled in chunks: the full (n_boot, n) index matrix is ~410 MB of int64
    at n_boot=10000 on the whole val split (n=5119), and the gathered values
    double that. Chunking caps the peak at chunk*n instead.
    """
    d = np.asarray(d, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = d.size
    means = np.empty(n_boot, dtype=np.float64)
    done = 0
    while done < n_boot:
        b = min(chunk, n_boot - done)
        idx = rng.integers(0, n, size=(b, n))
        means[done:done + b] = d[idx].mean(axis=1)
        done += b
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return [float(lo), float(hi)]


def paired_comparison(a: Sequence[float], b: Sequence[float],
                      n_boot: int = 10000, seed: int = 0) -> Dict[str, object]:
    """Paired GoT (`a`) vs baseline (`b`) on the same records.

    NEGATIVE mean_diff = GoT lower L2 = GoT better (same sign convention as the
    existing `got_minus_baseline`). win_rate counts strict wins only, so ties
    (identical trajectories -- common on stopped scenes) count against GoT.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size == 0 or a.size != b.size:
        return {"n_pairs": int(a.size)}
    d = a - b
    return {
        "n_pairs": int(d.size),
        "mean_diff": float(d.mean()),
        "median_diff": float(np.median(d)),
        "win_rate": float(np.mean(d < 0)),
        "tie_rate": float(np.mean(d == 0)),
        "wilcoxon_p": _wilcoxon_p(d),
        "mean_diff_ci95": _bootstrap_ci(d, n_boot=n_boot, seed=seed),
    }


# ──────────────────────────────────────────────────────────────────────────
# across-seed aggregation
# ──────────────────────────────────────────────────────────────────────────

def across_seeds(per_seed: Dict[int, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """{seed: {metric: value}} -> {metric: {mean, std, min, max, n_seeds}}.

    Non-numeric / missing values are skipped so a metric that only some seeds
    produced still aggregates over the seeds that have it.
    """
    keys = sorted({k for d in per_seed.values() for k in d})
    out: Dict[str, Dict[str, float]] = {}
    for k in keys:
        vals = [d[k] for d in per_seed.values()
                if isinstance(d.get(k), (int, float)) and not isinstance(d.get(k), bool)]
        if not vals:
            continue
        v = np.asarray(vals, dtype=np.float64)
        out[k] = {
            "mean": round(float(v.mean()), 4),
            "std": round(float(v.std(ddof=1)) if v.size > 1 else 0.0, 4),
            "min": round(float(v.min()), 4),
            "max": round(float(v.max()), 4),
            "n_seeds": int(v.size),
        }
    return out


# ──────────────────────────────────────────────────────────────────────────
# self-test: numpy only
# ──────────────────────────────────────────────────────────────────────────

def _m(l2_1, l2_2, l2_3):
    """Fake l2_metrics dict; avgL2 built so it is consistent with the L2 points."""
    return {"L2@1s": l2_1, "L2@2s": l2_2, "L2@3s": l2_3,
            "avgL2@1s": l2_1, "avgL2@2s": (l2_1 + l2_2) / 2,
            "avgL2@3s": (l2_1 + l2_2 + l2_3) / 3}


def _selftest():
    ok = True

    # ---- oracle / selection ------------------------------------------------
    cands = [_m(1.0, 2.0, 9.0),    # idx0  avgL2@3s = 4.0
             _m(1.0, 2.0, 3.0),    # idx1  avgL2@3s = 2.0   <- oracle
             _m(5.0, 6.0, 7.0),    # idx2  avgL2@3s = 6.0
             _m(2.0, 3.0, 4.0)]    # idx3  avgL2@3s = 3.0
    r = oracle_and_selection(cands, selected_idx=0)
    assert abs(r["minADE_C"] - 2.0) < 1e-9, r
    assert abs(r["oracle_L2@3s"] - 3.0) < 1e-9, r
    assert abs(r["selection_gap_L2@3s"] - 6.0) < 1e-9, r      # chose 9.0 over 3.0
    assert abs(r["selection_gap_avgL2@3s"] - 2.0) < 1e-9, r   # 4.0 - 2.0
    # true-L2 order by avgL2@3s: idx1(2.0) < idx3(3.0) < idx0(4.0) < idx2(6.0)
    assert r["selection_rank"] == 2.0, r
    assert r["selection_top1"] == 0.0, r
    assert abs(r["worst_candidate"] - 6.0) < 1e-9, r

    perfect = oracle_and_selection(cands, selected_idx=1)
    assert perfect["selection_rank"] == 0.0 and perfect["selection_top1"] == 1.0
    assert all(abs(v) < 1e-12 for k, v in perfect.items() if k.startswith("selection_gap_"))

    # no diversity -> spread 0, gap 0 (GoT literally had nothing to choose)
    flat = oracle_and_selection([_m(1.0, 2.0, 3.0)] * 4, selected_idx=0)
    assert abs(flat["candidate_spread"]) < 1e-12 and flat["selection_top1"] == 1.0

    assert oracle_and_selection([], 0) == {}
    assert "selection_rank" not in oracle_and_selection(cands, None)
    print("  oracle/selection: OK")

    # ---- tail --------------------------------------------------------------
    t = tail_stats([1.0, 2.0, 3.0, 4.0, 50.0], catastrophe_m=10.0)
    assert t["n"] == 5 and abs(t["p50"] - 3.0) < 1e-9 and abs(t["max"] - 50.0) < 1e-9
    assert abs(t["frac_gt_10m"] - 0.2) < 1e-9, t
    assert tail_stats([]) == {}
    assert tail_stats([1.0, None, 3.0])["n"] == 2
    print("  tail_stats: OK")

    # ---- rankdata ----------------------------------------------------------
    assert np.allclose(_rankdata(np.array([10.0, 20.0, 20.0, 30.0])), [1.0, 2.5, 2.5, 4.0])
    print("  rankdata (ties): OK")

    # ---- paired comparison -------------------------------------------------
    rng = np.random.default_rng(0)
    base = rng.normal(5.0, 1.0, 200)

    better = base - 0.5                       # GoT uniformly better
    pc = paired_comparison(better, base, n_boot=2000, seed=1)
    assert pc["n_pairs"] == 200 and abs(pc["mean_diff"] + 0.5) < 1e-9
    assert pc["win_rate"] == 1.0
    assert pc["wilcoxon_p"] is not None and pc["wilcoxon_p"] < 1e-6, pc
    lo, hi = pc["mean_diff_ci95"]
    assert hi < 0.0, f"CI must exclude 0 for a real effect: {pc}"

    same = base + rng.normal(0.0, 1.0, 200) * 0.0   # identical -> all ties
    pc2 = paired_comparison(same, base, n_boot=500, seed=1)
    assert pc2["tie_rate"] == 1.0 and pc2["wilcoxon_p"] is None, pc2

    noise = base + rng.normal(0.0, 1.0, 200)        # no systematic difference
    pc3 = paired_comparison(noise, base, n_boot=2000, seed=1)
    assert pc3["wilcoxon_p"] > 0.05, pc3
    lo3, hi3 = pc3["mean_diff_ci95"]
    assert lo3 < 0.0 < hi3, f"CI should straddle 0 for noise: {pc3}"
    assert paired_comparison([], []) == {"n_pairs": 0}
    print("  paired_comparison: OK")

    # ---- across seeds ------------------------------------------------------
    agg = across_seeds({42: {"L2@3s": 4.0, "x": 1.0}, 43: {"L2@3s": 5.0},
                        44: {"L2@3s": 6.0, "x": 3.0}})
    assert agg["L2@3s"]["mean"] == 5.0 and agg["L2@3s"]["n_seeds"] == 3
    assert abs(agg["L2@3s"]["std"] - 1.0) < 1e-9, agg          # ddof=1
    assert agg["x"]["n_seeds"] == 2
    print("  across_seeds: OK")

    print("planning_metrics self-test:", "OK" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if _selftest() else 1)
