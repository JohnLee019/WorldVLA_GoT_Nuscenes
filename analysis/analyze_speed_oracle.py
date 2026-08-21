#!/usr/bin/env python
"""Ceiling test for the PAST-FRAMES track: if the selector knew the ego's
near-term motion exactly, would deliberation finally beat greedy? No GPU, no
model, no new eval, no preprocessing.

Why this exists
---------------
§1.9 measured that the error is 7.5x longitudinal (3s: 6.011 m vs 0.797 m) and
that R^2(e_5 ~ 6*e_0) = 0.77 -- most of the final error variance is already
determined by the first 0.5 s. A single front frame cannot observe ego speed, so
the dominant axis is unobservable BY CONSTRUCTION in the current setup. That is
the one mechanism §1.1's six independent failure cases never ruled out: every
signal ever tried (kinematic, command, likelihood, geometry, WM, collision) was
derivable from what the generator had already seen, so redundancy was
guaranteed. Observed ego motion is the first candidate for a genuinely NEW axis.

The obvious next step is to retrain on 2 past frames. That costs re-preprocessing
+ ~33 h of training + the full §1 re-run chain (headline 3 seed + fusion + WM),
and it reverses a §8 decision. §11.8 says the project got it right three times by
measuring the ceiling first -- collision (30 min saved weeks), geometry reranker
(5 min saved 7-20 GPU-h), learn_selector (bounded the rule defect at 0.011 m).
This is the same move for past frames.

★ WHAT MAKES THIS AN UPPER BOUND, AND A GENEROUS ONE
----------------------------------------------------
The oracle signals below are read off GT, so they are STRICTLY BETTER than
anything two past frames could supply, and they are given to the SELECTOR ONLY
while the generator stays exactly as it is. That asymmetry is deliberate:
deliberation can only pay when the verifier knows something the generator does
not (§1.1), so handing past frames to BOTH -- which is what retraining actually
does -- is a weaker configuration than the one measured here. If the ceiling
measured under these conditions does not clear greedy, retraining cannot either.

⚠ Consequently NOTHING here is a reportable result on its own. §8: a signal that
reads GT at inference time is reported as an ORACLE UPPER BOUND, never as a
method. The same rule that governs analyze_candidate_collision.py governs this.

The blocks
----------
  [2] POOL SPAN (decides which verdict is even reachable). A selector can only
      pick from what was generated. If the candidates barely differ in their
      first-step longitudinal displacement, no motion signal -- perfect or not --
      can move the output, and the finding is about the GENERATOR, not about
      deliberation. Coverage is the honest form of the question: does GT's own
      first waypoint fall inside the range the pool spans?
  [3] ★ SELECTION under oracle motion signals, with a NOISE SWEEP. Perfect
      knowledge is not obtainable; two frames at 2 Hz give a speed estimate with
      some error. A gain that exists only at sigma = 0 is not a plan, so the
      sweep is part of the test rather than an afterthought, and the verdict
      reads the REALISTIC column.
  [4] ★ CORRECTION vs SELECTION -- the strategic split. Rescale each candidate's
      longitudinal profile so its first step matches GT, i.e. give the same
      information to the GENERATOR instead of the selector. If correction is
      large where selection is null, the answer to "should I add past frames?"
      is YES FOR ABSOLUTE L2 and NO FOR GoT: the deliberation layer stays behind
      and §1's conclusion survives the input change. That is a different, and
      more useful, answer than a flat pass/fail.

Reference points (600 records / 150 scenes, deterministic crop, seed 42):
greedy 3.5557, score pick 3.5954, random candidate 4.6560, oracle 2.9770.
Scene-cluster CI half-width is ~0.031, so anything under ~0.05 m is not a result.

Usage
-----
    # the pool columns exist only in results/fusion/*; final_top3 is the arm
    # whose pool is identical to ref (d_pool exactly 0, 20 calls)
    python analyze_speed_oracle.py results/fusion/final_top3/per_sample.csv \
        --records_json ./data/nuscenes_records/nuscenes_val_scenespread.json

    python analyze_speed_oracle.py --selftest        # no data needed

Run it from the repo root (got_drive/ must be importable as a package).
"""

from __future__ import annotations
# --- 리포 루트를 import 경로에 넣는다 -------------------------------------
# 이 파일은 2026-08-21에 루트에서 이 폴더로 옮겨졌다. 파이썬은 sys.path[0]에
# *스크립트가 있는 폴더*를 넣으므로, 이 두 줄이 없으면 `python analysis/x.py`가
# got_drive / model / xllmx 를 못 찾고 ModuleNotFoundError로 죽는다.
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
# -------------------------------------------------------------------------

import argparse
import sys
from collections import defaultdict

import numpy as np

from analyze_got_csv import cluster_bootstrap_ci, wilcoxon_p
from analyze_oracle_structure import (HZ, avg_l2, block_wiring, load_gt,
                                      load_pools)

try:
    from got_drive.scoring_driving import DEFAULT_DT, _feasible
except ImportError as e:                                    # pragma: no cover
    sys.exit(f"[fatal] run this from the repo root, not from got_drive/: {e}")

# preregistered decision thresholds -- fixed here before the run so the verdict
# cannot be talked into existence after seeing the numbers (§11.5)
T_PRIZE = 0.05          # scene-cluster noise floor; below this is not a result
T_GEN = 0.10            # a generator-side fix has to be clearly bigger to be
                        # worth the re-run chain, so it gets a stricter bar
T_COVERAGE = 0.50       # share of records whose GT first step the pool spans
SIGMA_REALISTIC = 0.5   # m/s. What 2 frames at 2 Hz can plausibly resolve; the
                        # verdict reads this column, not the sigma = 0 one.
SIGMA_SWEEP = (0.0, 0.25, 0.5, 1.0, 2.0)
N_NOISE_DRAWS = 16      # per record, per sigma; averaged, so the reported gain
                        # is the EXPECTED performance of a noisy estimator
N_CONTROL_DRAWS = 64    # the uninformative control is a uniform pick, so its
                        # Monte-Carlo error must be pushed well under T_PRIZE or
                        # the wiring check it performs cannot fire reliably
MIN_LON_FOR_SCALE = 0.5  # m. Below this the record is parked and a ratio
                         # rescale is numerically meaningless.


# --------------------------------------------------------------------------- #
# statistics (scene-clustered, per §9 -- record-level p is inflated ~34x)
# --------------------------------------------------------------------------- #

def _scene_means(pools, diffs):
    g = defaultdict(list)
    for p, d in zip(pools, diffs):
        g[p["scene"]].append(float(d))
    return g


def _ci(pools, diffs, n_boot=5000):
    return cluster_bootstrap_ci(_scene_means(pools, diffs), n_boot=n_boot)


def _psc(pools, diffs):
    """Wilcoxon over SCENE means, the only p this project reports."""
    sm = _scene_means(pools, diffs)
    return wilcoxon_p([float(np.mean(v)) for v in sm.values()])


def _stat(pools, vals, greedy, n_boot):
    d = np.asarray(vals) - np.asarray(greedy)
    lo, hi = _ci(pools, d, n_boot)
    return float(np.mean(vals)), float(d.mean()), lo, hi, _psc(pools, d)


# --------------------------------------------------------------------------- #
# [2] pool span -- can a selector express the fix at all?
# --------------------------------------------------------------------------- #

def block_span(pools):
    """Does the candidate pool actually span the axis the error lives on?

    §1.8 found the good candidates are the decode's own neighbours
    (displacement -0.1238), which says the pool is a lottery AROUND the output.
    That leaves open whether the lottery is wide enough on the longitudinal axis
    to contain the right answer at all. Coverage answers it directly and needs
    no null: either GT's first step falls inside [min, max] of what the pool
    proposed, or no selection rule can ever reach it.

    ★ Read the two numbers together. Low coverage with a large first-step error
    means the whole pool is displaced the same way -- a GENERATOR fault that
    deliberation is structurally unable to touch, whatever signal it is given.
    """
    print(f"\n{'=' * 78}\n[2] POOL SPAN -- can a selector express a speed fix?\n{'=' * 78}")
    cov, span, err0, err5, spread5 = [], [], [], [], []
    for p in pools:
        c0 = p["wps"][:, 0, 0]                       # first-step longitudinal
        g0 = float(p["gt"][0, 0])
        cov.append(float(c0.min()) <= g0 <= float(c0.max()))
        span.append(float(c0.max() - c0.min()))
        err0.append(abs(float(np.median(c0)) - g0))
        err5.append(abs(float(np.median(p["wps"][:, HZ, 0])) - float(p["gt"][HZ, 0])))
        spread5.append(float(p["wps"][:, HZ, 0].std()))
    cov = float(np.mean(cov))
    print(f"\n  first-step (0.5 s) longitudinal displacement, per record")
    print(f"    pool span  max-min          {np.mean(span):8.4f} m")
    print(f"    |pool median - GT|          {np.mean(err0):8.4f} m")
    print(f"    GT inside the pool's range  {cov:8.1%}   <- coverage")
    print(f"\n  final (3 s) longitudinal")
    print(f"    pool spread (std)           {np.mean(spread5):8.4f} m")
    print(f"    |pool median - GT|          {np.mean(err5):8.4f} m")
    ratio = float(np.mean(spread5)) / max(float(np.mean(err5)), 1e-9)
    print(f"    spread / error              {ratio:8.3f}")
    print()
    if cov < T_COVERAGE:
        print(f"  -> COVERAGE IS LOW ({cov:.1%} < {T_COVERAGE:.0%}). The pool mostly does not")
        print("     contain a candidate with the right initial speed, so the")
        print("     selection blocks below are bounded by generation, not by the")
        print("     signal. Block [4] is the one that matters in this regime.")
    else:
        print(f"  -> coverage {cov:.1%}: the pool does span GT's first step often")
        print("     enough that a perfect motion signal has something to pick.")
    return {"coverage": cov, "span": float(np.mean(span)),
            "err0": float(np.mean(err0)), "err5": float(np.mean(err5)),
            "spread5": float(np.mean(spread5))}


# --------------------------------------------------------------------------- #
# [3] selection under oracle motion signals
# --------------------------------------------------------------------------- #

def _pick(pools, keys):
    """argmin of a per-pool key array -> (picked value, picked trajectory)."""
    vals, trajs = [], []
    for p, k in zip(pools, keys):
        i = int(np.argmin(k))
        vals.append(float(p["vals"][i]))
        trajs.append(p["wps"][i])
    return np.asarray(vals), trajs


def _keys_wp0_full(pools, noise, rng):
    out = []
    for p in pools:
        obs = p["gt"][0] + rng.randn(2) * noise
        out.append(np.linalg.norm(p["wps"][:, 0, :] - obs, axis=-1))
    return out


def _keys_speed(pools, noise, rng):
    """Rank by |candidate first-step SPEED - observed speed|.

    Magnitude only, no direction: this is the honest shape of what visual
    odometry over two frames hands you, and it is strictly weaker than knowing
    the first waypoint outright.
    """
    out = []
    for p in pools:
        obs = float(np.linalg.norm(p["gt"][0])) + rng.randn() * noise
        out.append(np.abs(np.linalg.norm(p["wps"][:, 0, :], axis=-1) - obs))
    return out


def _keys_lon(pools, noise, rng):
    out = []
    for p in pools:
        obs = float(p["gt"][0, 0]) + rng.randn() * noise
        out.append(np.abs(p["wps"][:, 0, 0] - obs))
    return out


def _keys_random(pools, noise, rng):
    """Control: a key with no information. Must land on the random-candidate
    row, which is what proves the rule machinery is not leaking GT by itself."""
    return [rng.rand(len(p["wps"])) for p in pools]


RULES = (("wp0 full (2D)", _keys_wp0_full),
         ("speed magnitude", _keys_speed),
         ("longitudinal only", _keys_lon),
         ("random key (control)", _keys_random))


def block_selection(pools, n_boot=5000, rng_seed=0):
    """Give the SELECTOR perfect (then noisy) knowledge of the first 0.5 s.

    The bar is greedy, exactly as in §1.3's table -- not the score pick, and not
    a random candidate. Every rule here is an exact offline counterfactual: the
    pool and its true per-candidate errors are both logged, so no approximation
    enters (§1.3).

    ⚠ The noise sweep is not a robustness footnote, it IS the test. Position
    noise is sigma_v * dt, so a 1 m/s speed error moves the observed first
    waypoint by 0.5 m -- comparable to the pool's own spread. A gain that
    survives only at sigma = 0 describes an oracle, not a system.
    """
    print(f"\n{'=' * 78}\n[3] SELECTION under oracle motion signals\n{'=' * 78}")
    greedy = np.array([p["base"] for p in pools])
    orc = np.array([p["vals"].min() for p in pools])
    rnd = np.array([p["vals"].mean() for p in pools])
    lon_of = lambda t, p: abs(float(t[HZ, 0]) - float(p["gt"][HZ, 0]))

    print(f"\n  reference rows")
    print(f"    {'greedy free-run':<24} {greedy.mean():>9.4f}")
    keep = [p for p in pools if not p["fused"]]
    if keep:
        sp = np.array([p["vals"][0] for p in keep])
        print(f"    {'score pick (the arm)':<24} {sp.mean():>9.4f} "
              f"{sp.mean() - np.mean([p['base'] for p in keep]):>+10.4f}")
    print(f"    {'random candidate':<24} {rnd.mean():>9.4f} "
          f"{rnd.mean() - greedy.mean():>+10.4f}")
    print(f"    {'oracle (minADE_C)':<24} {orc.mean():>9.4f} "
          f"{orc.mean() - greedy.mean():>+10.4f}   <- in-sample, needs the true L2")

    print(f"\n  ORACLE UPPER BOUNDS -- these read GT at selection time (sec.8:")
    print(f"  report as a ceiling, never as a method)")
    print(f"\n  {'rule':<22} {'sigma_v':>8} {'avgL2@3s':>9} {'vs greedy':>10} "
          f"{'ci_sc':>21} {'p_sc':>7} {'lon@3s':>8}")

    table = {}
    for name, fn in RULES:
        for sig in SIGMA_SWEEP:
            if name.startswith("random") and sig != 0.0:
                continue                      # the control has no signal to blur
            rng = np.random.RandomState(rng_seed)
            pos_noise = sig * DEFAULT_DT
            # the noiseless oracle rules are deterministic, so one draw is
            # exact; the control is a random pick and needs averaging, or its
            # own sampling error (pool spread / sqrt(n)) swamps the comparison
            # it exists to make
            deterministic = sig == 0.0 and not name.startswith("random")
            draws = (1 if deterministic else
                     N_CONTROL_DRAWS if name.startswith("random") else
                     N_NOISE_DRAWS)
            acc = np.zeros(len(pools))
            acc_lon = np.zeros(len(pools))
            for _ in range(draws):
                v, tj = _pick(pools, fn(pools, pos_noise, rng))
                acc += v
                acc_lon += np.array([lon_of(t, p) for t, p in zip(tj, pools)])
            vals, lon = acc / draws, acc_lon / draws
            m, d, lo, hi, p = _stat(pools, vals, greedy, n_boot)
            table[(name, sig)] = {"mean": m, "d": d, "ci": (lo, hi), "p_sc": p}
            print(f"  {name:<22} {sig:>8.2f} {m:>9.4f} {d:>+10.4f} "
                  f"[{lo:>+9.4f},{hi:>+9.4f}] {p:>7.4f} {lon.mean():>8.4f}")

    ctl = table[("random key (control)", 0.0)]["mean"]
    if abs(ctl - rnd.mean()) > 0.05:
        print(f"\n  WARN control ({ctl:.4f}) does not match the random-candidate row "
              f"({rnd.mean():.4f}); the rule machinery may be leaking information.")
    else:
        print(f"\n  ok  the uninformative control lands on the random-candidate row "
              f"({ctl:.4f} vs {rnd.mean():.4f})")

    real = [(n, table[(n, SIGMA_REALISTIC)]) for n, _ in RULES
            if (n, SIGMA_REALISTIC) in table and not n.startswith("random")]
    perf = [(n, table[(n, 0.0)]) for n, _ in RULES if not n.startswith("random")]
    best_perf = min(perf, key=lambda kv: kv[1]["d"])
    best_real = min(real, key=lambda kv: kv[1]["d"]) if real else best_perf
    print(f"\n  best at sigma=0            {best_perf[0]:<22} {best_perf[1]['d']:+.4f} m")
    print(f"  best at sigma={SIGMA_REALISTIC} m/s (realistic)  "
          f"{best_real[0]:<12} {best_real[1]['d']:+.4f} m")
    return {"table": table, "best_perfect": best_perf, "best_realistic": best_real,
            "greedy": float(greedy.mean()), "oracle": float(orc.mean()),
            "random": float(rnd.mean())}


# --------------------------------------------------------------------------- #
# [4] correction -- the same information given to the generator instead
# --------------------------------------------------------------------------- #

def _corrected(traj, gt, min_lon=MIN_LON_FOR_SCALE):
    """Rescale the longitudinal profile so the first step matches GT's.

    This is the cheapest stand-in for "the model knew its own speed": the SHAPE
    of the trajectory is kept exactly and only the longitudinal scale is set by
    the observed first step. It is not what a retrained model would produce, but
    it isolates the level-vs-shape split, which is the thing §1.9(c) says
    dominates (R^2 0.77 of the final error is fixed in the first 0.5 s).

    Returns the trajectory unchanged for parked records, where the ratio is
    numerically meaningless.
    """
    c0 = float(traj[0, 0])
    if abs(c0) < min_lon:
        return traj, False
    out = traj.copy()
    out[:, 0] = traj[:, 0] * (float(gt[0, 0]) / c0)
    return out, True


def block_correction(pools, n_boot=5000):
    """Selector-side vs generator-side: which one does the information belong to?

    ★ This is the block that answers the actual decision. Past frames can be fed
    to the generator (retraining) or used to rank what the generator already
    produced (deliberation). Those are different projects with different costs
    and different conclusions for §1:

      correction large, selection null -> add past frames for ABSOLUTE L2, and
          expect GoT to stay behind: the fix is not expressible as a choice among
          the candidates, so the deliberation layer never gets to make it. §1's
          conclusion survives the input change.
      both large -> the pool spans the fix and a signal can find it: the past
          frames track can plausibly flip the GoT result, which is the only
          configuration that justifies the re-run chain.

    ⚠ A corrected trajectory is not a pool member and never passed the
    feasibility veto, so the infeasible fraction is printed on the §1.5
    convention (reference rate for non-fused arms is 2.5%).
    """
    print(f"\n{'=' * 78}\n[4] CORRECTION -- the same knowledge given to the generator\n{'=' * 78}")
    greedy = np.array([p["base"] for p in pools])
    rows, skipped_any = [], 0

    def add(name, trajs_of):
        nonlocal skipped_any
        vals, infeas, n_skip = [], 0, 0
        for p in pools:
            t = trajs_of(p)
            t2, ok = _corrected(t, p["gt"])
            n_skip += (not ok)
            vals.append(avg_l2(t2, p["gt"]))
            infeas += (not _feasible(t2))
        skipped_any = max(skipped_any, n_skip)
        m, d, lo, hi, ps = _stat(pools, vals, greedy, n_boot)
        rows.append((name, m, d, lo, hi, ps, infeas / len(pools)))

    add("score pick, corrected", lambda p: p["wps"][0])
    add("pool median, corrected", lambda p: np.median(p["wps"], axis=0))
    # what the correction is worth WITHOUT the correction, so the gain is
    # attributable to the rescale rather than to the row it was applied to
    unc = np.array([p["vals"][0] for p in pools])
    m0, d0, lo0, hi0, p0 = _stat(pools, unc, greedy, n_boot)

    print(f"\n  {'trajectory':<24} {'avgL2@3s':>9} {'vs greedy':>10} "
          f"{'ci_sc':>21} {'p_sc':>7} {'infeas':>7}")
    print(f"  {'greedy free-run':<24} {greedy.mean():>9.4f} {'--':>10} "
          f"{'--':>21} {'--':>7} {'2.5%':>7}")
    print(f"  {'score pick, as-is':<24} {m0:>9.4f} {d0:>+10.4f} "
          f"[{lo0:>+9.4f},{hi0:>+9.4f}] {p0:>7.4f}")
    for name, m, d, lo, hi, ps, inf in rows:
        print(f"  {name:<24} {m:>9.4f} {d:>+10.4f} "
              f"[{lo:>+9.4f},{hi:>+9.4f}] {ps:>7.4f} {inf:>6.1%}")
    if skipped_any:
        print(f"\n  note {skipped_any}/{len(pools)} records are parked "
              f"(|first step| < {MIN_LON_FOR_SCALE} m) and were left uncorrected")

    best = min(r[2] for r in rows)
    print(f"\n  -> best correction: {best:+.4f} m vs greedy "
          f"(negative = better).")
    print("     This is what the SAME information is worth when it reaches the")
    print("     trajectory itself instead of only the ranking over candidates.")
    return {"best_correction_d": float(best),
            "rows": [(r[0], r[2], r[3], r[4]) for r in rows]}


# --------------------------------------------------------------------------- #
# verdict
# --------------------------------------------------------------------------- #

def verdict(span, sel, corr):
    """-> 'GO' | 'WEAK' | 'GENERATOR' | 'STOP'.

    ★ The structure matters more than the thresholds. A selection gain is only
    a reason to build if it survives a REALISTIC estimator, because the oracle
    column is unobtainable by construction. And a null selection gain is NOT the
    end of the question: if the correction block fires, the information is real
    and belongs in the generator -- which answers "should I add past frames?"
    with yes, while answering "will GoT then win?" with no. Collapsing those two
    into one pass/fail is what would make this test useless.
    """
    print(f"\n{'=' * 78}\nVERDICT -- thresholds fixed in the source before the run\n{'=' * 78}")
    perf_d, perf_ci = sel["best_perfect"][1]["d"], sel["best_perfect"][1]["ci"]
    real_d, real_ci = sel["best_realistic"][1]["d"], sel["best_realistic"][1]["ci"]
    corr_d = corr["best_correction_d"]

    c_perf = perf_d < -T_PRIZE and perf_ci[1] < 0
    c_real = real_d < -T_PRIZE and real_ci[1] < 0
    c_corr = corr_d < -T_GEN
    c_cov = span["coverage"] >= T_COVERAGE

    for name, v, ok in (
            (f"coverage               >= {T_COVERAGE:.0%}", span["coverage"], c_cov),
            (f"perfect-oracle select  < -{T_PRIZE:.2f} m, CI excludes 0", perf_d, c_perf),
            (f"realistic select (sv={SIGMA_REALISTIC})  < -{T_PRIZE:.2f} m, CI excludes 0",
             real_d, c_real),
            (f"correction             < -{T_GEN:.2f} m", corr_d, c_corr)):
        print(f"  [{'PASS' if ok else 'FAIL'}]  {name:<48} {v:+.4f}")

    print()
    if c_real:
        label = "GO"
        print("  GO. A motion signal an estimator could realistically produce")
        print("  selects better than greedy from the pool the model ALREADY")
        print("  emits. This is the first configuration in the project where")
        print("  deliberation has an axis the generator does not already use.")
        print("  Next: add 2 past frames, but keep the asymmetry -- measure a")
        print("  selector-only arm before a both-sides arm, because feeding the")
        print("  generator too is what the absorption law (sec.1.5, 103%) says")
        print("  gets eaten.")
    elif c_perf:
        label = "WEAK"
        print("  WEAK. The gain exists only with knowledge better than two frames")
        print("  can supply: it decays before the realistic column. Do not start")
        print("  the retrain on this. What would change the answer is a measured")
        print("  estimator error (run visual odometry on the actual frames and")
        print("  put its sigma into the sweep), not another selection rule.")
    elif c_corr:
        label = "GENERATOR"
        print("  GENERATOR, NOT SELECTOR. Perfect knowledge of the first 0.5 s")
        print("  does not help when it can only RANK the candidates, but it does")
        print("  help when it reaches the trajectory. The pool does not contain")
        print("  the fix, so no deliberation layer can choose it.")
        print()
        print("  Read this as two separate answers:")
        print("    - adding past frames is worth it FOR ABSOLUTE L2")
        print("    - it will NOT make GoT beat greedy, and sec.1 survives the")
        print("      input change (the absorption law predicts the improved pool")
        print("      is absorbed, and this block shows why: the fix is a level,")
        print("      not a choice)")
        print("  Budget the re-run chain (headline 3 seed + fusion + WM) against")
        print("  the first bullet alone.")
    else:
        label = "STOP"
        print("  STOP. Even perfect knowledge of the first 0.5 s -- strictly more")
        print("  than two past frames could give, handed to the selector while the")
        print("  generator is left alone, which is the most favourable arrangement")
        print("  that exists -- neither selects better nor corrects better than")
        print("  greedy. The past-frames track cannot flip the GoT result, and on")
        print("  this evidence it does not buy absolute L2 either.")
        print("  Do not spend the retrain + the sec.1 re-run chain.")

    print("\n  Either way this is a measurement: it is the first time the project")
    print("  has priced the one axis sec.1.1's six failure cases never covered.")
    print("  ORACLE UPPER BOUND -- do not report any row above as a method (sec.8).")
    return label


# --------------------------------------------------------------------------- #
# self-test
# --------------------------------------------------------------------------- #

def _mk(token, scene, wps, gt, base):
    wps = np.asarray(wps, float)
    vals = np.array([avg_l2(w, gt) for w in wps])
    return {"token": token, "seed": "42", "scene": scene, "command": "straight",
            "wps": wps, "logged_vals": vals, "vals": vals, "gt": gt,
            "base": base, "got": None, "fused": False}


def _world(kind, n_rec=240, n_cand=8, seed=0):
    """Three synthetic worlds. All three have the SAME dominant longitudinal
    error, so scale alone cannot pass any of them.

    speed   GT is a straight run at a speed the model got wrong, and the pool
            SPANS that speed. A speed signal must recover most of it -> GO.
    lottery candidate deviations are iid per waypoint, so matching the first
            step says nothing about the last: the axis is wrong and both
            selection and correction must be null -> STOP.
    ★ narrow GT error is a pure speed error (the axis is RIGHT) but every
            candidate has nearly the same speed. Selection cannot express the
            fix; correction can. This is the world that separates "the signal is
            wrong" from "the pool cannot say it", and it is the distinction the
            whole strategic question turns on -> GENERATOR.
    """
    rng = np.random.RandomState(seed)
    T = HZ + 1
    steps = np.arange(1, T + 1) * DEFAULT_DT
    pools = []
    for r in range(n_rec):
        v = 8.0 + rng.rand() * 4.0                       # the model's speed
        dv = rng.randn() * 2.0                           # how wrong it is
        mu = np.stack([steps * v, np.zeros(T)], 1)
        if kind == "lottery":
            gt = mu + rng.randn(T, 2) * 2.0
        else:
            gt = np.stack([steps * (v + dv), np.zeros(T)], 1) + rng.randn(T, 2) * 0.2
        sig_v = {"speed": 2.5, "narrow": 0.05, "lottery": 0.0}[kind]
        cands = []
        for _ in range(n_cand):
            vj = v + rng.randn() * sig_v
            c = np.stack([steps * vj, np.zeros(T)], 1)
            if kind == "lottery":
                c = c + rng.randn(T, 2) * 2.0
            else:
                c = c + rng.randn(T, 2) * 0.2
            cands.append(c)
        cands = np.asarray(cands)
        # candidate 0 is the score's pick by convention; the synthetic score is
        # uninformative, so leave the draw order alone
        pools.append(_mk(f"t{r}", f"sc{r // 4}", cands, gt, base=avg_l2(mu, gt)))
    return pools


def _selftest():
    print("[selftest] statistics wiring")
    ps = [{"scene": "a"}] * 20 + [{"scene": "b"}] * 20
    assert np.isfinite(_ci(ps + [{"scene": "c"}], [1.0] * 41)[0])
    print("  ok  scene-clustered CI runs on >= 3 scenes")

    print("\n[selftest] correction is a pure longitudinal rescale")
    gt = np.stack([np.arange(1, HZ + 2) * 5.0, np.zeros(HZ + 1)], 1)
    t = np.stack([np.arange(1, HZ + 2) * 4.0, np.ones(HZ + 1)], 1)
    c, ok = _corrected(t, gt)
    assert ok and np.allclose(c[:, 0], gt[:, 0]) and np.allclose(c[:, 1], t[:, 1])
    parked = np.zeros((HZ + 1, 2))
    c2, ok2 = _corrected(parked, gt)
    assert not ok2 and np.allclose(c2, parked)
    print("  ok  the lateral profile is untouched and parked records are skipped")

    out = {}
    for kind in ("speed", "lottery", "narrow"):
        print(f"\n{'#' * 78}\n[selftest] {kind.upper()} world\n{'#' * 78}")
        p = _world(kind, seed=1)
        sp = block_span(p)
        se = block_selection(p, n_boot=400)
        co = block_correction(p, n_boot=400)
        out[kind] = (sp, se, co, verdict(sp, se, co))

    sp_s, se_s, co_s, v_s = out["speed"]
    sp_l, se_l, co_l, v_l = out["lottery"]
    sp_n, se_n, co_n, v_n = out["narrow"]

    print(f"\n{'#' * 78}\n[selftest] assertions\n{'#' * 78}")

    # 1. a real, expressible speed error must be found -- and must survive noise
    assert v_s == "GO", f"speed world must read GO, got {v_s}"
    assert se_s["best_perfect"][1]["d"] < -T_PRIZE
    assert se_s["best_realistic"][1]["d"] < -T_PRIZE
    print(f"  ok  speed world -> GO (perfect {se_s['best_perfect'][1]['d']:+.3f}, "
          f"realistic {se_s['best_realistic'][1]['d']:+.3f})")

    # 2. the noise sweep must actually bite: an estimator that is worse must do
    #    worse. If this ever fails, the sweep is not measuring what it claims and
    #    the verdict's realistic column is decorative.
    g0 = se_s["table"][("speed magnitude", 0.0)]["d"]
    g2 = se_s["table"][("speed magnitude", 2.0)]["d"]
    assert g2 > g0 + 0.05, (g0, g2)
    print(f"  ok  the noise sweep degrades monotonically enough to matter "
          f"({g0:+.3f} at sigma=0 -> {g2:+.3f} at sigma=2)")

    # 3. wrong axis -> nothing anywhere. Note the lottery pool has a LARGE first
    #    step spread, so it passes coverage: coverage alone must not be able to
    #    carry a verdict.
    assert v_l == "STOP", f"lottery world must read STOP, got {v_l}"
    assert se_l["best_perfect"][1]["d"] > -T_PRIZE, se_l["best_perfect"][1]["d"]
    assert co_l["best_correction_d"] > -T_GEN, co_l["best_correction_d"]
    print(f"  ok  lottery world -> STOP (select {se_l['best_perfect'][1]['d']:+.3f}, "
          f"correct {co_l['best_correction_d']:+.3f}) with coverage "
          f"{sp_l['coverage']:.0%}")

    # 4. ★ the separation the strategic question turns on: right axis, pool
    #    cannot express it. Selection must fail and correction must fire.
    assert v_n == "GENERATOR", f"narrow world must read GENERATOR, got {v_n}"
    assert se_n["best_perfect"][1]["d"] > -T_PRIZE, se_n["best_perfect"][1]["d"]
    assert co_n["best_correction_d"] < -T_GEN, co_n["best_correction_d"]
    assert sp_n["coverage"] < T_COVERAGE, sp_n["coverage"]
    print(f"  ok  narrow world -> GENERATOR: selection {se_n['best_perfect'][1]['d']:+.3f} "
          f"is null while correction {co_n['best_correction_d']:+.3f} fires, "
          f"coverage {sp_n['coverage']:.0%}")

    # 5. the two null-selection worlds must NOT be collapsed into one verdict --
    #    that collapse is the failure mode this script exists to avoid
    assert v_l != v_n, "lottery and narrow must not read the same"
    print("  ok  the two null-selection worlds get DIFFERENT verdicts "
          "(STOP vs GENERATOR)")

    # 6. the uninformative control must sit on the random-candidate row in every
    #    world, or the rule machinery is leaking GT through the argmin itself
    for kind, (_, se, _, _) in out.items():
        ctl = se["table"][("random key (control)", 0.0)]["mean"]
        assert abs(ctl - se["random"]) < 0.05, (kind, ctl, se["random"])
    print("  ok  the uninformative control lands on the random-candidate row in "
          "all three worlds")

    print("\nself-test PASS")


# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(
        "ceiling for the past-frames track: would a perfect ego-motion signal "
        "let selection beat greedy? (0 GPU)")
    ap.add_argument("csv", nargs="*",
                    help="per_sample.csv with the got_cand_wps column "
                         "(results/fusion/final_top3 is the arm whose pool "
                         "equals ref)")
    ap.add_argument("--records_json", default=None,
                    help="the evaluation set the run used -- REQUIRED, every "
                         "candidate error is recomputed from its GT")
    ap.add_argument("--n_boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0, help="for the noise draws")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        _selftest()
        return
    if not a.csv:
        ap.error("give per_sample.csv files (or --selftest)")
    if not a.records_json:
        ap.error("--records_json is required: this script recomputes every "
                 "candidate's error from GT and refuses to run on the rounded "
                 "logged values alone")

    gt = load_gt(a.records_json)
    pools, skipped = load_pools(a.csv, gt)
    if not pools:
        sys.exit(f"[fatal] no usable pools. skipped: {skipped}\n"
                 f"  got_cand_wps was added on 2026-08-03 and cannot be rebuilt "
                 f"offline; results/headline/* predate it. Use "
                 f"results/fusion/final_top3/per_sample.csv.")
    if skipped:
        print(f"[load] skipped {skipped}")
    pools = block_wiring(pools)
    span = block_span(pools)
    sel = block_selection(pools, a.n_boot, a.seed)
    corr = block_correction(pools, a.n_boot)
    verdict(span, sel, corr)


if __name__ == "__main__":
    main()
