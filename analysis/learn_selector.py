#!/usr/bin/env python
"""Can a LEARNED combination of the logged signals pick better than the score?

No GPU, no model, no re-evaluation. per_sample.csv already stores, per record:
the true avgL2@3s of every candidate (`got_cand_vals`) and every score
component in the same order (`got_cand_kin` / `_cmd` / `_total` / `_lik`).
That makes counterfactual selection EXACT offline -- for any ranking rule we
can look up the L2 of whatever candidate it would have committed to. No
approximation, no GPU, no re-planning.

Why this experiment
-------------------
Everything tried so far was a LINEAR, HAND-SET combination: --w_kinematic /
--w_command sweeps, the z-norm fix (`ranknorm`), kinematic-alone selection
(`cmd_prune_only`), adding self-likelihood. All null. And the headline finding
that "the sum (rho +0.5079) is below the best single component (+0.5225)" is
about an EQUAL-WEIGHT LINEAR sum.

A fitted nonlinear combination has never been tried. It is the last cheap way
to ask whether the failure is in the combination RULE or in the FEATURES:

  learned rule beats the best single component  -> the score design was the
                                                  problem after all
  it does not                                  -> the features are exhausted,
                                                  which is a much stronger
                                                  version of the paper's claim

Either outcome is a result. This is the last question that costs nothing.

Two tasks
---------
  T1  candidate ranking   features -> which candidate is best. Scored by the
                          L2 you actually get by taking its argmax, next to
                          the score's 3.6072, greedy's 3.5557 and the pool
                          ceiling minADE_C.
  T2  fallback detector   features -> will GoT beat greedy on this record?
                          Section E of analyze_selection_gap found all three
                          hand-made proxies at rho ~= 0; this asks whether a
                          fitted one does better. The payoff number is the
                          held-out L2 of "use GoT unless the detector says
                          defer", against greedy.

Honest evaluation
-----------------
★ Cross-validation splits by SCENE, never by seed and never by record. The
three seeds are the SAME 600 records replanned, so holding out a seed leaks
the record's difficulty into training. nuScenes records are consecutive
keyframes, so holding out records leaks the scene (PROJECT_HANDOFF sec 9 --
the trap that caused three retractions). Scene-level k-fold is the only split
that is not leaking here.

Baselines are reported next to every learned number, because a learned model
that lands between random and the existing score has learned nothing useful.

Usage
-----
    python learn_selector.py results/headline/*/per_sample.csv
    python learn_selector.py --selftest
"""

import argparse
import csv
import sys
from collections import defaultdict

import numpy as np

from analyze_got_csv import _f, _list, spearman

KEY = "avgL2@3s"


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #

def load_pools(paths, key=KEY):
    """-> list of per-record pool dicts, pooled over every csv given."""
    pools, skipped, mismatch = [], defaultdict(int), 0
    for path in paths:
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("got_status") != "ok" or r.get("base_status") != "ok":
                    skipped["arm_failed"] += 1
                    continue
                vals = _list(r, "got_cand_vals")
                kin = _list(r, "got_cand_kin")
                cmd = _list(r, "got_cand_cmd")
                tot = _list(r, "got_cand_total")
                if not vals or not kin or not cmd or not tot:
                    skipped["no_pool"] += 1
                    continue
                n = len(vals)
                if not (len(kin) == len(cmd) == len(tot) == n) or n < 2:
                    skipped["ragged_pool"] += 1
                    continue
                got, base = _f(r, f"got_{key}"), _f(r, f"base_{key}")
                if got is None or base is None:
                    skipped["missing_l2"] += 1
                    continue
                lik = _list(r, "got_cand_lik")
                fin = _list(r, "got_cand_final")
                sel_score = fin if (fin and len(fin) == n) else tot
                sel = int(np.argmax(sel_score))
                # the pipeline's committed trajectory must BE one of the pool
                # entries; if it is not, the counterfactual below is invalid.
                if abs(vals[sel] - got) > 5e-3:
                    mismatch += 1
                pools.append({
                    "scene": r.get("scene", "?"), "seed": r.get("seed", ""),
                    "token": r.get("sample_token", ""),
                    "vals": np.asarray(vals, float),
                    "kin": np.asarray(kin, float),
                    "cmd": np.asarray(cmd, float),
                    "tot": np.asarray(sel_score, float),
                    "lik": np.asarray(lik, float) if (lik and len(lik) == n) else None,
                    "sel": sel, "got": got, "base": base,
                })
    return pools, dict(skipped), mismatch


# --------------------------------------------------------------------------- #
# features
# --------------------------------------------------------------------------- #

def _z(a):
    s = a.std()
    return (a - a.mean()) / s if s > 1e-12 else np.zeros_like(a)


def _rank01(a):
    """Within-pool rank in [0,1], ties averaged. Scale-free, so records with
    wildly different score magnitudes become comparable -- the thing the
    `ranknorm` arm fixed for the linear rule and which is free here."""
    order = np.argsort(np.argsort(a))
    return order / max(len(a) - 1, 1)


def pool_features(p):
    """(n_candidates, n_features) for one record. Inference-time only: nothing
    here touches the ground truth."""
    kin, cmd, tot = p["kin"], p["cmd"], p["tot"]
    n = len(kin)
    cols = [
        _z(kin), _z(cmd), _z(tot),
        _rank01(kin), _rank01(cmd), _rank01(tot),
        _z(kin) * _z(cmd), _z(kin) ** 2, _z(cmd) ** 2,
        np.full(n, float(n)),
        np.full(n, float(kin.std())), np.full(n, float(cmd.std())),
    ]
    if p["lik"] is not None:
        cols += [_z(p["lik"]), _rank01(p["lik"])]
    else:
        cols += [np.zeros(n), np.zeros(n)]
    return np.column_stack(cols)


FEATURE_NAMES = ["kin_z", "cmd_z", "tot_z", "kin_r", "cmd_r", "tot_r",
                 "kin*cmd", "kin^2", "cmd^2", "n_cand", "kin_std", "cmd_std",
                 "lik_z", "lik_r"]


def record_features(p):
    """One row per record, for the T2 fallback detector."""
    kin, cmd, tot, vals = p["kin"], p["cmd"], p["tot"], p["vals"]
    srt = np.sort(tot)[::-1]
    margin = srt[0] - srt[1]
    return np.array([
        float(len(vals)), float(tot.std()), float(kin.std()), float(cmd.std()),
        margin, float(margin / (tot.std() + 1e-9)),
        float(np.corrcoef(kin, cmd)[0, 1]) if kin.std() > 1e-12 and cmd.std() > 1e-12 else 0.0,
        float(_rank01(tot)[p["sel"]]),
        float(_z(kin)[p["sel"]]), float(_z(cmd)[p["sel"]]),
    ], dtype=float)


RECORD_FEATURE_NAMES = ["n_cand", "tot_std", "kin_std", "cmd_std", "margin",
                        "margin/std", "corr(kin,cmd)", "sel_tot_rank",
                        "sel_kin_z", "sel_cmd_z"]


# --------------------------------------------------------------------------- #
# model: ridge (numpy only), optional gradient boosting when sklearn is there
# --------------------------------------------------------------------------- #

def fit_ridge(X, y, lam=1.0):
    Xb = np.column_stack([X, np.ones(len(X))])
    mu, sd = Xb.mean(0), Xb.std(0)
    sd[sd < 1e-12] = 1.0
    Xs = (Xb - mu) / sd
    A = Xs.T @ Xs + lam * np.eye(Xs.shape[1])
    w = np.linalg.solve(A, Xs.T @ y)
    return lambda Z: (((np.column_stack([Z, np.ones(len(Z))]) - mu) / sd) @ w), w


def fit_gbr(X, y):
    """Nonlinear learner. Returns None when sklearn is unavailable -- the ridge
    with interaction terms still answers the question, just more weakly."""
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
    except ImportError:
        return None
    m = HistGradientBoostingRegressor(max_iter=200, max_depth=4,
                                      learning_rate=0.08, random_state=0)
    m.fit(X, y)
    return lambda Z: m.predict(Z)


# --------------------------------------------------------------------------- #
# scene-level folds
# --------------------------------------------------------------------------- #

def scene_folds(pools, k=5, seed=0):
    scenes = sorted({p["scene"] for p in pools})
    rng = np.random.RandomState(seed)
    rng.shuffle(scenes)
    assign = {s: i % k for i, s in enumerate(scenes)}
    return [[p for p in pools if assign[p["scene"]] != f] for f in range(k)], \
           [[p for p in pools if assign[p["scene"]] == f] for f in range(k)], \
           len(scenes)


# --------------------------------------------------------------------------- #
# T1: candidate ranking
# --------------------------------------------------------------------------- #

def _rho_pool(score_higher_better, vals):
    """Within-pool Spearman in the project's convention: score vs -true error.

    Returns nan for pools that cannot carry rank information (fewer than 3
    candidates after dedup, or a constant input). Those must be DROPPED, not
    averaged in -- a single nan poisons np.mean and silently reports nan for
    the whole run.
    """
    if len(vals) < 3:
        return float("nan")
    return spearman(list(np.asarray(score_higher_better, float)),
                    list(-np.asarray(vals, float)))


def _mean_rho(rhos):
    v = [r for r in rhos if r == r]           # drop nan
    return (float(np.mean(v)) if v else float("nan")), len(v), len(rhos) - len(v)


def baseline_rhos(pools):
    """Same-convention within-pool rho for the hand-made components, computed on
    the SAME records the learner is scored on. The +0.5225 / +0.5079 in the
    handoff come from a different run (600 records, seed 42) -- recomputing here
    makes the comparison apples-to-apples instead of cross-run."""
    out = {}
    for name, get in (("kinematic", lambda p: p["kin"]),
                      ("command", lambda p: p["cmd"]),
                      ("path_score", lambda p: p["tot"]),
                      ("likelihood", lambda p: p["lik"])):
        rhos = []
        for p in pools:
            s = get(p)
            if s is None:
                continue
            rhos.append(_rho_pool(s, p["vals"]))
        if rhos:
            out[name] = _mean_rho(rhos)
    return out


def t1(pools, k=5):
    print("\n" + "=" * 78)
    print("T1. CAN A LEARNED RULE PICK A BETTER CANDIDATE?")
    print("=" * 78)

    ceil = float(np.mean([p["vals"].min() for p in pools]))
    rand = float(np.mean([p["vals"].mean() for p in pools]))
    cur = float(np.mean([p["vals"][p["sel"]] for p in pools]))
    kin_only = float(np.mean([p["vals"][int(np.argmax(p["kin"]))] for p in pools]))
    greedy = float(np.mean([p["base"] for p in pools]))
    print(f"\n  reference points ({len(pools)} record-plans)")
    print(f"    random candidate      {rand:.4f}")
    print(f"    kinematic alone       {kin_only:.4f}")
    print(f"    current score (GoT)   {cur:.4f}")
    print(f"    greedy free-run       {greedy:.4f}")
    print(f"    oracle  (minADE_C)    {ceil:.4f}")

    sizes = np.array([len(p["vals"]) for p in pools])
    print(f"    pool size: mean {sizes.mean():.2f}  min {sizes.min()}  "
          f"max {sizes.max()}   (<3 carries no rank info)")

    print("\n  hand-made components, same records, same convention "
          "(score vs -true error)")
    for name, (rho, nu, nd) in baseline_rhos(pools).items():
        print(f"    {name:<12} rho {rho:+.4f}   (n={nu}"
              + (f", {nd} pools dropped as nan)" if nd else ")"))

    tr_folds, te_folds, n_scenes = scene_folds(pools, k)
    print(f"\n  {k}-fold CV split by SCENE ({n_scenes} scenes) -- not by seed, "
          f"not by record")

    for name, fitter in (("ridge+interactions", fit_ridge),
                         ("gradient boosting", fit_gbr)):
        picks, rhos, top1 = [], [], []
        skipped = False
        for tr, te in zip(tr_folds, te_folds):
            Xtr = np.vstack([pool_features(p) for p in tr])
            # target: within-pool z of the true error, so easy and hard records
            # contribute equally instead of the model just learning difficulty
            ytr = np.concatenate([_z(p["vals"]) for p in tr])
            fitted = fitter(Xtr, ytr)
            # fit_ridge returns (predict_fn, weights); fit_gbr returns a
            # predict_fn or None when sklearn is missing.
            out = fitted[0] if isinstance(fitted, tuple) else fitted
            if out is None:
                skipped = True
                break
            for p in te:
                pred = out(pool_features(p))
                j = int(np.argmin(pred))       # lower predicted error = better
                picks.append(p["vals"][j])
                top1.append(1.0 if j == int(np.argmin(p["vals"])) else 0.0)
                # -pred is "higher = better", matching the component convention
                rhos.append(_rho_pool(-pred, p["vals"]))
        if skipped:
            print(f"\n  -- {name}: sklearn not installed; skipped "
                  f"(ridge result above still answers the question)")
            continue
        m = float(np.mean(picks))
        rho, n_used, n_nan = _mean_rho(rhos)
        print(f"\n  -- {name} (held-out)")
        print(f"     mean {KEY} if it selected   {m:.4f}   "
              f"({m - cur:+.4f} vs score, {m - greedy:+.4f} vs greedy)")
        print(f"     within-pool Spearman         {rho:+.4f}   "
              f"(n={n_used}" + (f", {n_nan} dropped as nan)" if n_nan else ")")
              + "  <- compare with the component table above")
        print(f"     top1                         {np.mean(top1):.4f}   "
              f"(current 0.2489, random {np.mean([1/len(p['vals']) for p in pools]):.4f})")
        frac = (cur - m) / (cur - ceil) if cur > ceil else 0.0
        print(f"     headroom recovered           {frac:6.1%}")


# --------------------------------------------------------------------------- #
# T2: fallback detector
# --------------------------------------------------------------------------- #

def t2(pools, k=5, fracs=(0.1, 0.2, 0.3, 0.5)):
    print("\n" + "=" * 78)
    print("T2. CAN A LEARNED DETECTOR SAY WHEN TO DEFER TO GREEDY?")
    print("=" * 78)
    got = float(np.mean([p["got"] for p in pools]))
    greedy = float(np.mean([p["base"] for p in pools]))
    ceil = float(np.mean([min(p["got"], p["base"]) for p in pools]))
    print(f"\n    GoT {got:.4f}   greedy {greedy:.4f}   "
          f"oracle fallback {ceil:.4f}")
    print("    hand-made proxies all landed at rho ~= 0 "
          "(analyze_selection_gap, section E)")

    tr_folds, te_folds, _ = scene_folds(pools, k)
    conf, order = {}, []
    for tr, te in zip(tr_folds, te_folds):
        Xtr = np.vstack([record_features(p) for p in tr])
        ytr = np.array([p["got"] - p["base"] for p in tr])
        pred, _w = fit_ridge(Xtr, ytr)
        for p in te:
            conf[id(p)] = float(pred(record_features(p)[None, :])[0])
            order.append(p)

    dev = [p["got"] - p["base"] for p in order]
    pr = [conf[id(p)] for p in order]
    print(f"\n  held-out Spearman(predicted deviation, true deviation) "
          f"= {spearman(pr, dev):+.4f}")
    print("  (this is the number that decides it; the sweep below is "
          "meaningless if it is ~0)")

    print(f"\n  {'defer':>7} {'held-out mean':>15} {'vs greedy':>11}")
    for fr in fracs:
        ranked = sorted(order, key=lambda p: conf[id(p)], reverse=True)
        deferred = {id(p) for p in ranked[:int(round(fr * len(ranked)))]}
        m = float(np.mean([(p["base"] if id(p) in deferred else p["got"])
                           for p in order]))
        flag = "  <- beats greedy" if m < greedy else ""
        print(f"  {fr:>6.0%} {m:>15.4f} {m - greedy:>+11.4f}{flag}")
    print("\n  A working detector shows an interior optimum. Monotone "
          "improvement with the deferral\n  fraction just means it is mixing "
          "toward greedy and carries no information.")


# --------------------------------------------------------------------------- #

def selftest():
    rng = np.random.RandomState(0)
    # a pool whose true error IS a nonlinear function of the features: the
    # learner must beat the linear score, otherwise the harness is broken
    pools = []
    for i in range(400):
        n = 8
        kin, cmd = rng.randn(n), rng.randn(n)
        vals = 3.0 + (kin * cmd) + 0.05 * rng.randn(n)      # pure interaction
        tot = kin + cmd                                      # linear score fails
        pools.append({"scene": f"s{i // 4}", "seed": "42", "token": f"t{i}",
                      "vals": vals, "kin": kin, "cmd": cmd, "tot": tot,
                      "lik": None, "sel": int(np.argmax(tot)),
                      "got": float(vals[int(np.argmax(tot))]),
                      "base": 3.0})
    tr, te, ns = scene_folds(pools, 5)
    assert ns == 100, ns
    tr_scenes = {p["scene"] for p in tr[0]}
    te_scenes = {p["scene"] for p in te[0]}
    assert not (tr_scenes & te_scenes), "scene folds must not overlap"
    print("  ok  folds split by scene with no overlap")

    X = np.vstack([pool_features(p) for p in tr[0]])
    y = np.concatenate([_z(p["vals"]) for p in tr[0]])
    pred, _ = fit_ridge(X, y)
    picks, cur = [], []
    for p in te[0]:
        j = int(np.argmin(pred(pool_features(p))))
        picks.append(p["vals"][j])
        cur.append(p["vals"][p["sel"]])
    assert np.mean(picks) < np.mean(cur), \
        f"learner must beat the linear score on interaction data: {np.mean(picks)} vs {np.mean(cur)}"
    print(f"  ok  ridge+interactions recovers a pure interaction "
          f"({np.mean(picks):.3f} < {np.mean(cur):.3f})")

    # and on pure noise it must NOT beat the score
    npools = []
    for i in range(400):
        n, vals = 8, 3.0 + rng.randn(8)
        kin, cmd = rng.randn(n), rng.randn(n)
        npools.append({"scene": f"s{i // 4}", "seed": "42", "token": f"t{i}",
                       "vals": vals, "kin": kin, "cmd": cmd, "tot": kin + cmd,
                       "lik": None, "sel": int(np.argmax(kin + cmd)),
                       "got": float(vals[int(np.argmax(kin + cmd))]), "base": 3.0})
    tr2, te2, _ = scene_folds(npools, 5)
    X = np.vstack([pool_features(p) for p in tr2[0]])
    y = np.concatenate([_z(p["vals"]) for p in tr2[0]])
    pred2, _ = fit_ridge(X, y)
    picks2 = [p["vals"][int(np.argmin(pred2(pool_features(p))))] for p in te2[0]]
    rand2 = float(np.mean([p["vals"].mean() for p in te2[0]]))
    assert abs(np.mean(picks2) - rand2) < 0.35, \
        f"on noise the learner must land near random, got {np.mean(picks2)} vs {rand2}"
    print("  ok  on pure noise it lands near random (no false positive)")

    assert abs(_rank01(np.array([5.0, 1.0, 3.0]))[1] - 0.0) < 1e-9
    assert abs(_z(np.array([1.0, 1.0]))).max() < 1e-9
    print("  ok  _rank01 / _z degenerate cases")

    # the nan trap that produced "+nan" on the first real run: a pool too small
    # to carry rank information must be DROPPED from the mean, not averaged in.
    assert _rho_pool(np.array([1.0, 2.0]), np.array([1.0, 2.0])) !=         _rho_pool(np.array([1.0, 2.0]), np.array([1.0, 2.0]))      # nan != nan
    m, used, dropped = _mean_rho([0.5, float("nan"), 0.7])
    assert abs(m - 0.6) < 1e-9 and used == 2 and dropped == 1, (m, used, dropped)
    # convention check: a score that is HIGH where the true error is LOW is a
    # good score, so rho must be +1 (not -1). Getting this backwards would flip
    # the sign of every number compared against the +0.5225 component table.
    assert _rho_pool(np.array([3.0, 2.0, 1.0]), np.array([1.0, 2.0, 3.0])) > 0.99
    assert _rho_pool(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0])) < -0.99
    print("  ok  nan pools are dropped, not propagated (the +nan bug)")
    print("\nself-test PASS")


def main():
    p = argparse.ArgumentParser(
        "learned selector / fallback detector from logged features (0 GPU)")
    p.add_argument("csv", nargs="*")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    if a.selftest:
        selftest()
        return
    if not a.csv:
        p.error("give per_sample.csv files (or --selftest)")

    pools, skipped, mismatch = load_pools(a.csv)
    if not pools:
        sys.exit("[fatal] no usable candidate pools -- does the csv have "
                 "got_cand_vals / got_cand_kin / got_cand_cmd / got_cand_total?")
    scenes = len({p["scene"] for p in pools})
    seeds = sorted({p["seed"] for p in pools})
    print(f"[load] {len(pools)} record-plans / {scenes} scenes / seeds {seeds}"
          + (f"   skipped {skipped}" if skipped else ""))
    if mismatch:
        print(f"[warn] {mismatch} records where cand_vals[argmax(score)] != "
              f"got_{KEY}. The counterfactual assumes the committed plan IS a "
              f"pool entry; a large count invalidates T1.")
    else:
        print("[check] cand_vals[argmax(score)] == got_avgL2@3s on every record "
              "-> offline counterfactual selection is exact.")

    t1(pools, a.folds)
    t2(pools, a.folds)


if __name__ == "__main__":
    main()
