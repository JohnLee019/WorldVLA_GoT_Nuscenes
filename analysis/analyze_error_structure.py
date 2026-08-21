#!/usr/bin/env python
"""Is the planning error one INITIAL VELOCITY MISTAKE, integrated over time?
No GPU, no model -- per_sample.csv plus the records' GT.

The aggregate numbers already point this way. Per-second error over the six
waypoints is 1.978 / 1.988 / 1.989 / 2.011 / 2.042 / 2.081 for greedy: the
horizon grows 6x and the per-second error grows 5%. The model-free
mean-trajectory prior is flatter still (3.10 / 3.11 / 3.11). Flat means the
error is a constant velocity offset being integrated, not a decision that goes
progressively wrong.

★ BUT A FLAT MEAN IS NOT THE CLAIM. Averages can be linear while no individual
record is: a mix of records that decelerate and records that accelerate averages
to something smooth. The claim worth making is per-record, and it is sharp --
if the error is one velocity mistake made at t=0, then for EVERY record

    e_k  ~=  (k+1) * e_0        (same direction, magnitude growing linearly)

so cos(e_5, e_0) ~= 1 and |e_5| / (6 |e_0|) ~= 1. That is what this measures,
against a null that keeps every magnitude and destroys only the pairing.

Why it matters: if it holds, the residual is a scalar the model cannot observe
(a static frame carries no ego speed), which explains in one mechanism why
selection (sec.1.1-1.6), aggregation (sec.1.5), distillation (sec.1.8) and
post-hoc smoothing all came back null -- none of them adds the missing scalar.
If it does NOT hold, that story is wrong and the flat mean was an averaging
artefact.

Usage
-----
    python analyze_error_structure.py results/headline/ref/per_sample.csv \
        --records_json ./data/nuscenes_records/nuscenes_val_scenespread.json
    python analyze_error_structure.py --selftest
"""

from __future__ import annotations

import argparse
import ast
import csv
import os
import sys
from collections import defaultdict

import numpy as np

from analyze_got_csv import cluster_bootstrap_ci, spearman

PRED_KEYS = ("got_pred", "pred")        # eval_got writes the first, eval_nuscenes the second
DT = 0.5


def _traj(row):
    for k in PRED_KEYS:
        v = row.get(k, "")
        if not v:
            continue
        try:
            a = np.asarray(ast.literal_eval(v), dtype=np.float64)
        except (ValueError, SyntaxError):
            continue
        if a.ndim == 2 and a.shape[1] == 2:
            return a
    return None


def load(paths, records_json):
    import json
    gt_by_tok = {}
    for r in json.load(open(records_json)):
        wp = np.asarray(r.get("waypoints") or [], dtype=np.float64)
        if wp.ndim == 2 and len(wp) >= 6:
            gt_by_tok[r["sample_token"]] = wp
    out, skipped = [], defaultdict(int)
    for p in paths:
        if not os.path.exists(p):
            sys.exit(f"[fatal] {p} does not exist")
        with open(p, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("got_status") not in (None, "ok") or \
                        row.get("status") not in (None, "ok"):
                    skipped["arm_failed"] += 1
                    continue
                gt = gt_by_tok.get(row.get("sample_token"))
                pred = _traj(row)
                if gt is None or pred is None or len(pred) < 6:
                    skipped["no_gt_or_pred"] += 1
                    continue
                out.append({"scene": row.get("scene", "?"),
                            "err": pred[:6] - gt[:6]})       # (6, 2)
    return out, dict(skipped)


def run(recs, n_boot=5000, seed=0):
    E = np.stack([r["err"] for r in recs])                   # (n, 6, 2)
    n = len(E)
    mag = np.linalg.norm(E, axis=2)                           # (n, 6)
    t = np.arange(1, 7) * DT

    print(f"\n{'=' * 78}\nERROR STRUCTURE: one initial velocity mistake?\n{'=' * 78}")
    print(f"\n  {n} records / {len({r['scene'] for r in recs})} scenes\n")
    print(f"  {'step':<8}" + "".join(f"{x:>9.1f}s" for x in t))
    print(f"  {'|e_k|':<8}" + "".join(f"{x:>10.3f}" for x in mag.mean(0)))
    print(f"  {'per sec':<8}" + "".join(f"{x:>10.3f}" for x in mag.mean(0) / t))

    # ---- the per-record test ------------------------------------------------
    e0, e5 = E[:, 0, :], E[:, 5, :]
    n0 = np.linalg.norm(e0, axis=1)
    n5 = np.linalg.norm(e5, axis=1)
    ok = (n0 > 1e-6) & (n5 > 1e-6)
    cos = np.sum(e0[ok] * e5[ok], axis=1) / (n0[ok] * n5[ok])
    ratio = n5[ok] / (6.0 * n0[ok])

    # null: pair each record's e_5 with ANOTHER record's e_0. Magnitudes and
    # their distributions are untouched; only the within-record link dies.
    rng = np.random.RandomState(seed)
    perm = rng.permutation(np.sum(ok))
    e0p = e0[ok][perm]
    cos_null = np.sum(e0p * e5[ok], axis=1) / (np.linalg.norm(e0p, axis=1) * n5[ok])

    print(f"\n  -- per-record: does e_5 point where e_0 points, 6x as long?")
    print(f"     cos(e_5, e_0)          {cos.mean():+.4f}   "
          f"(shuffled null {cos_null.mean():+.4f})")
    print(f"     |e_5| / (6*|e_0|)      {np.median(ratio):.4f}  median   "
          f"({(np.abs(ratio - 1) < 0.5).mean():.1%} of records within +-50%)")

    # variance of the 3 s error explained by extrapolating the 0.5 s error
    resid = e5 - 6.0 * e0
    r2 = 1.0 - float(np.mean(np.sum(resid ** 2, 1)) / np.mean(np.sum(e5 ** 2, 1)))
    print(f"     R^2 of e_5 ~= 6*e_0    {r2:+.4f}   "
          f"(1.0 = the 3 s error IS the 0.5 s error scaled)")
    print(f"     rho(|e_0|, |e_5|)      {spearman(list(n0), list(n5)):+.4f}")

    # ---- which axis ---------------------------------------------------------
    lon, lat = np.abs(E[:, 5, 0]).mean(), np.abs(E[:, 5, 1]).mean()
    print(f"\n  -- axis of the 3 s error:  longitudinal {lon:.3f} m   "
          f"lateral {lat:.3f} m   ratio {lon / max(lat, 1e-9):.1f}x")

    # ---- how much is a pure constant-velocity model? ------------------------
    # best single velocity per record, fitted by least squares on all 6 steps
    tt = t[:, None]
    v_hat = np.sum(E * tt, axis=1) / np.sum(t ** 2)           # (n, 2)
    pred = v_hat[:, None, :] * tt[None, :, :]
    unexplained = float(np.mean(np.sum((E - pred) ** 2, (1, 2)))
                        / np.mean(np.sum(E ** 2, (1, 2))))
    print(f"\n  -- fit a single constant velocity error per record (6 steps, LS):")
    print(f"     it explains {1 - unexplained:.1%} of the total squared error")
    print("     !! DO NOT READ THIS AS THE ANSWER. A least-squares line through six")
    print("        growing points fits almost anything: the self-test's COMPOUNDING")
    print("        world, where a fresh random acceleration hits every step and the")
    print("        story is false, still scores 95.5% here. The discriminating")
    print("        numbers are cos(e_5,e_0) and R^2 above (0.98/0.995 in the")
    print("        velocity world against 0.51/0.39 in the compounding one).")
    scenes = defaultdict(list)
    for r, v in zip(recs, 1.0 - np.sum((E - pred) ** 2, (1, 2)) / np.maximum(
            np.sum(E ** 2, (1, 2)), 1e-12)):
        scenes[r["scene"]].append(float(v))
    lo, hi = cluster_bootstrap_ci(scenes, n_boot=n_boot)
    print(f"     per-record share, scene-cluster CI [{lo:.4f}, {hi:.4f}]")
    print(f"     mean |v_err| {np.linalg.norm(v_hat, axis=1).mean():.3f} m/s "
          f"({np.linalg.norm(v_hat, axis=1).mean() * 3.6:.1f} km/h)")

    print(f"\n{'-' * 78}\nVERDICT\n{'-' * 78}")
    # thresholds set on the two self-test worlds, which the weak
    # const-velocity share cannot separate (0.9997 vs 0.9553)
    strong = r2 > 0.75 and cos.mean() > 0.85
    if strong:
        print("  The error IS one velocity mistake made at t=0 and integrated.")
        print("  A static front-camera frame carries no ego speed, so that scalar")
        print("  is not observable at inference. Every intervention that reorders,")
        print("  aggregates or smooths the model's OWN outputs leaves it untouched,")
        print("  which is one mechanism for all of sec.1.1-1.8 being null.")
        print("  It also predicts what WOULD work and why it is refused (sec.8):")
        print("  ego status supplies exactly this scalar -- and that is why")
        print("  BEV-Planner reaches SOTA on this metric without an image.")
    else:
        print("  REJECTED as stated: the error is NOT one velocity mistake.")
        print(f"  cos {cos.mean():+.4f} against 0.98 in the pure-velocity world, so")
        print("  the direction of the 3 s error is only partly set at 0.5 s. Do not")
        print("  write 'the error is an initial speed error' -- it is not.")
        # A middle band exists and it is NOT a pass. Naming it here so the
        # failure is not read as "nothing is there", which would be just as
        # wrong: the shuffled null is at ~0.00 and rho(|e_0|,|e_5|) is high.
        if r2 > 0.5 or cos.mean() > 0.5:
            print()
            print(f"  PARTIAL (descriptive, added after seeing the data -- not a")
            print(f"  preregistered pass): extrapolating e_0 by 6x still explains")
            print(f"  {r2:.0%} of the 3 s error against a null at "
                  f"{cos_null.mean():+.3f}, so a large share of the final error is")
            print("  already committed in the first half second while the rest")
            print("  accrues later. What the later part is -- the driver's actual")
            print("  future acceleration, also unobservable from one frame, or the")
            print("  model compounding -- this script cannot separate.")

    # An INDEPENDENT observation that owes nothing to the hypothesis above and
    # needs no threshold: whatever the error's time structure, it lives almost
    # entirely on the longitudinal axis. The selector scores kinematic
    # smoothness and a lateral command; neither can see whether the SPEED LEVEL
    # is right. That stands whether or not the velocity story survives.
    print(f"\n  INDEPENDENT OF THE ABOVE: the 3 s error is {lon / max(lat, 1e-9):.1f}x "
          f"longitudinal ({lon:.2f} m vs {lat:.2f} m).")
    print("  The score's two terms are smoothness and lateral command, so neither")
    print("  term measures the axis that carries the error. No threshold or null")
    print("  is involved in this one.")
    return {"r2_e0": r2, "cos": float(cos.mean()), "cos_null": float(cos_null.mean()),
            "const_v_share": 1 - unexplained, "strong": bool(strong),
            "lon": float(lon), "lat": float(lat)}


def _selftest():
    def world(kind, n=400, seed=0):
        rng = np.random.RandomState(seed)
        t = np.arange(1, 7) * DT
        recs = []
        for i in range(n):
            if kind == "velocity":
                # one wrong initial speed, integrated -- plus small jitter
                v = rng.randn(2) * np.array([2.0, 0.3])
                e = v[None, :] * t[:, None] + rng.randn(6, 2) * 0.05
            else:
                # compounding: a fresh random acceleration every step
                e = np.cumsum(np.cumsum(rng.randn(6, 2) * 0.6, 0), 0)
            recs.append({"scene": f"sc{i // 4}", "err": e})
        return recs

    print("[selftest] VELOCITY world")
    v = run(world("velocity", seed=1), n_boot=300)
    assert v["const_v_share"] > 0.9, v["const_v_share"]
    assert v["cos"] > 0.9 and v["cos_null"] < 0.3, (v["cos"], v["cos_null"])
    assert v["strong"]
    print(f"\n  ok  velocity world: const-v share {v['const_v_share']:.3f}, "
          f"cos {v['cos']:+.3f} vs null {v['cos_null']:+.3f}")

    print("\n[selftest] COMPOUNDING world")
    c = run(world("compound", seed=1), n_boot=300)
    assert not c["strong"], c
    assert c["r2_e0"] < v["r2_e0"] - 0.3, (c["r2_e0"], v["r2_e0"])
    assert c["cos"] < 0.7, c["cos"]
    print(f"\n  ok  compounding world is rejected "
          f"(R^2 {c['r2_e0']:+.3f}, cos {c['cos']:+.3f})")

    # ★ and the reason the verdict does NOT use the constant-velocity share: it
    # fails to separate the worlds at all. Asserted so that the next person
    # cannot quietly promote that number back into the decision.
    assert c["const_v_share"] > 0.9, c["const_v_share"]
    print(f"  ok  const-velocity share does NOT discriminate "
          f"({v['const_v_share']:.3f} vs {c['const_v_share']:.3f}) -- kept out "
          f"of the verdict on purpose")

    # the shuffled null must kill the cosine in BOTH worlds: it is testing the
    # within-record link, not the shape of the error
    assert abs(c["cos_null"]) < 0.3 and abs(v["cos_null"]) < 0.3
    print("  ok  the shuffled null removes the within-record link in both worlds")
    print("\nself-test PASS")


def main():
    p = argparse.ArgumentParser("is the error one initial velocity mistake? (0 GPU)")
    p.add_argument("csv", nargs="*")
    p.add_argument("--records_json")
    p.add_argument("--n_boot", type=int, default=5000)
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    if a.selftest:
        _selftest()
        return
    if not a.csv or not a.records_json:
        p.error("give per_sample.csv files and --records_json (or --selftest)")
    recs, skipped = load(a.csv, a.records_json)
    if not recs:
        sys.exit(f"[fatal] nothing usable. Needs a prediction column "
                 f"({' or '.join(PRED_KEYS)}) and matching GT. skipped: {skipped}")
    if skipped:
        print(f"[load] skipped {skipped}")
    run(recs, a.n_boot)


if __name__ == "__main__":
    main()
