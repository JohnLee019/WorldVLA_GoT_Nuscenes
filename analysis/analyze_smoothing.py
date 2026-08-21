"""E3: does output smoothing help, and does it help GoT MORE than greedy?

WHY THIS IS NOT JUST "TRY A POST-PROCESSOR"

sec.1.2 measured that the GoT-minus-greedy distribution is almost symmetric
(42.7% worse / 38.7% better / 18.7% identical) and that 46.2% of the positive
loss sits in the worst 5% of records. The reading was:

    "the deliberation layer does not create bias, it creates variance"

Smoothing is a pure variance-reduction operator -- it cannot add information
about where the car actually went. So if that reading is right, smoothing
should recover a real share of the +0.0397 gap on the GoT arm while doing
much less for greedy. That is an INDEPENDENT test of the mechanism, which is
worth more than the post-processor itself (which is expected to be null).

THE CONTROL IS THE EXPERIMENT. "Smoothing helps GoT" and "smoothing helps any
trajectory" are different claims and the GoT arm alone cannot separate them.
That needs `base_pred` in the csv (added to eval_got_nuscenes.py, session 15);
runs made before that only support --allow_no_control, which prints the
weaker claim and says so.

WHY POLYNOMIALS AND NOT A SMOOTHING SPLINE

Six points. `scipy.interpolate.CubicSpline` is interpolation, so evaluating it
at the original knots is the identity (measured: 3.55e-15), and the handoff
already burned that. A degree-d least-squares fit in t is a smoothness ladder
with physical rungs -- d=1 is constant velocity, d=2 constant acceleration --
and d = H-1 fits 6 points exactly, so THE IDENTITY IS A RUNG. That is a free
wiring check: if d=5 does not reproduce the raw number, the harness is wrong
before any verdict is read.

The origin is deliberately NOT forced into the fit. The model's own first
waypoint carries the initial speed, and anchoring at (0,0) would impose a
prior the raw trajectories do not have.

Usage:
    python analyze_smoothing.py results/headline/ref/per_sample.csv \
        --records_json ./data/nuscenes_records/nuscenes_val_scenespread.json
    python analyze_smoothing.py --selftest
"""

import argparse
import csv
import json
import sys
from collections import defaultdict

import numpy as np

# Preregistered before any real number was produced (sec.11.5: no post-hoc
# criteria) -- and CALIBRATED on the synthetic worlds, not on the data.
#
# The first draft used 0.50, and the self-test showed that is unreachable: in a
# world where the ENTIRE GoT-greedy gap is jitter by construction, the best rung
# recovers 45.7%. That is the instrument's ceiling, not a weak effect. A
# degree-1 fit through 6 points estimates 2 parameters, so it removes 2/3 of an
# iid variance and about 42% of the L2 -- the measurement matches the algebra.
#
# So the thresholds are expressed against that ceiling: STRONG is roughly
# two-thirds of what a pure-jitter world can give, WEAK is a fifth of it. The
# bias world is the negative control and produces a NEGATIVE differential, so
# the two worlds are separated by sign as well as size.
CEILING_PURE_JITTER = 0.457   # measured, _world("jitter")
RECOVER_STRONG = 0.30
RECOVER_WEAK = 0.10
HZ_STEP = 0.5             # seconds per waypoint (handoff sec.3)
WIRING_TOL = 2e-3         # csv stores 4 decimals; recomputation must match


# --------------------------------------------------------------------------- #
# smoothing
# --------------------------------------------------------------------------- #

def smooth_poly(traj, degree):
    """Least-squares polynomial in t, evaluated at the same knots.

    degree >= len(traj) - 1 is exact interpolation, i.e. the identity. That is
    kept rather than special-cased: it is the wiring check.
    """
    traj = np.asarray(traj, dtype=np.float64)
    n = traj.shape[0]
    if degree >= n - 1:
        return traj.copy()
    t = np.arange(1, n + 1, dtype=np.float64) * HZ_STEP
    out = np.empty_like(traj)
    for axis in range(traj.shape[1]):
        coef = np.polyfit(t, traj[:, axis], degree)
        out[:, axis] = np.polyval(coef, t)
    return out


def avg_l2(pred, gt):
    """avgL2@3s = mean over steps of the per-step L2 (handoff sec.7.3)."""
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)[: pred.shape[0]]
    return float(np.mean(np.linalg.norm(pred - gt, axis=1)))


# --------------------------------------------------------------------------- #
# scene-clustered paired statistics (handoff sec.8: report p_sc / ci_sc only)
# --------------------------------------------------------------------------- #

def scene_cluster_stats(diffs, scenes, n_boot=10000, seed=0):
    """Paired mean difference with a scene-CLUSTER bootstrap.

    Records from one scene are not independent -- resampling records instead of
    scenes is what made an earlier CI 3x too narrow (sec.1.4). Scenes are the
    unit, so scenes are what gets resampled.
    """
    diffs = np.asarray(diffs, dtype=np.float64)
    by_scene = defaultdict(list)
    for value, scene in zip(diffs, scenes):
        by_scene[scene].append(value)
    keys = sorted(by_scene)
    per_scene = [np.asarray(by_scene[k]) for k in keys]
    if not keys:
        return {"mean": float("nan"), "ci": (float("nan"), float("nan")), "p": 1.0}

    rng = np.random.RandomState(seed)
    boot = np.empty(n_boot)
    n = len(keys)
    for i in range(n_boot):
        pick = rng.randint(0, n, size=n)
        boot[i] = np.mean(np.concatenate([per_scene[j] for j in pick]))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    # two-sided bootstrap p: how often the resampled mean crosses zero
    share = float(np.mean(boot <= 0.0)) if np.mean(boot) > 0 else float(np.mean(boot >= 0.0))
    return {"mean": float(np.mean(diffs)), "ci": (float(lo), float(hi)),
            "p": min(1.0, 2.0 * share), "n_scenes": n}


# --------------------------------------------------------------------------- #
# io
# --------------------------------------------------------------------------- #

def load_rows(csv_path, records_json, out=print):
    with open(records_json) as handle:
        gt_by_token = {r["sample_token"]: np.asarray(r["waypoints"], dtype=np.float64)
                       for r in json.load(handle)}
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        for row in reader:
            if row.get("got_status") not in (None, "", "ok"):
                continue
            token = row.get("sample_token")
            if token not in gt_by_token or not row.get("got_pred"):
                continue
            entry = {"token": token, "scene": row.get("scene", token),
                     "gt": gt_by_token[token],
                     "got": np.asarray(json.loads(row["got_pred"]), dtype=np.float64)}
            if row.get("base_pred"):
                entry["base"] = np.asarray(json.loads(row["base_pred"]), dtype=np.float64)
            for key in ("got_avgL2@3s", "base_avgL2@3s"):
                if row.get(key):
                    entry[key] = float(row[key])
            rows.append(entry)
    out("columns in csv: %d, of which base_pred %s"
        % (len(columns), "PRESENT" if "base_pred" in columns else "ABSENT"))
    return rows


# --------------------------------------------------------------------------- #

def run(rows, out=print, allow_no_control=False):
    have_base = all("base" in r for r in rows) and rows
    horizon = rows[0]["got"].shape[0]

    raw_got = np.array([avg_l2(r["got"], r["gt"]) for r in rows])
    scenes = [r["scene"] for r in rows]

    # ---- wiring check 1: recomputation must reproduce the logged metric ----
    logged = [r["got_avgL2@3s"] for r in rows if "got_avgL2@3s" in r]
    if logged:
        delta = abs(float(np.mean(raw_got)) - float(np.mean(logged)))
        out("[wiring] recomputed GoT avgL2@3s %.4f vs csv %.4f  (diff %.5f)"
            % (np.mean(raw_got), np.mean(logged), delta))
        if delta > WIRING_TOL:
            raise SystemExit(
                "[fatal] recomputation does not reproduce the logged metric. "
                "Fix that before reading any verdict -- every number below is "
                "computed the same way.")

    if not have_base:
        out("")
        out("!! base_pred is ABSENT from this csv, so the CONTROL is missing.")
        out("   'smoothing helps GoT' cannot be told apart from 'smoothing")
        out("   helps any trajectory'. Re-run eval_got_nuscenes.py (it now logs")
        out("   base_pred) and use that csv.")
        if not allow_no_control:
            raise SystemExit("   refusing to print a verdict. --allow_no_control "
                             "prints the one-armed numbers only.")
        raw_base = None
    else:
        raw_base = np.array([avg_l2(r["base"], r["gt"]) for r in rows])
        gap = float(np.mean(raw_got) - np.mean(raw_base))
        out("[baseline] greedy %.4f  GoT %.4f  gap %+.4f"
            % (np.mean(raw_base), np.mean(raw_got), gap))

    out("")
    out("degree  1=const velocity ... %d=identity" % (horizon - 1))
    out("  %-6s %10s %10s %12s %22s %8s"
        % ("d", "GoT", "greedy", "gain_got", "gain_got - gain_base", "p_sc"))

    results = []
    for degree in range(1, horizon):
        sm_got = np.array([avg_l2(smooth_poly(r["got"], degree), r["gt"]) for r in rows])
        gain_got = raw_got - sm_got                     # positive = improved
        if raw_base is None:
            out("  %-6d %10.4f %10s %+12.4f %22s %8s"
                % (degree, np.mean(sm_got), "-", np.mean(gain_got), "-", "-"))
            results.append((degree, float(np.mean(gain_got)), None, None))
            continue
        sm_base = np.array([avg_l2(smooth_poly(r["base"], degree), r["gt"]) for r in rows])
        gain_base = raw_base - sm_base
        diff = gain_got - gain_base
        stats = scene_cluster_stats(diff, scenes)
        out("  %-6d %10.4f %10.4f %+12.4f   %+8.4f [%+.4f,%+.4f] %8.4f"
            % (degree, np.mean(sm_got), np.mean(sm_base), np.mean(gain_got),
               stats["mean"], stats["ci"][0], stats["ci"][1], stats["p"]))
        results.append((degree, float(np.mean(gain_got)), float(np.mean(gain_base)), stats))

    # ---- wiring check 2: the identity rung ----
    last = results[-1]
    if abs(last[1]) > 1e-9 or (last[2] is not None and abs(last[2]) > 1e-9):
        raise SystemExit(
            "[fatal] degree %d fits %d points exactly, so it MUST be the "
            "identity and gain MUST be 0. It is not, so the smoother or the "
            "metric is wrong." % (horizon - 1, horizon))
    out("  [wiring] identity rung d=%d gives gain exactly 0 -- ok" % (horizon - 1))

    if raw_base is None:
        return results

    # ---- preregistered verdict ----
    gap = float(np.mean(raw_got) - np.mean(raw_base))
    best = max((r for r in results if r[3] is not None), key=lambda r: r[3]["mean"])
    degree, _, _, stats = best
    recovered = stats["mean"] / gap if gap > 0 else 0.0
    excludes_zero = stats["ci"][0] > 0.0

    out("")
    out("best rung d=%d: differential %+.4f, recovers %.1f%% of the %+.4f gap"
        % (degree, stats["mean"], 100 * recovered, gap))
    out("  (a world whose gap is ENTIRELY jitter gives %.1f%% -- the ceiling of "
        "this smoother, not of the effect)" % (100 * CEILING_PURE_JITTER))
    if excludes_zero and recovered >= RECOVER_STRONG:
        out("VERDICT: CONFIRMS_VARIANCE -- a pure variance-reduction operator, "
            "which cannot know where the car went, recovers most of the gap on "
            "the GoT arm and not on greedy. sec.1.2's reading is independently "
            "supported.")
    elif excludes_zero and recovered >= RECOVER_WEAK:
        out("VERDICT: WEAK -- the differential is real but small. Report it as "
            "consistent with sec.1.2, not as confirmation.")
    else:
        out("VERDICT: NULL -- smoothing does not help GoT more than greedy. "
            "sec.1.2 stands on its own evidence; this adds nothing. Report the "
            "null (E3 was predicted null).")

    # ---- where does the gain sit? sec.1.2: 46.2% of loss in the worst 5% ----
    loss = raw_got - raw_base
    order = np.argsort(-loss)
    sm_got = np.array([avg_l2(smooth_poly(r["got"], degree), r["gt"]) for r in rows])
    gain = raw_got - sm_got
    out("")
    out("concentration of the gain at d=%d (records ranked by GoT-greedy loss):"
        % degree)
    for label, frac in (("worst 5%", 0.05), ("worst 20%", 0.20), ("all", 1.0)):
        k = max(1, int(round(len(rows) * frac)))
        out("    %-10s mean gain %+.4f   (mean loss %+.4f)"
            % (label, float(np.mean(gain[order[:k]])), float(np.mean(loss[order[:k]]))))
    out("    if the gain concentrates where the loss is, smoothing is removing")
    out("    exactly the excursions sec.1.2 attributed to variance.")
    return results


# --------------------------------------------------------------------------- #
# self-test: two worlds, because a check that only sees one proves nothing
# --------------------------------------------------------------------------- #

def _world(kind, n_scenes=30, per_scene=4, seed=0):
    """jitter: GoT = greedy + high-frequency noise (smoothing must recover it)
       bias:   GoT = greedy + a constant offset (smoothing must NOT recover it)"""
    rng = np.random.RandomState(seed)
    rows = []
    for scene in range(n_scenes):
        for _ in range(per_scene):
            t = np.arange(1, 7) * HZ_STEP
            speed = 4.0 + rng.randn() * 0.5
            gt = np.stack([speed * t, 0.1 * t ** 2], axis=1)
            base = gt + rng.randn(6, 2) * 0.30
            if kind == "jitter":
                got = base + rng.randn(6, 2) * 0.60
            else:
                got = base + np.array([0.45, 0.0])
            rows.append({"token": "%d" % len(rows), "scene": "s%02d" % scene,
                         "gt": gt, "got": got, "base": base})
    return rows


def _selftest():
    failures = []

    def check(label, ok, detail=""):
        print("  %-40s %s %s" % (label, "ok" if ok else "FAIL", detail))
        if not ok:
            failures.append(label)

    traj = np.array([[1.0, 0.1], [2.1, 0.3], [2.9, 0.8], [4.2, 1.4],
                     [4.8, 2.3], [6.3, 3.1]])
    check("identity at degree n-1",
          np.allclose(smooth_poly(traj, 5), traj, atol=1e-9))
    check("degree 1 is a straight line",
          np.allclose(np.diff(smooth_poly(traj, 1), n=2, axis=0), 0, atol=1e-9))
    check("smoothing reduces roughness",
          np.abs(np.diff(smooth_poly(traj, 2), n=2, axis=0)).sum()
          < np.abs(np.diff(traj, n=2, axis=0)).sum())

    stats = scene_cluster_stats(np.ones(40) * 0.5, ["s%d" % (i // 4) for i in range(40)])
    check("clear effect excludes zero", stats["ci"][0] > 0, str(stats["ci"]))
    noise = np.random.RandomState(1).randn(400) * 0.5
    stats = scene_cluster_stats(noise, ["s%d" % (i // 4) for i in range(400)])
    check("pure noise includes zero", stats["ci"][0] < 0 < stats["ci"][1], str(stats["ci"]))
    a = scene_cluster_stats(noise, ["s%d" % (i // 4) for i in range(400)])
    check("bootstrap is reproducible under seed", a["ci"] == stats["ci"])

    sink = []
    run(_world("jitter"), out=sink.append)
    jitter_text = "\n".join(sink)
    check("jitter world -> CONFIRMS_VARIANCE", "CONFIRMS_VARIANCE" in jitter_text,
          [l for l in sink if l.startswith("VERDICT")][:1])

    sink = []
    run(_world("bias"), out=sink.append)
    bias_text = "\n".join(sink)
    check("bias world -> NULL", "VERDICT: NULL" in bias_text,
          [l for l in sink if l.startswith("VERDICT")][:1])
    check("the two worlds get different verdicts", jitter_text != bias_text)
    # the bias world must fail by SIGN, not merely by size: smoothing a biased
    # trajectory helps the unbiased arm more, so the differential goes negative
    check("bias world differential is negative",
          any("-0." in l and "[" in l for l in sink if l.strip().startswith(("1", "2", "3"))))

    rows = _world("jitter")
    for row in rows:
        row.pop("base")
    try:
        run(rows, out=lambda *a: None)
        caught = False
    except SystemExit:
        caught = True
    check("refuses a verdict without the control", caught)

    if failures:
        print("\nSELFTEST FAILED: %s" % failures)
        return 1
    print("\nSELFTEST PASS")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?")
    ap.add_argument("--records_json")
    ap.add_argument("--allow_no_control", action="store_true",
                    help="print one-armed numbers from a csv without base_pred")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(_selftest())
    if not args.csv or not args.records_json:
        sys.exit("need a per_sample.csv and --records_json")
    rows = load_rows(args.csv, args.records_json)
    print("records: %d over %d scenes" % (len(rows), len({r["scene"] for r in rows})))
    if not rows:
        sys.exit("[fatal] no usable rows")
    run(rows, allow_no_control=args.allow_no_control)


if __name__ == "__main__":
    main()
