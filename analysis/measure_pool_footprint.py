"""Where does OUR candidate pool sit in NAVSIM vocabulary coordinates? (0 GPU)

WHY THIS IS THE NEXT MEASUREMENT

  handoff Step 2.9(d)(1) and Step 2.10(e) produced exactly one design
  requirement for the pivot:

      the GoT candidate set must cover the conservative end (deceleration,
      stop). A tight block sitting at v*1.0 or above has GAP_dep <= 0 --
      deliberation there is worth less than shipping one fixed plan.

  That was measured on a SYNTHETIC vocabulary grid. Our actual candidate
  generator has never been expressed in those coordinates, and the nuScenes
  evidence is ambiguous rather than reassuring:

      first-step longitudinal span   0.2703 m   (sec.1.10a -- very tight)
      3 s longitudinal spread (std)  6.9353 m   (sec.1.10a -- not tight)
      |pool median - GT| at 3 s      5.9757 m   (the pool overshoots)

  Tight at the first step, wide at the endpoint, and biased long. Which of
  those governs the NAVSIM speed scale is not derivable from the numbers
  already in the handoff, so it is measured here.

  This runs BEFORE navtrain preprocessing and retraining (weeks), which is
  the sec.11.8 protocol: measure the ceiling before building. It has been
  right five times out of five.

WHAT IT COMPUTES

  NAVSIM's vocabulary is (speed_scale x curvature) applied to the ego's
  CURRENT speed, so each of our candidates is mapped into those coordinates:

      v_ref  = constant velocity at the ego's speed at t=0
             = 6 * gt[0, lon]              (0.5 s step, 3 s horizon)
      scale_j     = lon_j(3s) / v_ref
      curvature_j = 2 * lat_j(3s) / lon_j(3s)^2      (arc, matching
                    vocab_agent.py:96-99's y = r*(1-cos(k*arc)))

  Sign convention matches: nuScenes ego frame and vocab_agent both put
  +y to the left, so +curvature is a left turn in both.

WHAT IT CANNOT SAY

  This is a nuScenes-trained model on a 3 s / 6-point output, read in the
  coordinates of a 4 s / 8-point benchmark. It is a PRIOR on what a
  NAVSIM-trained pool would look like, not a measurement of one. What
  transfers is the mechanism, not the number: sec.1.10a attributes the tight
  span to beam search sharing prefixes, which is architectural and survives
  a change of dataset.

  v_ref uses GT's first step, which contains that interval's acceleration and
  is therefore slightly STRONGER than true current speed (same caveat as
  sec.1.10c(i)). It biases scale_j downward, i.e. toward the favourable
  answer, so a TIGHT_HIGH verdict is conservative.

Usage (VLA_GoT env, from the repo root -- got_drive must be importable):
    python measure_pool_footprint.py results/fusion/final_top3/per_sample.csv \
        --records_json ./data/nuscenes_records/nuscenes_val_scenespread.json
    python measure_pool_footprint.py --selftest
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from analyze_oracle_structure import HZ, load_gt, load_pools

# --- preregistered thresholds (fixed before the run, sec.11.5) --------------
CONSERVATIVE = 0.50   # Step 2.10a: the positive-GAP peak is v*0.45-0.60
NEG_ZONE = 1.25       # Step 2.10a: GAP_dep goes negative from this row
COVER_MIN = 0.50      # a pool "covers the conservative end" if it does so in
                      # at least this share of records
MIN_LON = 0.5         # m. below this the record is parked and the ratio is
                      # numerically meaningless (same bar as sec.1.10)
OUT_TSV = "pool_footprint.tsv"


def footprint(pool):
    """(scale, curvature) per candidate, or None if the record is parked."""
    gt, wps = pool["gt"], pool["wps"]
    v_ref = (HZ + 1) * float(gt[0, 0])
    if v_ref < MIN_LON * (HZ + 1):
        return None
    lon = wps[:, HZ, 0].astype(float)
    lat = wps[:, HZ, 1].astype(float)
    safe = np.where(np.abs(lon) < MIN_LON, np.nan, lon)
    return np.stack([lon / v_ref, 2.0 * lat / (safe ** 2)], axis=1)


def summarise(rows, label, out=print):
    values = np.asarray(rows, float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        out("    %-26s (no finite values)" % label)
        return
    out("    %-26s min %7.3f  p25 %7.3f  med %7.3f  p75 %7.3f  max %7.3f"
        % (label, values.min(), np.percentile(values, 25),
           np.median(values), np.percentile(values, 75), values.max()))


def analyse(prints, out=print, write_tsv=None):
    """prints: list of (n_cand, 2) arrays, one per usable record."""
    sizes = [p.shape[0] for p in prints]
    modal = max(set(sizes), key=sizes.count)
    kept = [p for p in prints if p.shape[0] == modal]
    out("[1] POOLS  usable %d / modal pool size %d (kept %d)"
        % (len(prints), modal, len(kept)))
    out("")

    scales = np.concatenate([p[:, 0] for p in kept])
    curvs = np.concatenate([p[:, 1] for p in kept])
    out("[2] OUR POOL IN NAVSIM COORDINATES  (all candidates pooled)")
    summarise(scales, "speed_scale", out)
    summarise(curvs, "curvature [1/m]", out)
    out("")

    per_min = np.array([np.nanmin(p[:, 0]) for p in kept])
    per_max = np.array([np.nanmax(p[:, 0]) for p in kept])
    per_med = np.array([np.nanmedian(p[:, 0]) for p in kept])
    out("[3] PER-RECORD SPREAD  (this is what the selector actually sees)")
    summarise(per_med, "pool median scale", out)
    summarise(per_max - per_min, "pool scale span", out)
    covers = float(np.mean(per_min <= CONSERVATIVE))
    in_neg = float(np.mean(per_med >= NEG_ZONE))
    out("    records whose pool reaches v*%.2f or slower : %.1f%%  (need >=%.0f%%)"
        % (CONSERVATIVE, 100.0 * covers, 100.0 * COVER_MIN))
    out("    records whose pool MEDIAN sits at v*%.2f or faster : %.1f%%"
        % (NEG_ZONE, 100.0 * in_neg))
    out("")

    ranked = np.stack([p[np.argsort(p[:, 0])] for p in kept], axis=0)
    shape = np.nanmedian(ranked, axis=0)
    out("[4] REPRESENTATIVE FOOTPRINT  (median over records, rank by speed)")
    out("    %5s %13s %13s" % ("rank", "speed_scale", "curvature"))
    for i, (s, k) in enumerate(shape):
        out("    %5d %13.3f %13.5f" % (i, s, k))
    if write_tsv:
        with open(write_tsv, "w") as handle:
            handle.write("rank\tspeed_scale\tcurvature\n")
            for i, (s, k) in enumerate(shape):
                handle.write("%d\t%.6f\t%.6f\n" % (i, s, k))
        out("    written to %s -- feed this to navsim_tools/eval_footprint_pool.py"
            % write_tsv)
    out("")

    if covers >= COVER_MIN and shape[0, 0] <= CONSERVATIVE:
        verdict = "COVERS"
        out("VERDICT: COVERS -- the pool reaches the conservative end in most")
        out("  records, which is the Step 2.9(d)(1) requirement. The NAVSIM-side")
        out("  lookup should return a positive GAP_dep; run it to confirm.")
    elif float(np.median(per_med)) >= NEG_ZONE:
        verdict = "TIGHT_HIGH"
        out("VERDICT: TIGHT_HIGH -- the pool sits at or above v*%.2f, which is"
            % NEG_ZONE)
        out("  the region where Step 2.10a measured GAP_dep < 0. Deliberation")
        out("  over this pool is worth LESS than shipping one fixed plan.")
        out("  CANDIDATE GENERATION MUST CHANGE BEFORE navtrain RETRAINING.")
        out("  Note sec.1.10a: the tight span comes from beam search sharing")
        out("  prefixes, so temperature will not fix it.")
    else:
        verdict = "PARTIAL"
        out("VERDICT: PARTIAL -- the pool neither clearly covers the")
        out("  conservative end nor clearly sits in the negative region.")
        out("  The NAVSIM-side lookup decides this one; it is not readable off")
        out("  these marginals.")
    out("  In all three cases the number that decides is GAP_dep from")
    out("  navsim_tools/eval_footprint_pool.py, not this verdict.")
    return verdict, shape


# --- self-test --------------------------------------------------------------

def _world(kind, n_rec=120, n_cand=8, seed=0):
    rng = np.random.RandomState(seed)
    out = []
    for _ in range(n_rec):
        if kind == "covers":
            scales = np.linspace(0.2, 1.4, n_cand)
        elif kind == "tight_high":
            scales = 1.35 + 0.03 * rng.randn(n_cand)
        else:
            scales = 0.85 + 0.05 * rng.randn(n_cand)
        curv = 0.002 * rng.randn(n_cand)
        out.append(np.stack([scales, curv], axis=1))
    return out


def _selftest():
    sink = []
    failures = []
    for kind, expected in [("covers", "COVERS"), ("tight_high", "TIGHT_HIGH"),
                           ("tight_mid", "PARTIAL")]:
        got, _ = analyse(_world(kind), out=sink.append)
        ok = got == expected
        if not ok:
            failures.append((kind, expected, got))
        print("  world %-12s expected %-11s got %-11s %s"
              % (kind, expected, got, "ok" if ok else "FAIL"))
    if failures:
        print("\nSELFTEST FAILED: %s" % failures)
        return 1
    print("\nSELFTEST PASS -- the three pool shapes are separated")
    return 0


def main():
    ap = argparse.ArgumentParser(
        "express the GoT candidate pool in NAVSIM (speed_scale, curvature) "
        "coordinates (0 GPU)")
    ap.add_argument("csv", nargs="*",
                    help="per_sample.csv with got_cand_wps "
                         "(results/fusion/final_top3 -- its pool equals ref)")
    ap.add_argument("--records_json", default=None)
    ap.add_argument("--out_tsv", default=OUT_TSV)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        sys.exit(_selftest())
    if not a.csv:
        ap.error("give per_sample.csv files (or --selftest)")
    if not a.records_json:
        ap.error("--records_json is required: the reference speed is read from GT")

    gt = load_gt(a.records_json)
    pools, skipped = load_pools(a.csv, gt)
    if not pools:
        sys.exit("[fatal] no usable pools. skipped: %s\n"
                 "  got_cand_wps was added 2026-08-03; results/headline/* "
                 "predate it. Use results/fusion/final_top3/per_sample.csv."
                 % skipped)
    if skipped:
        print("[load] skipped %s" % skipped)

    prints, parked = [], 0
    for pool in pools:
        mapped = footprint(pool)
        if mapped is None:
            parked += 1
            continue
        prints.append(mapped)
    print("[load] pools %d / parked-excluded %d\n" % (len(prints), parked))
    if len(prints) < 20:
        sys.exit("[fatal] only %d usable pools" % len(prints))
    analyse(prints, write_tsv=a.out_tsv)


if __name__ == "__main__":
    main()
