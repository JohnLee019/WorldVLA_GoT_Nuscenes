"""
Are the candidate-scoring signals redundant with each other? Screening check for
whether a NEW scene-grounded signal (e.g. world-model plausibility) can be
expected to help, before spending 37 h training one.

Motivation. Each component correlates with candidate quality at about the same
strength -- kinematic +0.501, command +0.208, model self-likelihood +0.518
(Spearman vs -true_error, 600 records) -- yet their weighted combination scores
+0.487, i.e. WORSE than the best single component, and selection_top1 sits at
0.27 no matter which are used, reweighted, renormalised or added. That pattern is
the signature of mutually redundant signals: a second predictor correlated with
the first adds noise, not information.

The likelihood was the obvious cheap test of "add a signal that looks at the
image" and it changed nothing. A world model would be a second, far more
expensive signal derived from the SAME image, so the prior should be that it is
redundant too. This script measures that prior instead of assuming it:

  rho(lik, kin)          direct redundancy of the two orderings.
  ★ head-to-head        among records where ranking by likelihood and by
                        kinematic pick DIFFERENT candidates, how often is the
                        likelihood's pick the better one. THIS is the decision
                        number: 0.5 means the image-derived signal adds no
                        usable information even where it disagrees, so a world
                        model -- another signal from the same image -- should be
                        expected to behave the same way.
  disagreement rate      how often they differ at all. A low rate alone already
                        caps how much a new signal could change.

A partial correlation is also printed, but read the head-to-head first: the
first-order partial formula divides by sqrt(1 - rho(lik,kin)^2), so it becomes
unstable exactly when the two signals agree closely, which is the case under
test. On a synthetic pool where the likelihood was a near-copy of kinematic
(rho = 0.994) it reported +0.39, higher than for a genuinely independent signal
(+0.02) -- an artefact of the vanishing denominator, not information. It is
therefore restricted to records where the two signals are not near-collinear and
flagged as fragile.

    python check_signal_redundancy.py results/lik/lik_full/per_sample.csv
"""

import argparse
import math
from collections import defaultdict

from analyze_got_csv import (_f, _list, _mean, _median, cluster_bootstrap_ci,
                             spearman)


def partial_spearman(r_xy, r_xz, r_zy):
    """Spearman partial correlation of x with y, controlling for z.

    Standard first-order formula applied to rank correlations. None when the
    denominator collapses (one control explains a predictor almost exactly), so
    a degenerate record is dropped rather than contributing a spurious +-1.
    """
    den = (1.0 - r_xz ** 2) * (1.0 - r_zy ** 2)
    if den <= 1e-9:
        return None
    return (r_xy - r_xz * r_zy) / math.sqrt(den)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path", help="per_sample.csv from a run with --w_likelihood > 0")
    ap.add_argument("--key", default="avgL2@3s")
    args = ap.parse_args()

    import csv
    with open(args.csv_path, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r.get("got_status") == "ok"]

    pair = defaultdict(list)        # label -> [per-record rho]
    partial = []                    # partial rho(lik ; y | kin), fragile
    partial_by_scene = defaultdict(list)
    agree = []                      # argmax(lik) == argmax(kin)
    h2h = []                        # lik's pick beats kin's, where they differ
    h2h_by_scene = defaultdict(list)
    n_used = 0

    for r in rows:
        kin, cmd = _list(r, "got_cand_kin"), _list(r, "got_cand_cmd")
        lik, val = _list(r, "got_cand_lik"), _list(r, "got_cand_vals")
        if not (kin and cmd and lik and val):
            continue
        if not (len(kin) == len(cmd) == len(lik) == len(val) >= 3):
            continue
        n_used += 1
        y = [-v for v in val]                       # higher = better, like the scores

        r_lk = spearman(lik, kin)
        r_lc = spearman(lik, cmd)
        r_kc = spearman(kin, cmd)
        r_ly, r_ky = spearman(lik, y), spearman(kin, y)
        for lbl, v in (("lik~kin", r_lk), ("lik~cmd", r_lc), ("kin~cmd", r_kc),
                       ("lik~quality", r_ly), ("kin~quality", r_ky)):
            if not math.isnan(v):
                pair[lbl].append(v)

        # fragile near collinearity -- see the module docstring
        if (not any(math.isnan(v) for v in (r_ly, r_lk, r_ky))) and abs(r_lk) < 0.9:
            p = partial_spearman(r_ly, r_lk, r_ky)
            if p is not None:
                partial.append(p)
                partial_by_scene[r.get("scene", "?")].append(p)

        # head-to-head: only records where the two signals actually disagree can
        # tell us anything about which one is better informed
        i_lik = max(range(len(lik)), key=lambda i: lik[i])
        i_kin = max(range(len(kin)), key=lambda i: kin[i])
        agree.append(1.0 if i_lik == i_kin else 0.0)
        if i_lik != i_kin and val[i_lik] != val[i_kin]:
            w = 1.0 if val[i_lik] < val[i_kin] else 0.0   # lower error = better
            h2h.append(w)
            h2h_by_scene[r.get("scene", "?")].append(w)

    print(f"{args.csv_path}: {len(rows)} ok rows, {n_used} with all three score "
          f"components logged")
    if not n_used:
        print("  no usable rows -- the run needs --w_likelihood > 0 so that "
              "got_cand_lik is written")
        return

    print("\n=== pairwise agreement between signals (per-record Spearman) ===")
    print(f"  {'pair':<14} {'n':>5} {'mean':>8} {'median':>8}")
    for lbl in ("lik~kin", "lik~cmd", "kin~cmd", "lik~quality", "kin~quality"):
        v = pair.get(lbl, [])
        if v:
            print(f"  {lbl:<14} {len(v):>5} {_mean(v):>+8.4f} {_median(v):>+8.4f}")

    print("\n=== the decision number: head-to-head where they disagree ===")
    print(f"  argmax(likelihood) != argmax(kinematic) in "
          f"{1.0 - _mean(agree):.1%} of {len(agree)} records")
    if h2h:
        lo, hi = cluster_bootstrap_ci(h2h_by_scene)
        print(f"  of those {len(h2h)} disagreements ({len(h2h_by_scene)} scenes), the "
              f"LIKELIHOOD's\n  pick was the better one {_mean(h2h):.3f} of the time  "
              f"ci_sc=[{lo:.3f}, {hi:.3f}]   (0.5 = no information)")
    else:
        print("  the two signals never disagreed -> fully redundant by construction")

    if partial:
        plo, phi = cluster_bootstrap_ci(partial_by_scene)
        print(f"\n  [fragile, see docstring] partial rho(lik ; quality | kin) over the "
              f"{len(partial)}/{n_used}\n  records with |rho(lik,kin)| < 0.9:  "
              f"mean={_mean(partial):+.4f}  median={_median(partial):+.4f}  "
              f"ci_sc=[{plo:+.4f}, {phi:+.4f}]")

    print("\n=== how to read it ===")
    w = _mean(h2h) if h2h else 0.5
    lo, hi = cluster_bootstrap_ci(h2h_by_scene) if h2h else (float("nan"), float("nan"))
    if not h2h or (lo <= 0.5 <= hi):
        print(f"  Where the image-derived signal and trajectory geometry disagree, the")
        print(f"  image-derived one is right {w:.1%} of the time and the interval covers")
        print(f"  0.5: it adds no usable information even on the records where it has an")
        print(f"  opinion of its own. A world-model plausibility score is another signal")
        print(f"  from the SAME image, so expect the same. Train the WM for the evaluator")
        print(f"  role it was designed for (the four WM metrics of section 7.1.1) and do")
        print(f"  NOT budget the 37 h against fixing selection.")
    elif lo > 0.5:
        print(f"  The image-derived signal wins {w:.1%} of disagreements, interval clear")
        print(f"  of 0.5: it IS better informed where it differs, yet adding it left")
        print(f"  top-1 at 0.27. That points at the COMBINATION rule rather than the")
        print(f"  signal -- try likelihood-only selection")
        print(f"  (--final_weights 0 0 --w_likelihood 1) before writing the WM off.")
    else:
        print(f"  The image-derived signal LOSES its disagreements ({w:.1%}, interval")
        print(f"  below 0.5): it is actively worse informed than trajectory geometry.")
        print(f"  Adding it should hurt, which matches path_score ranking below both.")
    print("\n  (CIs are scene-clustered: records within a scene are consecutive")
    print("   keyframes of one manoeuvre, not independent trials.)")


if __name__ == "__main__":
    main()
