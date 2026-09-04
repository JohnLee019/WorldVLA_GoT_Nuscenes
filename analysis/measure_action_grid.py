"""What is the waypoint quantisation grid, really? (CPU, no GPU, no torch)

sec.1.16(f) left three numbers disagreeing about the action grid: the lattice
measured in the stored predictions (x 0.2855 / y 0.1305 m), a back-computed
0.2253 m, and `predict_waypoints_head`'s docstring claim of "RMS 0.065 m". The
handoff asked for the fix to come from the code rather than from arithmetic:
run the encode -> token -> decode -> un-normalise round trip and let it say what
the grid is.

That is what this does, and the answer is that there was never a disagreement.
The grid is (max - min) / 255 per axis -- 255 because `decode_token_ids_to_actions`
reconstructs to BIN CENTRES, and 256 edges have 255 midpoints. Every one of the
disagreeing numbers is reproduced by substituting the mini `nuscenes_norm.json`
that sits on disk for the trainval fit the incumbent was actually trained on
(sec.1.15 flagged that same file as overwritten; this is the second thing it broke).

!! THE GATE IS THE POINT. A norm range is not something to assume -- pass the
candidates and let the observed lattice pick. If none of them reproduces the
lattice in the csv, the tool exits non-zero instead of reporting a grid, because
at that point you do not know what grid the run used and any quantisation number
derived from it is fiction.

!! WHICH RMS TO QUOTE. Three different quantities are all called "quantisation
RMS" and they differ by 40%:
    per-axis           x 0.0770 / y 0.0337 m   (each coordinate on its own)
    pooled coordinate  0.0594 m                (both axes' errors in one pool)
    2-D displacement   0.0840 m                (what L2 metrics actually see)
Waypoint error is reported as a 2-D displacement everywhere in sec.1/sec.7, so the
displacement figure is the one that is commensurable with avgL2@3s. State which
one you mean.

Usage
-----
  python analysis/measure_action_grid.py --selftest          # 11/11, needs no data
  python analysis/measure_action_grid.py \
      --csv results/base_ckpt/incumbent_cont2_ep1/per_sample.csv

Exit codes: 0 grid identified / 1 no candidate reproduces the lattice / 2 the
source this tool transcribes has changed.
"""
import argparse
import ast
import csv as csvmod
import json
import os
import re
import sys

import numpy as np

# ---------------------------------------------------------------------------
# The arithmetic below is transcribed from three places. Transcriptions rot, so
# each one is pinned to a regex that must still be present in the source file --
# see check_source(). If the source moves, this tool fails instead of quietly
# reporting a grid the code no longer produces.
# ---------------------------------------------------------------------------
SOURCE_CHECKS = [
    ("data/item_processor.py", r"self\.bins\s*=\s*np\.linspace\(self\.min_action,\s*self\.max_action,\s*self\.n_bins\)"),
    ("data/item_processor.py", r"self\.n_bins,\s*self\.min_action,\s*self\.max_action\s*=\s*256,\s*-1,\s*1"),
    ("data/item_processor.py", r"bins\s*=\s*np\.linspace\(-1,\s*1,\s*256\)"),
    ("data/item_processor.py", r"bin_centers\s*=\s*\(bins\[:-1\]\s*\+\s*bins\[1:\]\)\s*/\s*2\.0"),
    ("data/item_processor.py", r"discretized_actions\s*=\s*dis_action\s*-\s*1\s*-\s*10004"),
    ("data/item_processor.py", r"np\.digitize\(norm_action,\s*self\.bins\)\s*\+\s*self\.token2id\(self\.action_start_token\)\s*\+\s*1"),
    ("eval_nuscenes.py", r"return\s*\(norm_wp\s*\+\s*1\)\s*/\s*2\s*\*\s*\(wp_max\s*-\s*wp_min\s*\+\s*1e-8\)\s*\+\s*wp_min"),
]

N_BINS = 256
START_ID = 10004                                    # token2id(action_start_token)
BINS = np.linspace(-1.0, 1.0, N_BINS)               # 256 edges
CENTERS = (BINS[:-1] + BINS[1:]) / 2.0              # 255 reconstruction levels

# sec.1.15: data/nuscenes_records/nuscenes_norm.json was overwritten with mini
# values, so the trainval fit lived here as a constant. It is now ALSO written out
# as data/nuscenes_records/nuscenes_norm_v1.0-trainval.json (same numbers, sourced
# from this constant / run_state.sh's gate) so that eval runs and the README can
# point at a file instead of at prose. Keep this constant as the in-code copy: this
# script's whole job is to compare the two grids, so it must not depend on which
# one happens to sit at the default path.
TRAINVAL_FIT = {
    "min": [-3.0241, -16.6451],
    "max": [69.786, 16.6451],
    "_note": "trainval fit, sec.1.15 -- the range the incumbent was trained with",
}


def check_source(repo_root):
    """Fail loudly if the code this tool transcribes has moved."""
    missing = []
    for rel, pat in SOURCE_CHECKS:
        path = os.path.join(repo_root, rel)
        try:
            with open(path, encoding="utf-8") as f:
                src = f.read()
        except OSError:
            missing.append(f"{rel}: not readable")
            continue
        if not re.search(pat, src):
            missing.append(f"{rel}: /{pat}/")
    if missing:
        print("SOURCE DRIFT -- this tool's transcription no longer matches the code:")
        for m in missing:
            print("   " + m)
        sys.exit(2)
    print(f"[source] {len(SOURCE_CHECKS)}/{len(SOURCE_CHECKS)} transcription checks PASS")


# --- the round trip, one function per stage in the real pipeline ------------
def norm_action(a, lo, hi):                         # item_processor.norm_action
    return np.clip(2 * (a - lo) / (hi - lo + 1e-8) - 1, -1.0, 1.0)


def encode(a, lo, hi):                              # item_processor.process_action
    return np.digitize(norm_action(a, lo, hi), BINS) + START_ID + 1


def decode(tok):                                    # decode_token_ids_to_actions
    return CENTERS[np.clip(tok - 1 - START_ID - 1, 0, CENTERS.shape[0] - 1)]


def unnorm(nw, lo, hi):                             # eval_nuscenes.unnorm_waypoints
    return (nw + 1) / 2 * (hi - lo + 1e-8) + lo


def grid_of(lo, hi):
    """Every value this axis can decode to, in metres."""
    return unnorm(CENTERS, lo, hi)


def fit_lattice(v):
    """Least-squares step of the lattice present in `v`. Returns (step, max resid)."""
    u = np.unique(v)
    if u.size < 3:
        return None, None
    gaps = np.diff(u)
    g = gaps[gaps > 1e-9].min()
    k = np.round((u - u.min()) / g)                 # integer lattice index
    (step, off), *_ = np.linalg.lstsq(np.stack([k, np.ones_like(k)], 1), u, rcond=None)
    return float(step), float(np.abs(u - (step * k + off)).max())


def identify_grid(pred, cands, tol):
    """Which candidate norm produces the lattice present in `pred`?

    Returns (obs_steps, obs_resid, rows, winner) where winner is
    (name, lo, hi, steps) or None. Kept separate from main() so the self-test
    can drive it on synthetic data with a known answer.
    """
    n_dim = pred.shape[1]
    obs, resid = [], []
    for d in range(n_dim):
        s, r = fit_lattice(pred[:, d])
        if s is None:
            return None, None, [], None
        obs.append(s)
        resid.append(r)

    rows, winner = [], None
    for name, st in cands.items():
        lo, hi = np.asarray(st["min"], float), np.asarray(st["max"], float)
        steps = [float(np.diff(grid_of(lo[d], hi[d])).mean()) for d in range(n_dim)]
        ok = all(abs(steps[d] - obs[d]) <= tol for d in range(n_dim))
        if ok and winner is None:               # first match wins
            winner = (name, lo, hi, steps)
        rows.append((name, steps, ok))
    return obs, resid, rows, winner


# ---------------------------------------------------------------------------
# Self-test. sec.11.7: a check that only ever runs on the world where the answer
# is "yes" can be passed by a tool that always says yes. So every case below has
# a counterpart where the right answer is different, and the pair has to split.
# ---------------------------------------------------------------------------
MINI_FIT = {"min": [-1.0193, -15.1637], "max": [56.432, 15.1637]}


def selftest():
    rng = np.random.default_rng(0)
    cands = {"trainval": TRAINVAL_FIT, "mini": MINI_FIT}
    results = []

    def check(name, ok, detail=""):
        results.append((name, bool(ok), detail))

    def sample_on_grid(st, n=3600):
        lo, hi = np.asarray(st["min"], float), np.asarray(st["max"], float)
        g = [grid_of(lo[d], hi[d]) for d in (0, 1)]
        # a contiguous band of levels, like real predictions (94 / 53 distinct)
        idx = [rng.integers(60, 160, n), rng.integers(100, 155, n)]
        return np.stack([g[d][idx[d]] for d in (0, 1)], 1), lo, hi

    # 1/2 -- the discriminating pair: same tool, two different true grids
    for want in ("trainval", "mini"):
        pred, _, _ = sample_on_grid(TRAINVAL_FIT if want == "trainval" else MINI_FIT)
        _, _, _, win = identify_grid(pred, cands, 2e-3)
        check(f"identifies the {want} grid", win is not None and win[0] == want,
              "picked " + (win[0] if win else "nothing"))

    # 3 -- the world with no grid at all must NOT be identified
    cont = np.stack([rng.uniform(0, 45, 3600), rng.uniform(-7, 7, 3600)], 1)
    _, _, _, win = identify_grid(np.round(cont, 3), cands, 2e-3)
    check("refuses a continuous (off-lattice) arm", win is None,
          "picked " + (win[0] if win else "nothing"))

    # 4 -- csv rounding must not break identification (real csvs store 3 decimals)
    pred, _, _ = sample_on_grid(TRAINVAL_FIT)
    obs, _, _, win = identify_grid(np.round(pred, 3), cands, 2e-3)
    check("survives the csv's 1e-3 rounding", win is not None and win[0] == "trainval",
          f"step {obs[0]:.6f} vs true {np.diff(grid_of(-3.0241, 69.786)).mean():.6f}")

    # 5 -- values already on the grid round-trip EXACTLY
    lo, hi = np.array(TRAINVAL_FIT["min"]), np.array(TRAINVAL_FIT["max"])
    on = sample_on_grid(TRAINVAL_FIT)[0]
    back = np.stack([unnorm(decode(encode(on[:, d], lo[d], hi[d])), lo[d], hi[d])
                     for d in (0, 1)], 1)
    check("on-grid values round-trip exactly", np.abs(back - on).max() < 1e-9,
          f"max|e| {np.abs(back - on).max():.2e}")

    # 6 -- arbitrary values round-trip within half a step (the quantiser is sane)
    arb = np.stack([rng.uniform(0, 45, 5000), rng.uniform(-7, 7, 5000)], 1)
    err = np.stack([unnorm(decode(encode(arb[:, d], lo[d], hi[d])), lo[d], hi[d])
                    for d in (0, 1)], 1) - arb
    steps = [float(np.diff(grid_of(lo[d], hi[d])).mean()) for d in (0, 1)]
    within = all(np.abs(err[:, d]).max() <= steps[d] / 2 + 1e-9 for d in (0, 1))
    check("round-trip error <= half a step", within,
          "max|e| " + str(["%.4f" % np.abs(err[:, d]).max() for d in (0, 1)])
          + " vs step/2 " + str(["%.4f" % (s / 2) for s in steps]))

    # 7 -- and it must be LOOSER than that for a narrower grid: a tool that
    #      hardcoded a step would pass 6 and fail here
    mlo, mhi = np.array(MINI_FIT["min"]), np.array(MINI_FIT["max"])
    err_m = unnorm(decode(encode(arb[:, 0], mlo[0], mhi[0])), mlo[0], mhi[0]) - arb[:, 0]
    check("a narrower grid gives a smaller error",
          np.abs(err_m).max() < np.abs(err[:, 0]).max(),
          f"mini {np.abs(err_m).max():.4f} < trainval {np.abs(err[:, 0]).max():.4f}")

    # 8 -- saturation is detected rather than silently clamped
    far = np.array([[200.0, 0.0], [-50.0, 0.0], [10.0, 0.0]])
    check("out-of-range values are flagged as clipped",
          int((np.abs(norm_action(far, lo, hi)) >= 1.0).sum()) == 2,
          str(int((np.abs(norm_action(far, lo, hi)) >= 1.0).sum())) + " of 2 expected")

    # 9 -- encoded tokens stay inside the grammar eval_nuscenes.py enforces
    tok = np.stack([encode(arb[:, d], lo[d], hi[d]) for d in (0, 1)], 1)
    check("token ids inside the grammar range",
          tok.min() >= START_ID + 2 and tok.max() <= START_ID + 1 + N_BINS,
          f"{tok.min()}..{tok.max()} vs {START_ID+2}..{START_ID+1+N_BINS}")

    # 10 -- too few distinct values is a refusal, not a fabricated lattice
    flat = np.zeros((50, 2))
    check("refuses a degenerate axis", identify_grid(flat, cands, 2e-3)[3] is None)

    # 11 -- the 255-vs-256 slip that caused sec.1.16(f) in the first place
    span = hi[0] - lo[0]
    check("grid divisor is 255 (bin centres), not 256",
          abs(steps[0] - span / 255) < 1e-9 and abs(steps[0] - span / 256) > 1e-4,
          f"step {steps[0]:.6f}  span/255 {span/255:.6f}  span/256 {span/256:.6f}")

    n_ok = sum(ok for _, ok, _ in results)
    for name, ok, detail in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"   ({detail})" if detail else ""))
    print(f"\nself-test {n_ok}/{len(results)} " + ("PASS" if n_ok == len(results) else "FAIL"))
    return 0 if n_ok == len(results) else 1


def load_csv(path):
    pred, gt = [], []
    with open(path, encoding="utf-8") as f:
        for r in csvmod.DictReader(f):
            if r.get("status") != "ok":
                continue
            if r.get("pred"):
                pred.append(np.asarray(ast.literal_eval(r["pred"]), dtype=np.float64))
            if r.get("gt"):
                gt.append(np.asarray(ast.literal_eval(r["gt"]), dtype=np.float64))
    return (np.concatenate(pred, 0) if pred else None,
            np.concatenate(gt, 0) if gt else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="per_sample.csv with `pred` (and `gt`) columns")
    ap.add_argument("--selftest", action="store_true",
                    help="synthetic worlds with known answers; needs no data on disk")
    ap.add_argument("--norm", action="append", default=[],
                    help="candidate norm json; repeatable. The trainval fit and "
                         "data/nuscenes_records/nuscenes_norm.json are always included.")
    ap.add_argument("--tol", type=float, default=2e-3,
                    help="metres; a candidate matches if every axis step is within this")
    ap.add_argument("--repo_root", default=".")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(selftest())
    if not args.csv:
        ap.error("--csv is required (or pass --selftest)")

    check_source(args.repo_root)

    # ---- candidates -------------------------------------------------------
    cands = {"trainval fit (sec.1.15, constant in this file)": TRAINVAL_FIT}
    default_json = os.path.join(args.repo_root, "data/nuscenes_records/nuscenes_norm.json")
    for p in ([default_json] if os.path.exists(default_json) else []) + args.norm:
        with open(p, encoding="utf-8") as f:
            cands[p] = json.load(f)

    pred, gt = load_csv(args.csv)
    if pred is None:
        sys.exit(f"{args.csv} has no usable `pred` rows")
    n_dim = pred.shape[1]
    print(f"[csv] {args.csv}: {pred.shape[0]} waypoints, action_dim={n_dim}")

    # ---- the observed lattice, and which candidate produces it -------------
    obs, resid, rows, winner = identify_grid(pred, cands, args.tol)
    if obs is None:
        sys.exit("an axis has fewer than 3 distinct predicted values -- cannot fit a lattice")
    print(f"[observed] step per axis = {['%.6f' % s for s in obs]}  "
          f"max|resid| = {['%.4f' % r for r in resid]} (csv `pred` is rounded to 1e-3)")

    print(f"\n{'candidate norm range':<52} {'derived step per axis':<30} match")
    for name, steps, ok in rows:
        print(f"{name[:52]:<52} {str(['%.6f' % s for s in steps]):<30} {'YES' if ok else 'no'}")

    if winner is None:
        print("\nGATE FAIL: no candidate norm reproduces the observed lattice.")
        print("  The grid this run used is unknown, so no quantisation figure derived")
        print("  from it is trustworthy. Find the norm json the run was launched with")
        print("  (it is echoed in the eval log: '[eval] waypoint un-norm range: ...').")
        sys.exit(1)

    name, lo, hi, steps = winner
    print(f"\n[grid] identified: {name}")
    print(f"       min={lo.tolist()} max={hi.tolist()}")
    for d in range(n_dim):
        span = hi[d] - lo[d]
        print(f"       axis {d}: span {span:.4f} / {len(CENTERS)} levels = {steps[d]:.6f} m"
              f"   (normalised step {2/(N_BINS-1):.6f}, shared by every axis)")

    # ---- the quantisation error, measured rather than assumed --------------
    if gt is None:
        print("\n(no `gt` column -- skipping the round-trip error)")
        return
    tok = np.stack([encode(gt[:, d], lo[d], hi[d]) for d in range(n_dim)], 1)
    back = np.stack([unnorm(decode(tok[:, d]), lo[d], hi[d]) for d in range(n_dim)], 1)
    err = back - gt
    n_clipped = int((np.abs(norm_action(gt, lo, hi)) >= 1.0).sum())

    print(f"\n[round trip] {gt.shape[0]} real GT waypoints, encode -> token -> decode -> metres")
    print(f"       token ids {tok.min()}..{tok.max()}  (the grammar in eval_nuscenes.py "
          f"allows {START_ID+2}..{START_ID+1+N_BINS})")
    print(f"       clipped at the norm range: {n_clipped} / {gt.size}"
          + ("   <- saturating; the range is too narrow for this data" if n_clipped else ""))
    for d in range(n_dim):
        e = err[:, d]
        print(f"       axis {d}: RMS {np.sqrt((e**2).mean()):.6f}  max|e| {np.abs(e).max():.6f}"
              f"   (uniform-quantisation theory step/sqrt12 = {steps[d]/np.sqrt(12):.6f})")
    disp = np.linalg.norm(err, axis=1)
    print(f"\n       pooled per-coordinate RMS : {np.sqrt((err**2).mean()):.6f} m")
    print(f"       2-D displacement RMS      : {np.sqrt((disp**2).mean()):.6f} m  "
          f"<- the one commensurable with avgL2@3s")
    print(f"       2-D displacement mean     : {disp.mean():.6f} m")
    print("\n  Read against the effect sizes it has to be small compared to: the GoT-vs-greedy")
    print("  gap is +0.0397 m (sec.1 claim 1) and avgL2@3s itself is 3.5557 m. The grid is")
    print("  shared by every arm, so it cancels in paired comparisons -- it bounds what an")
    print("  arm could be worth in ABSOLUTE terms, not the resolution of a paired test.")


if __name__ == "__main__":
    main()
