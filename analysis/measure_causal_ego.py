"""Ceiling for HONEST (causal) ego status, before spending the 33 h retrain.

§1.10(c) rescaled each trajectory's longitudinal profile so its first step matched
the **GT** first step and got `avgL2@3s` 3.5557 -> 1.44.  That number is an upper
bound that leaks the future: the GT first step is 0.5 s of motion that has not
happened yet, so it already contains that interval's accel.  A model fed real ego
status knows only the past.  This tool measures the two halves of the gap:

  Stage A (`--records`)  how well does causal v0 (the PREVIOUS keyframe's step)
                         predict the GT first step?  -> sigma, in metres/0.5 s
  Stage B (`--per_sample`) redo the §1.10(c) rescale driven by `GT + N(0, sigma)`
                         instead of GT, sweeping sigma so the verdict does not
                         rest on one calibration.

Both stages are GPU 0 and import no torch.

★Stage A needs a records json whose keyframes are CONSECUTIVE (0.5 s apart) --
that is `nuscenes_v1.0-trainval_val.json`, NOT `nuscenes_val_scenespread.json`
(the eval set keeps 4 records per scene ~5.5 s apart, so it has no predecessors).
Stage B is the other way round: it joins the eval set by `sample_token`.

Usage
-----
  python analysis/measure_causal_ego.py \
      --records      ./data/nuscenes_records/nuscenes_v1.0-trainval_val.json \
      --per_sample   ./results/headline/ref_basepred/per_sample.csv \
      --eval_records ./data/nuscenes_records/nuscenes_val_scenespread.json
"""
import argparse
import ast
import csv
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_got_csv import cluster_bootstrap_ci  # noqa: E402

TS_RE = re.compile(r"__CAM_FRONT__(\d+)\.jpg")
GREEDY_REF = 3.5557  # S7.2, seed 42, 600 records / 150 scenes


def _timestamp(rec):
    m = TS_RE.search(rec["images"][0])
    if m is None:
        raise SystemExit(f"cannot parse timestamp from {rec['images'][0]!r}")
    return int(m.group(1))


# --------------------------------------------------------------------------- A
def stage_a(records_path):
    """Skill of causal v0 (previous keyframe's displacement) at the GT first step."""
    recs = json.load(open(records_path))
    by_scene = defaultdict(list)
    for r in recs:
        by_scene[r["scene"]].append(r)
    for s in by_scene:
        by_scene[s].sort(key=_timestamp)

    pairs = defaultdict(list)  # scene -> [(causal_lon, target_lon), ...]
    for scene, seq in by_scene.items():
        for a, b in zip(seq, seq[1:]):
            dt = (_timestamp(b) - _timestamp(a)) / 1e6
            if not 0.45 < dt < 0.55:
                continue  # not adjacent keyframes
            pairs[scene].append((a["waypoints"][0][0], b["waypoints"][0][0]))

    n = sum(len(v) for v in pairs.values())
    print(f"[A] {os.path.basename(records_path)}: {n} adjacent pairs / {len(recs)} "
          f"records, {len(pairs)} scenes")
    if n == 0:
        print("[A] [warn] no adjacent keyframes -- wrong records json (see docstring). SKIP")
        return None
    if len(pairs) < 10:
        print(f"[A] [warn] only {len(pairs)} scenes. S9: this is the `--limit N` trap, "
              f"not a sample. Treat as a wiring check, NOT a result.")

    causal = np.array([c for v in pairs.values() for c, _ in v])
    target = np.array([t for v in pairs.values() for _, t in v])
    err = np.abs(causal - target)
    mae_mean = np.abs(target - target.mean()).mean()

    print(f"[A] GT first-step lon : mean {target.mean():.4f}  std {target.std():.4f} m/0.5s")
    print(f"[A] |causal - GT| lon : MAE {err.mean():.4f}  RMSE "
          f"{np.sqrt(((causal-target)**2).mean()):.4f}")
    print(f"[A] corr(causal, GT)  : {np.corrcoef(causal, target)[0, 1]:.4f}")
    print(f"[A] skill vs mean-predictor: {1 - err.mean()/mae_mean:+.1%}")

    clusters = {s: [abs(c - t) for c, t in v] for s, v in pairs.items()}
    lo, hi = cluster_bootstrap_ci(clusters)
    print(f"[A] MAE ci_sc [{lo:.4f}, {hi:.4f}]")
    print(f"[A] reference: |pool median - GT| first step = 1.0015 m (S1.10a)")
    return float(np.sqrt(((causal - target) ** 2).mean()))


# --------------------------------------------------------------------------- B
def _avg_l2(pred, gt):
    """ST-P3 convention (S7.3): mean of the per-step L2 over steps 0..5."""
    return np.linalg.norm(pred - gt, axis=2).mean(axis=1)


def stage_b(per_sample, eval_records, sigmas, pred_col, n_rep, seed):
    gt_by_tok = {r["sample_token"]: np.array(r["waypoints"], float)
                 for r in json.load(open(eval_records))}

    P, G, S = [], [], []
    with open(per_sample) as fh:
        rows = list(csv.DictReader(fh))
    if pred_col not in rows[0]:
        raise SystemExit(f"{per_sample} has no `{pred_col}` column -- rerun the arm "
                         f"with base-pred logging (results/headline/ref_basepred has it)")
    for r in rows:
        g = gt_by_tok.get(r["sample_token"])
        if g is None or not r[pred_col]:
            continue
        p = np.array(ast.literal_eval(r[pred_col]), float)
        if p.shape != (6, 2) or g.shape != (6, 2):
            continue
        P.append(p); G.append(g); S.append(r["scene"])
    P, G, S = np.array(P), np.array(G), np.array(S)
    print(f"\n[B] {len(P)} records / {len(set(S))} scenes  (pred column `{pred_col}`)")

    base = _avg_l2(P, G)
    ok = abs(base.mean() - GREEDY_REF) < 0.01
    print(f"[B] wiring check: unmodified avgL2@3s = {base.mean():.4f} vs "
          f"S7.2 {GREEDY_REF} -> {'PASS' if ok else '*FAIL -- do not read further'}")
    if not ok:
        return

    # S1.10(c) excluded parked records: rescaling a ~zero first step explodes.
    stationary = np.abs(G[:, -1, 0]) < 0.5
    denom = P[:, 0, 0]
    usable = (~stationary) & (np.abs(denom) > 1e-3)
    print(f"[B] stationary/degenerate excluded from rescale: {(~usable).sum()}")

    def rescale(vref):
        out = P.copy()
        k = np.where(usable, vref / np.where(np.abs(denom) > 1e-3, denom, 1.0), 1.0)
        out[:, :, 0] = P[:, :, 0] * np.clip(k, -10, 10)[:, None]
        return out

    rng = np.random.default_rng(seed)
    print(f"\n[B] {'first-step info driving the rescale':<38}"
          f"{'avgL2@3s':>10}{'vs greedy':>11}{'ci_sc':>22}")
    for sigma in sigmas:
        reps = 1 if sigma == 0 else n_rep
        means, last_diff = [], None
        for _ in range(reps):
            vref = G[:, 0, 0] + rng.normal(0, sigma, len(G))
            vals = _avg_l2(rescale(vref), G)
            means.append(vals.mean())
            last_diff = vals - base
        clusters = defaultdict(list)
        for s, d in zip(S, last_diff):
            clusters[s].append(float(d))
        lo, hi = cluster_bootstrap_ci(clusters)
        m = float(np.mean(means))
        tag = "GT oracle (=S1.10c, LEAKS FUTURE)" if sigma == 0 else f"causal, sigma={sigma:.2f} m"
        print(f"    {tag:<38}{m:>10.4f}{m - base.mean():>+11.4f}"
              f"{f'[{lo:+.4f}, {hi:+.4f}]':>22}")

    print("\n[B] [warn] read this as the GENERATOR ceiling only. S1.10(b) already measured "
          "that the same information given to the SELECTOR is +0.0205 (ns) -- ego "
          "status does not make GoT beat greedy, it lowers absolute L2.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", default=None,
                    help="records json with CONSECUTIVE keyframes (trainval val)")
    ap.add_argument("--per_sample", default=None)
    ap.add_argument("--eval_records", default="./data/nuscenes_records/nuscenes_val_scenespread.json")
    ap.add_argument("--pred_col", default="base_pred")
    ap.add_argument("--sigmas", nargs="+", type=float, default=None,
                    help="noise levels for stage B; default sweeps around stage A's RMSE")
    ap.add_argument("--n_rep", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    sigma_hat = stage_a(args.records) if args.records else None
    if args.per_sample:
        sigmas = args.sigmas
        if sigmas is None:
            s = sigma_hat if sigma_hat else 0.25
            sigmas = [0.0, s, 2 * s, 4 * s]
        stage_b(args.per_sample, args.eval_records, sigmas,
                args.pred_col, args.n_rep, args.seed)


if __name__ == "__main__":
    main()
