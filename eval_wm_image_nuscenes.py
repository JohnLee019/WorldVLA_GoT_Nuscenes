"""
Offline WORLD-MODEL image evaluation for nuScenes GoT planning (PROJECT_HANDOFF
§7.4 / §2③). The auxiliary, image-space counterpart of the L2 waypoint metric:
does the trajectory the planner commits lead to a future the world model renders
close to what nuScenes actually recorded?

Pipeline per val record
------------------------
    base model --GoT plan--> 6 waypoints (t0 ego frame)          [the trajectory]
    GT                      = record["waypoints"] (t0 ego frame) [the reference]

    for each segment s (teacher-forced -- §2③②):
        real_anchor = real CAM_FRONT frame at the START of segment s
        real_target = real CAM_FRONT frame at the END   of segment s
        pred_plan   = WM(real_anchor, plan_action_s)     # 1 s-ahead prediction
        pred_gt     = WM(real_anchor, gt_action_s)        # WM's own floor
        d_plan[s]   = frame_distance(pred_plan, real_target)
        d_gt[s]     = frame_distance(pred_gt,   real_target)

Teacher forcing (feeding the REAL frame at every segment start) stops per-segment
WM errors from compounding, so each d[s] is a clean per-step number.

Session 15: the rank-correlation gate was rewired from d_plan to DELTA and
renamed `spearman_delta_vs_L2`. A summary.json carrying the old key
`spearman_d_vs_L2` was produced by the old definition, whose PASS came from the
reconstruction floor and scene difficulty being shared with L2 (rho +0.4217)
rather than from the WM ranking anything (rho -0.0091 on delta, sec.11.9).
Do not compare the two keys' verdicts.

Reading the numbers (§2 "background cancels")
--------------------------------------------
The WM is lossy, so BOTH d_plan and d_gt carry a large, trajectory-independent
reconstruction floor. The signal is the RELATIVE quantity:

    delta = d_plan - d_gt

d_gt is the best the WM can do (it is fed the TRUE action, whose future is the
real frame), so delta isolates the plan's contribution: delta ~ 0 means the plan
leads to a future as plausible as the ground truth's; delta > 0 means the plan
steers somewhere the WM finds less like reality. Absolute d_plan / d_gt are
reported too, but delta is the headline. This is a *secondary* metric -- L2 stays
primary (see eval_got_nuscenes.py).

Actions: both the plan's and the GT's per-segment actions are re-based into each
segment's local ego frame with the SAME helper (seg_local_actions), matching the
WM's training convention ("k waypoints in the current frame's ego coords"), so
plan and GT are scored under an identical convention.

Two GPUs recommended: base model on --device, WM on --wm_device (2x7B > 24 GB on
one card). See PROJECT_HANDOFF §2③ (base=GPU0 / WM=GPU1).

Status: NOT YET RUN end-to-end -- needs a world model trained by
train_nuscenes_wm.py (mask_image_logits=False); the old WM was discarded and a
retrain is pending. The metric math is unit-tested (got_drive/wm_image_metric.py
self-test); this script wires it to the real models and data.

Usage
-----
    python eval_wm_image_nuscenes.py \
        --resume_path ./output/nuscenes_trainval_full_r256/epoch4 \
        --wm_path     ./output/nuscenes_wm/epochN \
        --tokenizer_path ../ckpts/Lumina-mGPT-7B-768 \
        --records_json  ./data/nuscenes_records/nuscenes_v1.0-trainval_val.json \
        --wm_eval_json  ./data/nuscenes_wm_records/nuscenes_wm_eval_v1.0-trainval_val.json \
        --norm_path     ./data/nuscenes_records/nuscenes_norm_v1.0-trainval.json \
        --output_dir ./results/wm_image --device 0 --wm_device 1 --limit 200
"""

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

# Model/VQGAN imports are deliberately deferred into the functions that need them
# (see main / get_args). The gate arithmetic below is the part most likely to be
# wrong, and it must be checkable on a laptop with no torch:
#     python eval_wm_image_nuscenes.py --selftest
# A module-level `import torch` would make that impossible.

METRIC_KEYS = ("codebook_l2", "codebook_cosine", "pixel_mae")


def get_args():
    from data.dataset_nuscenes import DEFAULT_PROMPT
    p = argparse.ArgumentParser("nuScenes offline WM-image eval")
    # --- base model (shared with eval_nuscenes) ---
    p.add_argument("--resume_path", default=None,
                   help="trained base checkpoint dir. Required ONLY when 'plan' is among "
                        "--arms; the §7.5.1 gate run (gt/mirror/copy) never plans, so it can "
                        "and should be omitted there.")
    p.add_argument("--tokenizer_path", default=None,
                   help="not needed with --summarize_only")
    p.add_argument("--records_json", default=None,
                   help="planning val records (preprocess_nuscenes.py). Not needed with --summarize_only.")
    p.add_argument("--wm_eval_json", default=None,
                   help="future-frame records (preprocess_nuscenes_wm_eval.py)")
    p.add_argument("--norm_path", default=None, help="nuscenes_norm.json used at TRAINING time")
    p.add_argument("--output_dir", default="./results/wm_image")
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--max_seq_len", type=int, default=4096)
    p.add_argument("--action_dim", type=int, default=2)
    p.add_argument("--time_horizon", type=int, default=6)
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--device", type=int, default=0, help="GPU for the base (planning) model")
    p.add_argument("--load_in_4bit", action="store_true", default=False)
    p.add_argument("--limit", type=int, default=0, help="evaluate only N joined records after --offset (0 = all)")
    p.add_argument("--stride", type=int, default=1,
                   help="keep every Nth joined record (applied after --offset, before "
                        "--limit). Records are grouped by scene on disk, so --limit alone "
                        "draws one scene: on NAVSIM navtest that is ~510 consecutive "
                        "records from a single log, and every CI here is clustered by "
                        "scene. --stride 2000 --limit 30 spreads 30 records over ~30 logs.")
    p.add_argument("--offset", type=int, default=0,
                   help="skip the first N joined records (the val split opens on a stationary "
                        "scene-0003 run, so a small --limit alone only sees parked frames)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--legacy_random_crop", action="store_true", default=False,
                   help="reproduce pre-fix behaviour: random-crop on every forward call, and "
                        "compare against a target that wm_image_metric downsamples with a "
                        "single BICUBIC resize instead of the model's progressive-BOX chain. "
                        "Both put pred and target on different geometry -- measured 10.69 "
                        "codebook_l2 between a frame and ITSELF. See got_drive/eval_crop.py.")
    # --- world model ---
    p.add_argument("--wm_path", default=None,
                   help="world-model checkpoint dir (train_nuscenes_wm.py, mask_image_logits=False)")
    p.add_argument("--wm_device", type=int, default=None,
                   help="GPU for the world model (default: same as --device). Use a SEPARATE card "
                        "from the base model to avoid 2x7B (28GB) on one 24GB GPU.")
    p.add_argument("--wm_max_new_tokens", type=int, default=700,
                   help="token budget for one WM future-frame generation")
    # --- GoT config (must match how the plan under test is produced) ---
    p.add_argument("--n_segments", type=int, default=3)
    p.add_argument("--segment_len", type=int, default=2)
    p.add_argument("--k_candidates", type=int, default=4)
    p.add_argument("--beam_width", type=int, default=2)
    p.add_argument("--temperatures", type=float, nargs="+", default=[1.0, 1.2, 1.4, 1.6])
    p.add_argument("--w_kinematic", type=float, default=1.0)
    p.add_argument("--w_command", type=float, default=1.0)
    p.add_argument("--verbose_plan", action="store_true", default=False)
    # --- validity gates (PROJECT_HANDOFF §7.5.1) ---
    p.add_argument("--arms", nargs="+", default=["gt", "plan"],
                   choices=["gt", "plan", "mirror", "copy"],
                   help="which action to feed the WM. 'gt' is the reference every other arm "
                        "is paired against and is always included. 'plan' = the GoT plan (the "
                        "only arm that needs the base model -- omit it and the base model is "
                        "NOT loaded, which is what the §7.5.1 gate run wants). 'mirror' = GT "
                        "with the lateral component negated (the sensitivity gate). 'copy' = "
                        "no WM call at all, the anchor frame asserted to be the future "
                        "(d_copy, the floor d_gt must beat).")
    p.add_argument("--eps", type=float, nargs="*", default=[],
                   help="lateral-offset arms in metres, e.g. --eps 0.5 1 2 3. Builds the "
                        "delta->'effective metres' conversion curve. ★Stay at or below ~3 m: "
                        "the measured displacement-resolution curve peaks at 3-4 m and DECLINES "
                        "beyond it, so a 4 m arm tests monotonicity in the region where the "
                        "metric is not monotonic (session 7).")
    p.add_argument("--plan_error_flag_m", type=float, default=4.0,
                   help="report the fraction of segments whose plan-vs-GT endpoint error "
                        "exceeds this. Above it the metric is non-monotonic, so a WORSE plan "
                        "can score a LOWER distance and delta flips sign.")
    p.add_argument("--wm_do_sample", action="store_true",
                   help="E4: sample the WM's image tokens instead of greedy decoding. "
                        "GREEDY IS THE DEFAULT and matches upstream RynnVLA-002 -- every "
                        "number in sec.1 was produced with it, so a run with this flag is "
                        "NOT comparable to them. Sampling can add high-frequency detail "
                        "that makes codebook_l2 worse, so adopting it means re-measuring "
                        "delta and all three gates (~6 GPU-h). Use it with a small --limit "
                        "and --save_frames for the qualitative figure only.")
    p.add_argument("--wm_temperature", type=float, default=1.0,
                   help="only meaningful with --wm_do_sample")
    p.add_argument("--wm_top_k", type=int, default=None,
                   help="only meaningful with --wm_do_sample")
    p.add_argument("--n_boot", type=int, default=10000,
                   help="record-clustered bootstrap resamples for the gate CIs")
    p.add_argument("--plans_csv", default=None, metavar="CSV",
                   help="take the 'plan' arm's trajectories from an existing per_sample.csv "
                        "(column got_pred, joined on sample_token) instead of re-planning. "
                        "The base model is then NOT loaded, so the run needs one card "
                        "(WM only, ~14 GB) instead of two -- 'plan' normally forces base 7B + "
                        "WM 7B = 28 GB. Use results/headline/ref/per_sample.csv to render the "
                        "exact seed-42 trajectories the reported numbers came from; anything "
                        "else and the frames illustrate a plan no table describes. Records "
                        "absent from the csv are skipped for the plan arm only.")
    p.add_argument("--save_frames", default=None, metavar="DIR",
                   help="write the frames instead of only their distances: per record and "
                        "segment, real_<s>.png (the actual future), <arm>_<s>.png (what the WM "
                        "drew for that arm) and anchor_<s>.png (the frame it was conditioned "
                        "on). This is the qualitative figure -- the WM is a detector, not a "
                        "ruler (rho(delta,L2) = -0.009), so the picture carries what the "
                        "number cannot. Use a small --limit: a full run writes thousands.")
    p.add_argument("--save_frames_max", type=int, default=20, metavar="N",
                   help="stop writing frames after N records (the distances keep being "
                        "computed for every record). Guards against filling the disk when "
                        "--save_frames is combined with a large --limit.")
    p.add_argument("--summarize_only", nargs="+", default=None, metavar="CSV",
                   help="skip all inference: rebuild the summary and gates from one or more "
                        "existing per_sample.csv files. This is how a sharded run is merged -- "
                        "shard by --offset/--limit across GPUs, then pass the shards' CSVs "
                        "here. Loads no model, so it needs no --wm_path/--tokenizer_path.")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────
# per-segment action re-basing (shared by plan and GT -> identical convention)
# ──────────────────────────────────────────────────────────────────────────

def _rot(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def seg_local_actions(traj, segment_len, n_segments):
    """(T,2) trajectory in the t0 ego frame -> list of (segment_len,2) actions,
    each re-expressed in ITS OWN segment-start ego frame.

    This is the inverse of got_pipeline_drive.map_local_to_original: a segment's
    anchor is the previous segment's end position, its heading the direction of
    travel into that point (matching advance_pose). Segment 0 anchors at the
    origin with heading 0 (so its local action == its t0-frame waypoints, exactly
    as Mode A generates it). The result matches the WM's action convention: k
    waypoints in the current (anchor) frame's ego coordinates.
    """
    traj = np.asarray(traj, dtype=np.float64)
    out = []
    for s in range(n_segments):
        seg = traj[s * segment_len:(s + 1) * segment_len]        # (segment_len, 2), t0 frame
        if s == 0:
            base_p, base_theta = np.zeros(2), 0.0
        else:
            end = traj[s * segment_len - 1]
            prev = traj[s * segment_len - 2] if s * segment_len - 2 >= 0 else np.zeros(2)
            d = end - prev
            base_p, base_theta = end, float(np.arctan2(d[1], d[0]))
        out.append((seg - base_p) @ _rot(base_theta))            # -> segment-start local frame
    return out


# ──────────────────────────────────────────────────────────────────────────
# aggregation
# ──────────────────────────────────────────────────────────────────────────

def arm_action(arm, s, gt_actions, plan_actions):
    """The action fed to the WM for `arm` on segment `s`, or None to skip.

    Every arm is a transform of the SAME GT action in the segment's own ego frame,
    so `gt` and its variants differ only in the quantity under test.
    """
    if arm == "gt":
        return gt_actions[s]
    if arm == "plan":
        return None if plan_actions is None else plan_actions[s]
    if arm == "mirror":
        # negate the lateral component: same speed profile, opposite steering. If
        # the WM ignores its action input this scores exactly like gt, which is
        # precisely what the sensitivity gate is built to detect.
        a = np.array(gt_actions[s], dtype=np.float64, copy=True)
        a[:, 1] = -a[:, 1]
        return a
    if arm.startswith("eps"):
        a = np.array(gt_actions[s], dtype=np.float64, copy=True)
        a[:, 1] = a[:, 1] + float(arm[3:])
        return a
    raise ValueError(f"unknown arm {arm!r}")


def _mean_or_none(xs):
    xs = [x for x in xs if x is not None]
    return round(float(np.mean(xs)), 5) if xs else None


def _ranks(xs):
    """Average ranks, ties shared -- the Spearman prerequisite. No scipy."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def _spearman(a, b):
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None
             and np.isfinite(x) and np.isfinite(y)]
    if len(pairs) < 3:
        return None
    ra, rb = _ranks([p[0] for p in pairs]), _ranks([p[1] for p in pairs])
    ra, rb = np.asarray(ra), np.asarray(rb)
    sa, sb = ra.std(), rb.std()
    if sa == 0 or sb == 0:
        return None
    return round(float(((ra - ra.mean()) * (rb - rb.mean())).mean() / (sa * sb)), 4)


def _cluster_boot_ci(triples, n_boot, seed=0):
    """95% CI on the mean of `triples` = [(record_idx, segment, value), ...],
    resampling RECORDS, not segments.

    Segments inside one record share an anchor frame and a scene, so treating
    them as independent would understate the interval -- the same clustering
    argument §9 makes for scenes in the planning metrics.
    """
    if not triples:
        return None
    by_rec = {}
    for rec, _s, v in triples:
        by_rec.setdefault(rec, []).append(v)
    keys = list(by_rec)
    if len(keys) < 3:
        return None
    rng = np.random.default_rng(seed)
    groups = [np.asarray(by_rec[k], dtype=np.float64) for k in keys]
    means = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        means[b] = np.concatenate([groups[p] for p in pick]).mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return [round(float(lo), 5), round(float(hi), 5)]


def _paired_block(triples, seg_horizons, n_seg, n_boot):
    flat = [v for _r, _s, v in triples]
    return {
        "overall": _mean_or_none(flat),
        "n_pairs": len(flat),
        "n_records": len({r for r, _s, _v in triples}),
        "ci95_record_clustered": _cluster_boot_ci(triples, n_boot),
        "per_horizon": {seg_horizons[s]: _mean_or_none([v for _r, ss, v in triples if ss == s])
                        for s in range(n_seg)},
    }


def _summary_from_csvs(paths, n_seg, seg_horizons, n_boot, plan_error_flag_m):
    """Rebuild the full summary + gates from per_sample.csv files, no inference.

    This is what makes sharding safe: split the records across GPUs with
    --offset/--limit, then merge here. Records are keyed by sample_token, so a
    shard boundary cannot silently duplicate or drop one, and the record index
    used for the clustered bootstrap is assigned over the merged set.
    """
    rows, seen = [], set()
    for p in paths:
        with open(p, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                tok = r.get("sample_token")
                if tok in seen:
                    continue                     # overlapping shards: keep the first
                seen.add(tok)
                rows.append(r)

    def val(r, key):
        v = r.get(key)
        if v in (None, ""):
            return None
        try:
            return float(v)
        except ValueError:
            return None

    # arms are whatever produced a distance column; delta_* columns are derived
    arms = []
    for r in rows:
        for k in r:
            if k.startswith("delta_"):
                continue
            for mk in METRIC_KEYS:
                for s in range(n_seg):
                    suf = f"_{mk}_s{s}"
                    if k.endswith(suf):
                        a = k[: -len(suf)]
                        if a not in arms:
                            arms.append(a)
    arms = ["gt"] + [a for a in arms if a != "gt"]

    bucket = {a: {mk: [[] for _ in range(n_seg)] for mk in METRIC_KEYS} for a in arms}
    paired = {a: {mk: [] for mk in METRIC_KEYS} for a in arms}
    attempts = {a: [0, 0] for a in arms}
    spearman_pairs = {"delta": [], "L2": []}
    plan_seg_errors = []

    for i, r in enumerate(rows):
        for s in range(n_seg):
            e = val(r, f"plan_seg_err_s{s}")
            if e is not None:
                plan_seg_errors.append(e)
            for a in arms:
                # "attempted" is inferred: a planned arm is attempted on every
                # segment of every record where the plan existed. Exact for
                # gt/mirror/copy/eps; for `plan` it follows plan_ok.
                if a == "plan" and str(r.get("plan_ok", "")).lower() not in ("true", "1"):
                    continue
                attempts[a][1] += 1
                if val(r, f"{a}_codebook_l2_s{s}") is not None:
                    attempts[a][0] += 1
                for mk in METRIC_KEYS:
                    v = val(r, f"{a}_{mk}_s{s}")
                    if v is None:
                        continue
                    bucket[a][mk][s].append(v)
                    ref = val(r, f"gt_{mk}_s{s}")
                    if a != "gt" and ref is not None:
                        paired[a][mk].append((i, s, v - ref))
        dp = [val(r, f"delta_plan_codebook_l2_s{s}") for s in range(n_seg)]
        dp = [x for x in dp if x is not None]
        l2 = val(r, "plan_avgL2")
        if dp and l2 is not None:
            spearman_pairs["delta"].append(float(np.mean(dp)))
            spearman_pairs["L2"].append(l2)

    out = summarize(bucket, paired, arms, seg_horizons, n_seg, attempts, n_boot,
                    spearman_pairs, plan_seg_errors, plan_error_flag_m)
    out["merged_from"] = list(paths)
    out["n_records_merged"] = len(rows)
    return out


def summarize(bucket, paired, arms, seg_horizons, n_seg, attempts, n_boot,
              spearman_pairs, plan_seg_errors, plan_error_flag_m):
    """bucket[arm][metric] -> per-segment lists of absolute distances.
    paired[arm][metric]   -> [(record_idx, segment, d_arm - d_gt), ...] where BOTH
                             that arm and `gt` produced a well-formed frame.

    `gt` is the reference: every other arm is reported as a paired delta against
    it, because the WM's reconstruction floor is trajectory-independent and only
    cancels in the difference (§2 "background cancels").
    """
    out = {"arms": arms}

    for arm in arms:
        out[arm] = {"success_rate": (round(attempts[arm][0] / attempts[arm][1], 4)
                                     if attempts[arm][1] else None),
                    "n_ok": attempts[arm][0], "n_attempted": attempts[arm][1]}
        for mk in METRIC_KEYS:
            per_seg = bucket[arm][mk]
            out[arm][mk] = {
                "overall": _mean_or_none([v for seg in per_seg for v in seg]),
                "per_horizon": {seg_horizons[s]: _mean_or_none(per_seg[s])
                                for s in range(n_seg)},
            }

    # ---- paired deltas vs gt (the only quantity that is interpretable) ----
    out["delta_vs_gt_paired"] = {}
    for arm in arms:
        if arm == "gt":
            continue
        out["delta_vs_gt_paired"][arm] = {
            mk: _paired_block(paired[arm][mk], seg_horizons, n_seg, n_boot)
            for mk in METRIC_KEYS}

    # ---- Layer 2 gates (§7.5.1) ----
    gates = {}
    prim = "codebook_l2"

    if "copy" in arms:
        blk = out["delta_vs_gt_paired"]["copy"][prim]      # d_copy - d_gt
        ci = blk["ci95_record_clustered"]
        gates["wm_gain_over_copy"] = {
            **blk,
            "verdict": ("PASS" if (ci and ci[0] > 0) else
                        "FAIL" if (ci and ci[1] < 0) else "INCONCLUSIVE"),
            "reading": ("d_copy - d_gt. POSITIVE and CI clear of 0 => the WM beats asserting "
                        "the scene is unchanged. <=0 => it adds nothing and nothing downstream "
                        "of it can be believed."),
        }

    if "mirror" in arms:
        blk = out["delta_vs_gt_paired"]["mirror"][prim]    # d_mirror - d_gt
        ci = blk["ci95_record_clustered"]
        gates["sensitivity_mirror"] = {
            **blk,
            "verdict": ("PASS" if (ci and ci[0] > 0) else
                        "FAIL" if (ci and ci[1] < 0) else "INCONCLUSIVE"),
            "reading": ("d_mirror - d_gt with the GT action's lateral component negated. "
                        "POSITIVE and CI clear of 0 => the WM actually responds to the ACTION. "
                        "~0 => it renders the same future whatever it is told, so delta ~ 0 "
                        "would mean 'measures nothing', not 'the plan is good'."),
        }

    if spearman_pairs["delta"]:
        rho = _spearman(spearman_pairs["delta"], spearman_pairs["L2"])
        # renamed on purpose: a summary.json carrying the OLD key was
        # produced by the OLD (d_plan) definition and its PASS is void.
        gates["spearman_delta_vs_L2"] = {
            "rho": rho, "n_records": len(spearman_pairs["delta"]),
            "verdict": ("FAIL_uninformative" if rho is None or abs(rho) < 0.1 else
                        "FAIL_redundant" if abs(rho) > 0.9 else "PASS"),
            "reading": ("rank correlation between DELTA (d_plan - d_gt) and the plan's L2 "
                        "error. Near 0 => the WM sees nothing L2 sees. Near 1 => it is just a "
                        "noisy re-derivation of L2 and adds no independent evidence. The "
                        "useful outcome is INTERMEDIATE."),
        }

    eps_arms = [a for a in arms if a.startswith("eps")]
    if eps_arms:
        curve = {a: out["delta_vs_gt_paired"][a][prim]["overall"] for a in eps_arms}
        vals = [curve[a] for a in eps_arms if curve[a] is not None]
        gates["eps_curve"] = {
            "curve": curve,
            "monotonic": (all(b >= a for a, b in zip(vals, vals[1:])) if len(vals) > 1 else None),
            "reading": ("delta for a GT action displaced laterally by eps metres -- the "
                        "conversion from a delta value to 'effective metres of error'. Must "
                        "increase with eps. ★Only valid for eps <= ~3 m: the measured "
                        "displacement-resolution curve peaks at 3-4 m and declines after."),
        }

    if plan_seg_errors:
        arr = np.asarray(plan_seg_errors, dtype=np.float64)
        gates[f"frac_seg_plan_error_gt_{plan_error_flag_m:g}m"] = {
            "value": round(float((arr > plan_error_flag_m).mean()), 4),
            "n_segments": int(arr.size),
            "median_seg_error_m": round(float(np.median(arr)), 4),
            "reading": ("fraction of segments whose plan-vs-GT endpoint displacement lands in "
                        "the metric's NON-MONOTONIC region, where a worse plan can score a "
                        "lower distance. Large => report delta split by this threshold."),
        }
    out["gates"] = gates
    return out


def _load_plans_csv(path):
    """{sample_token: (T,2) trajectory} from an eval_got_nuscenes per_sample.csv.

    Reads `got_pred`, which eval_got writes for every record whose plan came back
    well-formed. Rows are kept only when got_status == 'ok' (a malformed plan has
    no trajectory to render) and the first row wins, so a multi-seed csv resolves
    to its first seed rather than silently mixing seeds across records.
    """
    plans, n_rows, n_bad = {}, 0, 0
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            n_rows += 1
            if r.get("got_status") not in (None, "", "ok"):
                continue
            tok, raw = r.get("sample_token"), r.get("got_pred") or ""
            if not tok or not raw or tok in plans:
                continue
            try:
                arr = np.asarray(json.loads(raw), dtype=np.float64)
            except (ValueError, TypeError):
                n_bad += 1
                continue
            if arr.ndim == 2 and arr.size:
                plans[tok] = arr
    if not plans:
        raise SystemExit(
            f"[fatal] --plans_csv {path} yielded no usable trajectories from {n_rows} rows. "
            f"It needs a got_pred column, which eval_got_nuscenes.py writes; a csv from "
            f"another script will not have one.")
    if n_bad:
        print(f"[wm-image] --plans_csv: {n_bad} rows had an unparsable got_pred")
    return plans


def main():
    args = get_args()

    # ---- merge-only path: no models, no GPU, no data files ----
    # Placed BEFORE the heavy imports so a merge never touches torch/VQGAN.
    if args.summarize_only:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        seg_horizons = {s: f"{(s + 1) * args.segment_len * 0.5:.1f}s"
                        for s in range(args.n_segments)}
        summary = _summary_from_csvs(args.summarize_only, args.n_segments, seg_horizons,
                                     args.n_boot, args.plan_error_flag_m)
        with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(json.dumps(summary, indent=2))
        print(f"\nmerged {summary['n_records_merged']} records -> "
              f"{os.path.join(args.output_dir, 'summary.json')}")
        return

    from data.item_processor import FlexARItemProcessor_Action_NuScenes
    from got_drive.eval_crop import crop_for_eval
    from got_drive.got_pipeline_drive import (DriveGoTConfig, DriveGoTPipeline,
                                              make_model_generate_fn)
    from got_drive.world_model import predict_next_frame
    from got_drive.wm_image_metric import frame_distances
    from eval_nuscenes import load_model, set_seed
    from eval_got_nuscenes import load_world_model

    set_seed(args.seed)
    assert args.n_segments * args.segment_len == args.time_horizon, (
        f"n_segments*segment_len ({args.n_segments}*{args.segment_len}) != "
        f"time_horizon ({args.time_horizon})")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # segment index -> horizon label (segment s ends at (s+1)*segment_len*0.5 s)
    seg_horizons = {s: f"{(s + 1) * args.segment_len * 0.5:.1f}s" for s in range(args.n_segments)}

    # ---- join planning records with future-frame records on sample_token ----
    with open(args.records_json) as f:
        plan_recs = {r["sample_token"]: r for r in json.load(f)}
    with open(args.wm_eval_json) as f:
        frame_recs = json.load(f)
    joined = [(fr, plan_recs[fr["sample_token"]]) for fr in frame_recs
              if fr["sample_token"] in plan_recs]
    missing = len(frame_recs) - len(joined)
    n_all = len(joined)
    if args.offset:
        joined = joined[args.offset:]
    # --stride before --limit, so --limit still means "N records evaluated".
    # Records arrive grouped by scene (nuScenes: ~40 per scene; NAVSIM navtest:
    # 69,405 over 136 logs = ~510 per log), so a plain --limit 30 on NAVSIM is
    # ONE log -- and every CI here is record-clustered by scene (sec.9), which
    # with a single cluster is not a CI at all. Spread the sample instead.
    if args.stride > 1:
        joined = joined[:: args.stride]
    if args.limit:
        joined = joined[: args.limit]
    # n_scenes is the cluster count the bootstrap actually gets. It is printed
    # because "30 records" and "30 records from 1 scene" look identical in the
    # arguments and completely different in the CI.
    n_scenes = len({fr.get("scene") for fr, _ in joined})
    print(f"[wm-image] frame_recs={len(frame_recs)} joinable={n_all} "
          f"evaluating={len(joined)} over {n_scenes} scene(s) "
          f"(offset={args.offset}, stride={args.stride}, limit={args.limit}; "
          f"dropped {missing} without a planning record)")
    if len(joined) > 1 and n_scenes == 1:
        print("[wm-image] WARNING: every record is from ONE scene. The "
              "record-clustered CIs will have a single cluster and mean "
              "nothing. Use --stride to spread the sample.")

    # arm list: gt is the reference and is always present; eps arms are appended.
    arms = list(dict.fromkeys(["gt"] + list(args.arms)))
    arms += [f"eps{v:g}" for v in args.eps]
    want_plan = "plan" in arms
    # Cached plans replace the base model entirely: the only thing it was loaded
    # for is turning the t0 frame into a trajectory, and got_pred already is one.
    cached_plans = _load_plans_csv(args.plans_csv) if (want_plan and args.plans_csv) else None
    need_plan = want_plan and cached_plans is None
    if cached_plans is not None:
        print(f"[wm-image] plan arm from {args.plans_csv}: {len(cached_plans)} cached "
              f"trajectories (base model SKIPPED, one card is enough)")
    print(f"[wm-image] arms = {arms}  (base model {'LOADED' if need_plan else 'SKIPPED'})")
    frames_dir = None
    if args.save_frames:
        frames_dir = Path(args.save_frames)
        frames_dir.mkdir(parents=True, exist_ok=True)
        print(f"[wm-image] writing frames to {frames_dir} for the first "
              f"{args.save_frames_max} records")

    item_processor = FlexARItemProcessor_Action_NuScenes(
        tokenizer=args.tokenizer_path, target_size=args.resolution, norm_path=args.norm_path,
    )
    # The §7.5.1 gate run scores gt/mirror/copy only, none of which needs a plan.
    # Skipping the base model there frees ~14 GB and the 20 planning calls per record.
    model = generate_fn = cfg = None
    if need_plan:
        if not args.resume_path:
            raise SystemExit("--resume_path is required when 'plan' is among --arms")
        model = load_model(args)
        generate_fn = make_model_generate_fn(model, item_processor, args.prompt, args)
        cfg = DriveGoTConfig(
            n_segments=args.n_segments, segment_len=args.segment_len,
            k_candidates=args.k_candidates, beam_width=args.beam_width,
            time_horizon=args.time_horizon, temperatures=tuple(args.temperatures),
            w_kinematic=args.w_kinematic, w_command=args.w_command, verbose=args.verbose_plan,
        )

    print(f"[wm-image] loading world model from {args.wm_path}")
    wm = load_world_model(args)

    # bucket[arm][metric][segment] -> absolute distances
    bucket = {a: {mk: [[] for _ in range(args.n_segments)] for mk in METRIC_KEYS} for a in arms}
    # paired[arm][metric] -> [(record_idx, segment, d_arm - d_gt)] where BOTH succeeded.
    # Paired because the WM's reconstruction floor only cancels in the difference, and
    # because an unpaired mean subtracts means over DIFFERENT samples whenever the WM
    # drops frames unevenly -- which a weak WM does constantly.
    paired = {a: {mk: [] for mk in METRIC_KEYS} for a in arms}
    attempts = {a: [0, 0] for a in arms}          # [n_ok, n_attempted]
    spearman_pairs = {"delta": [], "L2": []}
    plan_seg_errors = []
    rows = []
    n_plan_failed = 0
    n_skipped = 0
    n_saved = [0]                       # list so the loop body can mutate it

    for i, (fr, prec) in enumerate(joined):
        frames = fr["frames"]                                    # n_segments+1 paths
        gt_all = np.array(prec["waypoints"], dtype=np.float64)
        # Guard both joins. A short GT would let seg_local_actions silently hand the
        # WM a 1-waypoint action where it expects segment_len (numpy just returns a
        # shorter slice), and too few frames would IndexError at frames[s+1]. Either
        # way the distances would be quietly wrong rather than obviously broken.
        if gt_all.shape[0] < args.time_horizon or len(frames) < args.n_segments + 1:
            n_skipped += 1
            continue
        gt_traj = gt_all[: args.time_horizon]
        command = prec["command"]

        gt_actions = seg_local_actions(gt_traj, args.segment_len, args.n_segments)

        # ---- GoT plan (Mode A), only when an arm needs it ----
        merged, plan_actions, plan_ok = None, None, False
        if need_plan:
            t0_img = crop_for_eval(Image.open(frames[0]).convert("RGB"),
                                   item_processor, args.legacy_random_crop)
            # Re-seed per record so a record's plan is identical whichever --offset /
            # --limit window it falls in. The gate run and the full run overlap on
            # records, and a paired reading is only valid if they planned the same way.
            set_seed((args.seed * 1_000_003 + args.offset + i) % (2 ** 31 - 1))
            pipe = DriveGoTPipeline(cfg, generate_fn, initial_image=t0_img)
            merged, _ = pipe.plan(command)
            plan_ok = merged is not None and merged.shape == (args.time_horizon, args.action_dim)
            if plan_ok:
                plan_actions = seg_local_actions(merged, args.segment_len, args.n_segments)
            else:
                n_plan_failed += 1
        elif cached_plans is not None:
            # Same trajectory the reported table was computed from, so the frames
            # illustrate the measured result rather than a fresh re-plan that would
            # differ by the sampling noise the 3-seed spread already quantifies.
            merged = cached_plans.get(fr["sample_token"])
            plan_ok = (merged is not None
                       and merged.shape == (args.time_horizon, args.action_dim))
            if plan_ok:
                plan_actions = seg_local_actions(merged, args.segment_len, args.n_segments)
            else:
                merged = None
                n_plan_failed += 1

        row = {"sample_token": fr["sample_token"], "scene": fr["scene"],
               "command": command, "plan_ok": plan_ok}
        save_here = (frames_dir is not None and n_saved[0] < args.save_frames_max)
        rec_dir = None
        if save_here:
            rec_dir = frames_dir / f"{i:04d}_{fr['sample_token'][:12]}_{command}"
            rec_dir.mkdir(parents=True, exist_ok=True)
            n_saved[0] += 1

        # ---- teacher-forced per-segment rollout, all arms on the same frames ----
        delta_record = []          # per-segment (d_plan - d_gt), NOT d_plan
        for s in range(args.n_segments):
            # Both frames go through the SAME deterministic crop. The anchor must,
            # so the WM's input (and hence its prediction's grid) is fixed; the
            # TARGET must too, because otherwise frame_distances falls back to a
            # single BICUBIC resize down to the prediction's size, while the
            # prediction came through the model's progressive-BOX chain. Measured
            # cost of that mismatch: codebook_l2 10.69 between a frame and itself.
            anchor = crop_for_eval(Image.open(frames[s]).convert("RGB"),
                                   item_processor, args.legacy_random_crop)
            target = crop_for_eval(Image.open(frames[s + 1]).convert("RGB"),
                                   item_processor, args.legacy_random_crop)
            if save_here:
                # the cropped versions, not the raw files: these are what the WM saw
                # and what the distances were computed against, so the figure and the
                # number describe the same pixels.
                anchor.save(rec_dir / f"s{s}_anchor.png")
                target.save(rec_dir / f"s{s}_real.png")

            if plan_ok:
                # displacement between where the plan puts the ego at this segment's
                # end and where it actually was -- the viewpoint error the metric has
                # to resolve, and the quantity the non-monotonic region is defined on.
                j = (s + 1) * args.segment_len - 1
                seg_err = float(np.linalg.norm(merged[j] - gt_traj[j]))
                plan_seg_errors.append(seg_err)
                row[f"plan_seg_err_s{s}"] = round(seg_err, 5)

            d = {}
            for arm in arms:
                if arm == "copy":
                    # no WM call: assert the anchor IS the future
                    attempts[arm][0] += 1
                    attempts[arm][1] += 1
                    d[arm] = frame_distances(item_processor, anchor, target)
                    continue
                act = arm_action(arm, s, gt_actions, plan_actions)
                if act is None:
                    continue
                attempts[arm][1] += 1
                pred = predict_next_frame(
                    wm, item_processor, anchor, act, wm.device,
                    max_new_tokens=args.wm_max_new_tokens,
                    do_sample=args.wm_do_sample,
                    temperature=args.wm_temperature,
                    top_k=args.wm_top_k)
                if pred is not None:
                    attempts[arm][0] += 1
                    if save_here:
                        pred.save(rec_dir / f"s{s}_{arm}.png")
                d[arm] = frame_distances(item_processor, pred, target)

            for arm, dist in d.items():
                for mk in METRIC_KEYS:
                    v = dist.get(mk)
                    if v is None:
                        continue
                    bucket[arm][mk][s].append(v)
                    row[f"{arm}_{mk}_s{s}"] = round(v, 5)
                    ref = d.get("gt", {}).get(mk)
                    if arm != "gt" and ref is not None:
                        paired[arm][mk].append((i, s, v - ref))
                        row[f"delta_{arm}_{mk}_s{s}"] = round(v - ref, 5)
            # sec.7.7: the gate must correlate DELTA with L2, not d_plan.
            # d_plan carries the WM's trajectory-independent reconstruction
            # floor and the scene's difficulty, and those are a component
            # shared with L2 -- which is why rho(d_plan, L2) = +0.4217 while
            # rho(delta, L2) = -0.0091 on the same 250 records (sec.11.9).
            # The old definition made this gate PASS on a WM that cannot
            # rank plans at all.
            if (d.get("plan", {}).get("codebook_l2") is not None
                    and d.get("gt", {}).get("codebook_l2") is not None):
                delta_record.append(d["plan"]["codebook_l2"]
                                    - d["gt"]["codebook_l2"])

        # per-record pair for the spearman gate: mean DELTA vs the plan's actual error
        if plan_ok:
            # written for every planned record (not only the ones with a d_plan) so a
            # sharded run can be re-summarised from the CSVs alone -- see --summarize_only
            row["plan_avgL2"] = round(float(np.linalg.norm(merged - gt_traj, axis=1).mean()), 5)
        if plan_ok and delta_record:
            spearman_pairs["delta"].append(float(np.mean(delta_record)))
            spearman_pairs["L2"].append(row["plan_avgL2"])

        rows.append(row)
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(joined)}  (plan_failed={n_plan_failed})", flush=True)

    # ---- per-sample csv ----
    csv_path = os.path.join(args.output_dir, "per_sample.csv")
    if rows:
        keys = sorted({k for r in rows for k in r})
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)

    # ---- summary ----
    summary = {
        "seed": args.seed,
        "n_joined": len(joined),
        "n_skipped_short_gt_or_frames": n_skipped,
        "n_plan_failed": n_plan_failed,
        "n_segments": args.n_segments,
        "segment_horizons": seg_horizons,
        "note": ("READ THE GATES FIRST (§7.5.1). delta_vs_gt_paired['plan'] is only "
                 "interpretable once wm_gain_over_copy and sensitivity_mirror PASS: a WM "
                 "that ignores its action input scores d_plan ~ d_gt, i.e. delta ~ 0, which "
                 "is indistinguishable from 'the plan is as plausible as the ground truth'. "
                 "Read n_pairs and n_records -- a weak WM drops frames and a tiny pair count "
                 "is not a result. codebook_l2 is primary; codebook_cosine and pixel_mae are "
                 "reported beside it because the metric saturates (session 7: it is graded "
                 "only over ~0-3 m of viewpoint displacement and DECLINES beyond ~4 m)."),
        **summarize(bucket, paired, arms, seg_horizons, args.n_segments, attempts,
                    args.n_boot, spearman_pairs, plan_seg_errors, args.plan_error_flag_m),
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== nuScenes offline WM-image eval ===")
    print(json.dumps(summary, indent=2))
    print(f"\nper-sample -> {csv_path}")


# ──────────────────────────────────────────────────────────────────────────
# self-test: gate arithmetic and action transforms only. No torch/GPU/VQGAN.
#   python eval_wm_image_nuscenes.py --selftest
# ──────────────────────────────────────────────────────────────────────────

def _selftest():
    # ---- action transforms ----
    gt = [np.array([[1.0, 2.0], [3.0, -4.0]])]
    assert np.allclose(arm_action("gt", 0, gt, None), gt[0])
    assert np.allclose(arm_action("mirror", 0, gt, None), [[1.0, -2.0], [3.0, 4.0]])
    assert np.allclose(arm_action("eps1.5", 0, gt, None), [[1.0, 3.5], [3.0, -2.5]])
    assert np.allclose(gt[0], [[1.0, 2.0], [3.0, -4.0]]), "must not mutate the GT action"
    assert arm_action("plan", 0, gt, None) is None
    try:
        arm_action("nope", 0, gt, None); raise SystemExit("unknown arm must raise")
    except ValueError:
        pass

    # ---- rank / spearman ----
    assert _ranks([10, 20, 20, 30]) == [1.0, 2.5, 2.5, 4.0]
    assert _spearman([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]) == 1.0
    assert _spearman([1, 2, 3, 4, 5], [5, 4, 3, 2, 1]) == -1.0
    assert _spearman([1, 1, 1], [1, 2, 3]) is None, "zero variance -> None"

    # ---- record-clustered bootstrap ----
    same = [(r, 0, 5.0) for r in range(10)]
    ci = _cluster_boot_ci(same, 500)
    assert ci == [5.0, 5.0], f"degenerate data must give a point CI, got {ci}"
    assert _cluster_boot_ci([(0, 0, 1.0)], 500) is None, "<3 records -> None"
    # segments of one record must NOT count as independent samples: 3 records x 20
    # identical segments must stay as uncertain as 3 records x 1.
    wide = _cluster_boot_ci([(r, s, float(r)) for r in range(3) for s in range(20)], 2000)
    narrow_if_wrong = _cluster_boot_ci([(r, 0, float(r % 3)) for r in range(60)], 2000)
    assert (wide[1] - wide[0]) > (narrow_if_wrong[1] - narrow_if_wrong[0]) * 2, (
        f"clustering not applied: {wide} vs {narrow_if_wrong}")

    # ---- gate verdicts ----
    hz = {0: "1.0s"}
    def _mk(arms, deltas, spear=None, errs=None):
        bucket = {a: {mk: [[]] for mk in METRIC_KEYS} for a in arms}
        paired = {a: {mk: [] for mk in METRIC_KEYS} for a in arms}
        for a, vals in deltas.items():
            paired[a]["codebook_l2"] = [(r, 0, v) for r, v in enumerate(vals)]
        att = {a: [10, 10] for a in arms}
        sp = spear or {"delta": [], "L2": []}
        return summarize(bucket, paired, arms, hz, 1, att, 400, sp, errs or [], 4.0)

    g = _mk(["gt", "mirror", "copy"],
            {"mirror": [2.0] * 12, "copy": [3.0] * 12})["gates"]
    assert g["sensitivity_mirror"]["verdict"] == "PASS", g["sensitivity_mirror"]
    assert g["wm_gain_over_copy"]["verdict"] == "PASS", g["wm_gain_over_copy"]

    g = _mk(["gt", "mirror"], {"mirror": [-2.0] * 12})["gates"]
    assert g["sensitivity_mirror"]["verdict"] == "FAIL", "WM worse than mirrored GT"

    g = _mk(["gt", "mirror"], {"mirror": [5.0, -5.0] * 6})["gates"]
    assert g["sensitivity_mirror"]["verdict"] == "INCONCLUSIVE", g["sensitivity_mirror"]

    # spearman: 1.0 is redundant with L2, ~0 is uninformative, middle passes
    xs = list(range(20))
    g = _mk(["gt", "plan"], {"plan": [0.0] * 20},
            spear={"delta": xs, "L2": xs})["gates"]
    assert g["spearman_delta_vs_L2"]["verdict"] == "FAIL_redundant"
    mid = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17, 19]
    g = _mk(["gt", "plan"], {"plan": [0.0] * 20},
            spear={"delta": xs, "L2": [float(v) + 6 * ((-1) ** v) for v in mid]})["gates"]
    assert g["spearman_delta_vs_L2"]["verdict"] == "PASS", g["spearman_delta_vs_L2"]

    # eps curve monotonicity
    s = _mk(["gt", "eps0.5", "eps2"], {"eps0.5": [1.0] * 12, "eps2": [3.0] * 12})
    assert s["gates"]["eps_curve"]["monotonic"] is True, s["gates"]["eps_curve"]
    s = _mk(["gt", "eps0.5", "eps2"], {"eps0.5": [3.0] * 12, "eps2": [1.0] * 12})
    assert s["gates"]["eps_curve"]["monotonic"] is False

    # non-monotonic-region flag
    s = _mk(["gt"], {}, errs=[1.0, 2.0, 5.0, 9.0])
    assert s["gates"]["frac_seg_plan_error_gt_4m"]["value"] == 0.5

    # paired block bookkeeping
    blk = _paired_block([(0, 0, 1.0), (0, 1, 3.0), (1, 0, 2.0)], {0: "1.0s", 1: "2.0s"}, 2, 400)
    assert blk["n_pairs"] == 3 and blk["n_records"] == 2 and blk["overall"] == 2.0
    assert blk["per_horizon"]["1.0s"] == 1.5 and blk["per_horizon"]["2.0s"] == 3.0

    # ---- shard merge (--summarize_only) ----
    import tempfile
    hz2 = {0: "1.0s", 1: "2.0s"}
    def _write(path, toks, base):
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=[
                "sample_token", "plan_ok", "plan_avgL2",
                "gt_codebook_l2_s0", "plan_codebook_l2_s0", "copy_codebook_l2_s0",
                "gt_codebook_l2_s1", "plan_codebook_l2_s1", "copy_codebook_l2_s1",
                "delta_plan_codebook_l2_s0", "delta_plan_codebook_l2_s1",
                "plan_seg_err_s0", "plan_seg_err_s1"])
            w.writeheader()
            for n, t in enumerate(toks):
                v = base + n
                w.writerow({"sample_token": t, "plan_ok": "True", "plan_avgL2": 1.0 + v,
                            "gt_codebook_l2_s0": 10.0, "plan_codebook_l2_s0": 10.0 + v,
                            "copy_codebook_l2_s0": 12.0,
                            "gt_codebook_l2_s1": 10.0, "plan_codebook_l2_s1": 10.0 + v,
                            "copy_codebook_l2_s1": 12.0,
                            "delta_plan_codebook_l2_s0": float(v),
                            "delta_plan_codebook_l2_s1": float(v),
                            "plan_seg_err_s0": 1.0, "plan_seg_err_s1": 9.0})
    with tempfile.TemporaryDirectory() as d:
        a, b = os.path.join(d, "a.csv"), os.path.join(d, "b.csv")
        _write(a, [f"t{i}" for i in range(6)], 0)
        _write(b, [f"t{i}" for i in range(4, 10)], 4)   # 2 overlapping tokens
        s = _summary_from_csvs([a, b], 2, hz2, 400, 4.0)
        assert s["n_records_merged"] == 10, f"dedupe by sample_token failed: {s['n_records_merged']}"
        assert set(s["arms"]) == {"gt", "plan", "copy"}, s["arms"]
        # copy - gt is +2 everywhere
        assert s["gates"]["wm_gain_over_copy"]["overall"] == 2.0
        # half the segments have plan error 9 > 4
        assert s["gates"][f"frac_seg_plan_error_gt_4m"]["value"] == 0.5
        # delta rises with plan_avgL2 by construction -> rho == 1 -> redundant.
        # NOTE the fixture now writes the delta_ columns: the re-summarise
        # path reads DELTA, so a csv without them yields no gate at all --
        # which is how this test caught the rewiring.
        assert s["gates"]["spearman_delta_vs_L2"]["verdict"] == "FAIL_redundant"
        assert s["plan"]["success_rate"] == 1.0

    # ---- --plans_csv loader ----
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "per_sample.csv")
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["sample_token", "seed", "got_status", "got_pred"])
            w.writeheader()
            w.writerow({"sample_token": "tokA", "seed": 42, "got_status": "ok",
                        "got_pred": "[[1.0, 0.0], [2.0, 0.5]]"})
            w.writerow({"sample_token": "tokA", "seed": 43, "got_status": "ok",
                        "got_pred": "[[9.0, 9.0], [9.0, 9.0]]"})     # later seed must NOT win
            w.writerow({"sample_token": "tokB", "seed": 42, "got_status": "malformed_plan",
                        "got_pred": ""})                             # no trajectory to render
            w.writerow({"sample_token": "tokC", "seed": 42, "got_status": "ok",
                        "got_pred": "not-a-list"})                   # unparsable
        got = _load_plans_csv(p)
        assert set(got) == {"tokA"}, f"expected only tokA, got {sorted(got)}"
        assert np.allclose(got["tokA"], [[1.0, 0.0], [2.0, 0.5]]), "first row must win"
        empty = os.path.join(td, "empty.csv")
        with open(empty, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=["sample_token"]).writeheader()
        try:
            _load_plans_csv(empty)
            raise AssertionError("a csv with no got_pred must exit, not return {}")
        except SystemExit:
            pass
    print("  ok  --plans_csv: first seed wins, malformed/unparsable skipped, empty csv exits")

    print("eval_wm_image self-test: OK (action transforms, spearman, record-clustered "
          "bootstrap, 4 gate verdicts, shard merge incl. sample_token dedupe, plans_csv)")


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        _selftest()
    else:
        main()
