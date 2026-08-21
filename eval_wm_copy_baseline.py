"""
d_copy: the "nothing moves" baseline for the WM-image evaluation
(PROJECT_HANDOFF §7.5.1 Layer 1 -- the first and cheapest WM validity gate).

What it is
----------
For every teacher-forced segment the WM eval scores, we ask what distance you get
by simply CLAIMING THE ANCHOR FRAME IS THE FUTURE -- i.e. predicting that the
scene does not change over the 1 s segment:

    d_copy[s] = frame_distance(real_frame[s], real_frame[s+1])

No world model, no base model, NO GENERATION AT ALL. Only the VQGAN encoder the
item_processor already owns, so this runs in minutes and can be computed BEFORE a
world model exists.

Why it is the first gate
------------------------
The WM-image headline is delta = d_plan - d_gt, and §7.5.1 records the trap: a WM
that ignores its action input and just redraws the anchor produces d_plan ~ d_gt,
so delta ~ 0 -- which reads identically to "the plan is as plausible as the
ground truth". Good result and meaningless result are the same number.

d_copy breaks the tie from below. It is exactly the score of a WM that learned
nothing except "copy the input":

    d_gt > d_copy   ->  the WM is WORSE than not predicting at all. Stop here.
    d_gt < d_copy   ->  wm_gain_over_copy = d_copy - d_gt is what the WM bought.

Necessary, not sufficient: passing does not license reading delta. The
sensitivity_mirror and spearman_d_vs_L2 gates (§7.5.1 Layer 2) test whether the
WM responds to the ACTION, which copying trivially does not.

★ Read it STRATIFIED BY EGO SPEED, never as one mean
-----------------------------------------------------
Measured ladder on the real val frames (codebook_l2, deterministic crop):

    identical frame            0.00
    parked   1 s               7.97
    parked   3 s              13.99
    driving  1 s              18.89      <- 94% of the ceiling already
    driving  3 s              20.14
    two UNRELATED scenes      20.04      <- the ceiling

Position-wise token distance saturates under ego motion: one second at urban
speed shifts the whole image, so patch (i,j) of the anchor and patch (i,j) of the
target are unrelated content. Consequences:

  * A single mean over mixed records measures HOW MANY PARKED SCENES the sample
    contained, not how much the scene changes -- the same class of error as the
    §9 --limit trap. So we bucket by ego speed derived from the GT waypoints.
  * On moving records d_copy sits at the ceiling, which makes the gate easy to
    read (any real prediction should come down off it) but makes `delta` fragile:
    if the WM is weak, d_plan and d_gt both pin near 20 and delta -> 0 by
    SATURATION rather than by plan quality. That is the §7.5.1 trap wearing a new
    hat, so check the gate before reading any delta.
  * codebook_cosine and pixel_mae are reported beside codebook_l2 precisely so
    the saturation can be compared across metrics; §7.5 fixed codebook_l2 as the
    primary metric before this ladder was known.

Geometry
--------
Frames go through got_drive.eval_crop.eval_center_crop, the deterministic
counterpart of the training-time random crop. Using the raw frame instead costs
codebook_l2 10.69 BETWEEN A FRAME AND ITSELF (different aspect and a single
BICUBIC downsample vs the model's progressive-BOX chain). `sanity_self_distance`
loads one frame twice, independently, and asserts the distance is 0.

Pairing
-------
The gate is a PAIRED comparison on (record, segment) -- the standing rule for
anything compared across runs (§7.5 paired-delta fix, §9 "arm 비교는 반드시
paired"). Per-segment values go to per_sample.csv keyed by sample_token, so a
later WM run joins against them via --compare_csv.

Usage
-----
    python eval_wm_copy_baseline.py \
        --tokenizer_path ../ckpts/Lumina-mGPT-7B-768 \
        --records_json  ./data/nuscenes_records/nuscenes_v1.0-trainval_val.json \
        --wm_eval_json  ./data/nuscenes_wm_records/nuscenes_wm_eval_v1.0-trainval_val.json \
        --norm_path     ./data/nuscenes_records/nuscenes_norm.json \
        --output_dir ./results/wm_copy --device 0 --limit 0

    # later, once a WM run exists: the actual gate number
    ... --compare_csv ./results/wm_image/per_sample.csv
"""

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from data.item_processor import FlexARItemProcessor_Action_NuScenes
from got_drive.eval_crop import eval_center_crop
from got_drive.wm_image_metric import frame_distances

METRIC_KEYS = ("codebook_l2", "codebook_cosine", "pixel_mae")

# ego speed (m/s) -> bucket. Boundaries are coarse on purpose: the point is to
# stop parked and driving records being averaged together, not to model speed.
SPEED_BUCKETS = ((1.0, "parked"), (5.0, "slow"), (10.0, "medium"), (float("inf"), "fast"))


def get_args():
    p = argparse.ArgumentParser("d_copy baseline for the WM-image eval (no generation)")
    p.add_argument("--tokenizer_path", required=True)
    p.add_argument("--records_json", required=True)
    p.add_argument("--wm_eval_json", required=True)
    p.add_argument("--norm_path", default=None)
    p.add_argument("--output_dir", default="./results/wm_copy")
    p.add_argument("--resolution", type=int, default=256,
                   help="MUST match the WM eval's --resolution: it sets the crop grid")
    p.add_argument("--n_segments", type=int, default=3)
    p.add_argument("--segment_len", type=int, default=2)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--limit", type=int, default=0, help="0 = all")
    p.add_argument("--offset", type=int, default=0,
                   help="pairing is by sample_token, not index, so this need NOT match the "
                        "WM run's window -- prefer --limit 0 (generation-free, so covering "
                        "every scene is cheap and sidesteps the §9 scene-ordering trap)")
    p.add_argument("--compare_csv", default=None,
                   help="per_sample.csv from eval_wm_image_nuscenes.py; emits the paired "
                        "gate wm_gain_over_copy = d_copy - d_gt")
    return p.parse_args()


def _speed_bucket(waypoints, horizon_s=3.0):
    """Straight-line ego displacement over the GT horizon / time -> mean speed.

    Path length would double-count jitter in the annotations; the chord is the
    stable quantity and we only need coarse buckets.
    """
    wp = np.asarray(waypoints, dtype=np.float64)
    if wp.size == 0:
        return None, "unknown"
    v = float(np.linalg.norm(wp[-1])) / horizon_s
    for hi, name in SPEED_BUCKETS:
        if v < hi:
            return v, name
    return v, "fast"


def _stats(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    a = np.asarray(xs, dtype=np.float64)
    return {"mean": round(float(a.mean()), 5),
            "median": round(float(np.median(a)), 5),
            "p90": round(float(np.percentile(a, 90)), 5),
            "n": int(a.size)}


def _collect(rows, mk, s, key="copy", where=None):
    out = []
    for r in rows:
        if where is not None and r.get("speed_bucket") != where:
            continue
        v = r.get(f"{key}_{mk}_s{s}")
        if v is not None:
            out.append(float(v))
    return out


def _summarize(rows, n_segments, seg_horizons):
    buckets = sorted({r["speed_bucket"] for r in rows})
    out = {}
    for mk in METRIC_KEYS:
        out[mk] = {
            "overall": _stats([v for s in range(n_segments) for v in _collect(rows, mk, s)]),
            "per_horizon": {seg_horizons[s]: _stats(_collect(rows, mk, s))
                            for s in range(n_segments)},
            "by_speed": {b: {seg_horizons[s]: _stats(_collect(rows, mk, s, where=b))
                             for s in range(n_segments)}
                         for b in buckets},
        }
    return out


def _paired_gate(rows, compare_csv, n_segments, seg_horizons):
    """Join d_copy against a WM run's d_gt on (sample_token, segment).

    Paired, because segments differ enormously in difficulty (a parked record has
    d_copy ~ 8, a moving one ~ 19) and an unpaired mean would mostly measure which
    records each run happened to keep. Also split by speed for the same reason.
    """
    with open(compare_csv, newline="", encoding="utf-8") as f:
        wm_rows = {r["sample_token"]: r for r in csv.DictReader(f)}

    merged = []
    for r in rows:
        wr = wm_rows.get(r["sample_token"])
        if wr is None:
            continue
        m = {"sample_token": r["sample_token"], "speed_bucket": r["speed_bucket"]}
        for mk in METRIC_KEYS:
            for s in range(n_segments):
                c, g = r.get(f"copy_{mk}_s{s}"), wr.get(f"gt_{mk}_s{s}")
                if c is None or g in (None, ""):
                    continue
                m[f"copy_{mk}_s{s}"] = float(c) - float(g)   # reuse _collect's key shape
        merged.append(m)

    out = {"n_tokens_joined": len(merged),
           "_reading": ("wm_gain_over_copy = d_copy - d_gt. POSITIVE => the WM predicts the "
                        "future better than asserting the scene is unchanged. NEGATIVE or ~0 "
                        "=> the WM adds nothing over copying; per §7.5.1 the WM-image results "
                        "cannot be reported and drop to a limitation. Read the parked bucket "
                        "separately: on moving records d_copy sits at the metric ceiling, so "
                        "beating it there is a much weaker statement.")}
    out.update(_summarize(merged, n_segments, seg_horizons) if merged else {})
    return out


def main():
    args = get_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    horizon_s = args.n_segments * args.segment_len * 0.5

    with open(args.records_json) as f:
        plan_recs = {r["sample_token"]: r for r in json.load(f)}
    with open(args.wm_eval_json) as f:
        frame_recs = json.load(f)
    joined = [(fr, plan_recs[fr["sample_token"]]) for fr in frame_recs
              if fr["sample_token"] in plan_recs]
    n_all = len(joined)
    if args.offset:
        joined = joined[args.offset:]
    if args.limit:
        joined = joined[: args.limit]
    print(f"[d_copy] frame_recs={len(frame_recs)} joinable={n_all} evaluating={len(joined)} "
          f"(offset={args.offset}, limit={args.limit})", flush=True)

    item_processor = FlexARItemProcessor_Action_NuScenes(
        tokenizer=args.tokenizer_path, target_size=args.resolution,
        norm_path=args.norm_path, device=f"cuda:{args.device}",
    )
    csl = item_processor.crop_size_list
    load = lambda p: eval_center_crop(Image.open(p).convert("RGB"), csl)

    seg_horizons = {s: f"{(s + 1) * args.segment_len * 0.5:.1f}s" for s in range(args.n_segments)}
    rows = []
    n_skipped = 0
    sanity = None

    for i, (fr, prec) in enumerate(joined):
        frames = fr["frames"]
        if len(frames) < args.n_segments + 1:
            n_skipped += 1
            continue

        v, bucket = _speed_bucket(prec.get("waypoints", []), horizon_s)
        row = {"sample_token": fr["sample_token"], "scene": fr["scene"],
               "command": prec.get("command"),
               "speed_mps": None if v is None else round(v, 3), "speed_bucket": bucket}

        if sanity is None:
            # TWO INDEPENDENT LOADS of the same file -- this is the check that
            # catches a non-deterministic crop. Comparing one object against
            # itself does not, and that is how the random crop stayed hidden.
            sanity = frame_distances(item_processor, load(frames[0]), load(frames[0]))

        for s in range(args.n_segments):
            d = frame_distances(item_processor, load(frames[s]), load(frames[s + 1]))
            for mk in METRIC_KEYS:
                if d[mk] is not None:
                    row[f"copy_{mk}_s{s}"] = round(d[mk], 5)

        rows.append(row)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(joined)}", flush=True)

    csv_path = os.path.join(args.output_dir, "per_sample.csv")
    if rows:
        keys = sorted({k for r in rows for k in r})
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)

    summary = {
        "n_records": len(rows),
        "n_skipped_short_frames": n_skipped,
        "resolution": args.resolution,
        "segment_horizons": seg_horizons,
        "speed_bucket_edges_mps": {name: hi for hi, name in SPEED_BUCKETS},
        "n_by_speed": {b: sum(1 for r in rows if r["speed_bucket"] == b)
                       for b in sorted({r["speed_bucket"] for r in rows})} if rows else {},
        "sanity_self_distance": sanity,   # two independent loads; must be ~0
        "d_copy": _summarize(rows, args.n_segments, seg_horizons) if rows else {},
        "note": ("d_copy = distance from the segment's REAL start frame to its REAL end frame, "
                 "i.e. the score of predicting 'nothing changes' -- the floor the trained WM's "
                 "d_gt must beat (§7.5.1). READ by_speed, not the overall mean: the metric "
                 "saturates under ego motion (driving 1s ~ 19 vs an unrelated-scene ceiling of "
                 "~20), so a single mean mostly reports how many parked records the sample had."),
    }
    if args.compare_csv:
        summary["wm_gain_over_copy"] = _paired_gate(rows, args.compare_csv,
                                                    args.n_segments, seg_horizons)

    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== d_copy baseline (no generation) ===")
    print(json.dumps(summary, indent=2))
    print(f"\nper-sample -> {csv_path}")


if __name__ == "__main__":
    main()
