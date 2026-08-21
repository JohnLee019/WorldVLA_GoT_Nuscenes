"""
Write a turn-only view of a nuScenes planning records json.

Why. Across the 500-record val slice, GoT is a measured no-op on 'straight'
(diff -0.030, win 0.498, p=0.61) and wins on turns (diff -0.150, win 0.576,
p=0.029). Since ~82% of val is 'straight', an eval spends most of its GPU
budget on the regime that provably carries no signal: --limit 500 buys only
~92 turn records. Evaluating the same 500 records drawn from turns only buys
~5x the statistics exactly where the effect lives, at identical cost.

This is a subsample of the eval set, not a change to the method: the records,
the GT and the metrics are untouched, and the straight side stays reportable
from the existing full-split runs. Report both, and label the turn runs as a
turn-conditioned evaluation -- never as the headline split.

    python make_turn_subset.py \
        --records ./data/nuscenes_records/nuscenes_v1.0-trainval_val.json \
        --out ./data/nuscenes_records/nuscenes_v1.0-trainval_val_turns.json
"""

import argparse
import json
from collections import Counter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep", nargs="+", default=["left", "right"],
                    help="commands to keep (default: left right)")
    ap.add_argument("--min_waypoints", type=int, default=6,
                    help="drop short-GT records here rather than having every "
                         "eval skip them separately (eval_got reports them as "
                         "n_skipped_short_gt)")
    ap.add_argument("--per_scene", type=int, default=0,
                    help="keep at most K evenly-spaced records per scene (0 = keep all). "
                         "THE fix for --limit N: val is 150 scenes but the first 500 records "
                         "cover 15 of them and the first 200 cover SIX, so a limited run is a "
                         "handful of manoeuvres seen repeatedly, not a sample of driving. "
                         "--per_scene 4 gives ~600 records across ALL scenes for the same GPU "
                         "cost. Evenly spaced rather than random so the subset is deterministic "
                         "and covers each scene start-to-end.")
    args = ap.parse_args()

    with open(args.records) as f:
        recs = json.load(f)

    keep = set(args.keep)
    before = Counter(r.get("command") for r in recs)
    out, n_short = [], 0
    for r in recs:
        if r.get("command") not in keep:
            continue
        if len(r.get("waypoints") or []) < args.min_waypoints:
            n_short += 1
            continue
        out.append(r)

    if args.per_scene:
        by_scene = {}
        for r in out:
            by_scene.setdefault(r.get("scene"), []).append(r)
        thinned, order = [], []
        for sc, rs in by_scene.items():
            k = min(args.per_scene, len(rs))
            # evenly spaced indices spanning the scene, endpoints included
            idx = [round(i * (len(rs) - 1) / (k - 1)) for i in range(k)] if k > 1 else [0]
            for i in sorted(set(idx)):
                thinned.append(rs[i])
                order.append((sc, i))
        print(f"per_scene={args.per_scene}: {len(out)} -> {len(thinned)} records "
              f"over {len(by_scene)} scenes")
        out = thinned

    with open(args.out, "w") as f:
        json.dump(out, f)

    after = Counter(r["command"] for r in out)
    print(f"in : {len(recs):>6} records  {dict(before)}")
    print(f"out: {len(out):>6} records  {dict(after)}  "
          f"(dropped {n_short} with <{args.min_waypoints} waypoints)")
    print(f"wrote {args.out}")
    # scene spread matters: records are stored scene-ordered, so a --limit N on
    # this file is still the first N, i.e. the first few turning scenes. Say how
    # many scenes N would actually cover.
    scenes = []
    for r in out:
        s = r.get("scene")
        if not scenes or scenes[-1] != s:
            scenes.append(s)
    print(f"{len(set(scenes))} distinct scenes; --limit 500 would cover "
          f"~{len(set(s for s in (r.get('scene') for r in out[:500])))}")


if __name__ == "__main__":
    main()
