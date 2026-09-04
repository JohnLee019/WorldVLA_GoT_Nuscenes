"""
(A) building-block check: can the model CONTINUE a partial waypoint trajectory?

Segment-wise driving GoT rests on one assumption -- that feeding the model the
first k waypoints as a prefix and asking for the rest yields a coherent
continuation. This script tests that assumption on the (mini) val set, before
any GoT scaffolding is built on top of it.

Three things are measured per record (all greedy unless noted):

  1. PLUMBING (self-consistency).  Free-run all H waypoints, then feed the
     model's OWN first k back as a prefix and regenerate the tail. Under greedy
     decoding these MUST match: same context => same tokens. A near-zero max
     discrepancy proves the prefix is spliced in at exactly the right place
     (byte-for-byte training format, no missing separator). A large discrepancy
     means the plumbing is wrong, not the model.

  2. CONTINUATION HEALTH.  How often the prefixed generation returns the right
     number of well-formed groups (not malformed). If the model chokes whenever
     it is handed a prefix, segment-wise GoT is not viable on this checkpoint.

  3. VALUE SIGNAL.  Feed the GROUND-TRUTH first k as a prefix and compare the
     tail L2 (indices k..H-1) against the free-run tail L2 on the same indices.
     If conditioning on a correct prefix does not reduce tail error, the model
     is essentially ignoring the prefix -- worth knowing early.

Usage
-----
    python verify_got_prefix_nuscenes.py \
        --resume_path ./output/nuscenes_mini/epoch24 \
        --tokenizer_path ~/ckpts/models--Alpha-VLLM--Lumina-mGPT-7B-768 \
        --records_json ./data/nuscenes_records/nuscenes_v1.0-mini_val.json \
        --norm_path ./data/nuscenes_records/nuscenes_norm_v1.0-trainval.json \
        --prefix_k 2 --limit 20

Read the caveats in eval_nuscenes.py first: mini val is not comparable to mini
train, and open-loop L2 is a weak metric. This is a mechanism check, not a
driving-quality benchmark.
"""
# --- 리포 루트를 import 경로에 넣는다 -------------------------------------
# 이 파일은 2026-08-21에 루트에서 이 폴더로 옮겨졌다. 파이썬은 sys.path[0]에
# *스크립트가 있는 폴더*를 넣으므로, 이 두 줄이 없으면 `python analysis/x.py`가
# got_drive / model / xllmx 를 못 찾고 ModuleNotFoundError로 죽는다.
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
# -------------------------------------------------------------------------

import argparse
import json

import numpy as np
from PIL import Image

from data.item_processor import FlexARItemProcessor_Action_NuScenes
from data.dataset_nuscenes import DEFAULT_PROMPT
from eval_nuscenes import load_model
from got_drive.eval_crop import crop_for_eval
from got_drive.segment_generation import predict_segment


def get_args():
    p = argparse.ArgumentParser("nuScenes (A) prefix-continuation check")
    p.add_argument("--resume_path", required=True)
    p.add_argument("--tokenizer_path", required=True)
    p.add_argument("--records_json", required=True)
    p.add_argument("--norm_path", default=None)
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--max_seq_len", type=int, default=4096)
    p.add_argument("--action_dim", type=int, default=2)
    p.add_argument("--time_horizon", type=int, default=6)
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--load_in_4bit", action="store_true", default=False)
    p.add_argument("--prefix_k", type=int, default=2,
                   help="number of prefix waypoints to condition on")
    p.add_argument("--legacy_random_crop", action="store_true", default=False,
                   help="reproduce pre-fix behaviour. Test (1) PLUMBING generates the same "
                        "record twice and demands identical output; with the random crop "
                        "active the two calls see DIFFERENT crops of the image, which is the "
                        "likely cause of the 2.4 m (worst 11.4 m) self-consistency gap §7.2 "
                        "attributed to bf16. See got_drive/eval_crop.py.")
    p.add_argument("--limit", type=int, default=20,
                   help="evaluate only the first N records (0 = all)")
    return p.parse_args()


def tail_l2(pred_tail, gt_tail):
    """Mean per-step L2 (metres) over the continued segment."""
    return float(np.linalg.norm(np.asarray(pred_tail) - np.asarray(gt_tail), axis=-1).mean())


def main():
    args = get_args()
    k, H = args.prefix_k, args.time_horizon
    assert 0 < k < H, f"prefix_k must be in (0, {H})"

    with open(args.records_json) as f:
        records = json.load(f)
    if args.limit:
        records = records[: args.limit]

    item_processor = FlexARItemProcessor_Action_NuScenes(
        tokenizer=args.tokenizer_path,
        target_size=args.resolution,
        norm_path=args.norm_path,
    )
    print(f"[verify] waypoint un-norm range: min={item_processor.wp_min.tolist()} "
          f"max={item_processor.wp_max.tolist()}")
    model = load_model(args)

    # per-record accumulators
    n = len(records)
    self_consistent_diffs = []   # (1) plumbing: max |free_tail - self_prefixed_tail|
    n_free_malformed = 0
    n_gt_prefix_malformed = 0
    n_self_prefix_malformed = 0
    l2_free_tail = []            # (3) free-run L2 on indices k..H-1
    l2_gt_prefix_tail = []       # (3) GT-prefixed L2 on the same indices

    for i, rec in enumerate(records):
        # Deterministic crop: test (1) compares two generations of the SAME record,
        # so a per-call random crop would make them differ for a reason that has
        # nothing to do with prefix plumbing.
        image = crop_for_eval(Image.open(rec["images"][0]).convert("RGB"),
                              item_processor, args.legacy_random_crop)
        gt = np.array(rec["waypoints"], dtype=np.float64)  # (H, 2)

        # free run: all H waypoints, greedy
        free = predict_segment(model, item_processor, image, args.prompt, args,
                               prefix_wp=None, n_generate=H,
                               do_sample=False)
        if free is None:
            n_free_malformed += 1
        else:
            l2_free_tail.append(tail_l2(free[k:], gt[k:]))

            # (1) self-consistency: feed the model's own first k back
            self_cont = predict_segment(model, item_processor, image, args.prompt, args,
                                        prefix_wp=free[:k], n_generate=H - k,
                                        do_sample=False)
            if self_cont is None:
                n_self_prefix_malformed += 1
            else:
                self_consistent_diffs.append(
                    float(np.abs(np.asarray(self_cont) - free[k:]).max()))

        # (2)+(3) teacher-forced: feed the GROUND-TRUTH first k
        gt_cont = predict_segment(model, item_processor, image, args.prompt, args,
                                  prefix_wp=gt[:k], n_generate=H - k,
                                  do_sample=False)
        if gt_cont is None:
            n_gt_prefix_malformed += 1
        else:
            l2_gt_prefix_tail.append(tail_l2(gt_cont, gt[k:]))

        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{n}")

    # ---------- verdict ----------
    def m(xs):
        return round(float(np.mean(xs)), 4) if xs else None

    print("\n=== (A) prefix-continuation check ===")
    print(f"records evaluated         : {n}   (prefix_k={k}, horizon={H})")
    print(f"free-run malformed        : {n_free_malformed}/{n}")
    print(f"GT-prefix malformed       : {n_gt_prefix_malformed}/{n}")
    print(f"self-prefix malformed     : {n_self_prefix_malformed}/{n}")

    print("\n(1) PLUMBING  self-consistency (greedy free tail vs self-prefixed tail)")
    if self_consistent_diffs:
        arr = np.array(self_consistent_diffs)
        print(f"    max |diff| over records : mean={arr.mean():.4g}  worst={arr.max():.4g} (metres)")
        ok = arr.max() < 1e-6
        print(f"    -> {'PASS' if ok else 'CHECK'}: prefix splices in "
              f"{'exactly where the model resumes' if ok else 'with a mismatch -- inspect token layout'}")
    else:
        print("    (no successful free runs to compare)")

    print("\n(2) CONTINUATION HEALTH")
    healthy = n - n_gt_prefix_malformed
    print(f"    GT-prefixed continuations that were well-formed: {healthy}/{n}")

    print("\n(3) VALUE SIGNAL  tail L2 on indices "
          f"{k}..{H - 1} (metres, lower is better)")
    print(f"    free-run tail L2       : {m(l2_free_tail)}   (n={len(l2_free_tail)})")
    print(f"    GT-prefixed tail L2    : {m(l2_gt_prefix_tail)}   (n={len(l2_gt_prefix_tail)})")
    if l2_free_tail and l2_gt_prefix_tail:
        delta = m(l2_free_tail) - m(l2_gt_prefix_tail)
        print(f"    improvement from prefix: {round(delta, 4)}  "
              f"({'prefix helps' if delta > 0 else 'prefix does NOT help'})")

    print("\nInterpretation:")
    print("  (1) PASS  => segment splicing is correct; build got_pipeline on it.")
    print("  (2) high  => model tolerates prefixes; segment-wise GoT is viable.")
    print("  (3) >0    => model actually uses the prefix (conditioning is real).")
    print("  If (1) fails: fix token layout. If (2) is low or (3)<=0: the base")
    print("  checkpoint can't be steered by a prefix -> make training segment-aware.")


if __name__ == "__main__":
    main()
