"""
Decisive diagnostic for the (A) PLUMBING failure in verify_got_prefix_nuscenes.

verify_*.py found self-consistency max|diff| ~2.5 m (should be <1e-6). Two
mutually-exclusive causes:

  (LAYOUT)   the re-encoded prefix tokens differ from the tokens the model
             actually emitted  -> a real round-trip/token-layout bug to fix.
  (NUMERIC)  the re-encoded prefix tokens are byte-identical, but bf16
             prefill-vs-decode non-determinism flips a greedy argmax at a tie,
             which then cascades  -> NOT a bug; the <1e-6 test threshold is too
             strict for bf16 generation, and segment-wise GoT is still fine.

This script tells them apart per record by dumping, for the first k groups:
    emitted_prefix_ids   (what the model produced in free-run)
    reencoded_prefix_ids (what tokenize_waypoint_prefix feeds back)
and, when those are identical, whether the *continuation* tokens still diverge
(which can only be NUMERIC).

Usage mirrors verify_got_prefix_nuscenes.py:

    python diag_prefix_tokens.py \
        --resume_path output/nuscenes_trainval_full_r256/epoch0 \
        --tokenizer_path ../ckpts/Lumina-mGPT-7B-768 \
        --records_json data/nuscenes_records/nuscenes_v1.0-trainval_val.json \
        --norm_path data/nuscenes_records/nuscenes_norm_v1.0-trainval.json \
        --prefix_k 2 --limit 5 --device 0
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
import torch
from PIL import Image
from transformers import GenerationConfig

from data.item_processor import FlexARItemProcessor_Action_NuScenes
from data.dataset_nuscenes import DEFAULT_PROMPT
from eval_nuscenes import load_model
from model.chameleon import ChameleonForConditionalGeneration
from got_drive.eval_crop import eval_center_crop
from got_drive.segment_generation import (
    build_context_ids, tokenize_waypoint_prefix, unnorm_waypoints,
    SEP_EOS_TOKEN, ACTION_START_TOKEN, ACTION_END_TOKEN,
)


def get_args():
    p = argparse.ArgumentParser("prefix-token round-trip diagnostic")
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
    p.add_argument("--prefix_k", type=int, default=2)
    p.add_argument("--limit", type=int, default=5)
    return p.parse_args()


@torch.no_grad()
def raw_generate(model, input_ids, args, n_generate):
    """Return the raw emitted token ids (after the context), greedy."""
    max_new = n_generate * (args.action_dim + 2) + 8
    gc = GenerationConfig(
        max_new_tokens=max_new,
        max_length=model.config.max_position_embeddings,
        do_sample=False, top_k=None, temperature=1.0,
        eos_token_id=[SEP_EOS_TOKEN],
    )
    model.init_input_ids = None
    res = ChameleonForConditionalGeneration.generate(
        model, input_ids=input_ids, generation_config=gc,
        output_hidden_states=True, training=False, return_dict_in_generate=True,
    )
    return res["sequences"][0, input_ids.shape[1]:].detach().cpu().tolist()


def parse_groups(token_list):
    """Split a flat token stream into [10004 ... 15004] groups (markers kept)."""
    groups, start = [], None
    for i, t in enumerate(token_list):
        if t == ACTION_START_TOKEN:
            start = i
        elif t == ACTION_END_TOKEN and start is not None:
            groups.append(token_list[start:i + 1])
            start = None
    return groups


def decode_groups_to_metres(model, item_processor, groups, action_dim):
    """Each group's inner tokens -> bin centres -> metres."""
    inner = []
    for g in groups:
        toks = torch.tensor(g[1:1 + action_dim], dtype=torch.int64,
                            device=model.device)
        inner.append(model.decode_token_ids_to_actions(toks).float().cpu().numpy())
    norm = np.stack(inner, axis=0)
    return unnorm_waypoints(norm, item_processor.wp_min, item_processor.wp_max)


def main():
    args = get_args()
    k, H, D = args.prefix_k, args.time_horizon, args.action_dim

    with open(args.records_json) as f:
        records = json.load(f)[: args.limit]

    ip = FlexARItemProcessor_Action_NuScenes(
        tokenizer=args.tokenizer_path, target_size=args.resolution,
        norm_path=args.norm_path,
    )
    model = load_model(args)

    n_layout_mismatch = 0
    n_numeric_only = 0
    n_clean = 0

    for idx, rec in enumerate(records):
        # This script calls build_context_ids TWICE per record (free run, then the
        # re-encoded prefix). Each call re-runs process_image, which random-crops --
        # so without a deterministic crop the two contexts hold different IMAGE
        # tokens and any divergence is mis-read as numeric. The byte-identical
        # finding recorded in §7.2 covered the WAYPOINT prefix tokens only, which
        # are unaffected by the crop, so it could not have caught this.
        image = eval_center_crop(Image.open(rec["images"][0]).convert("RGB"),
                                 ip.crop_size_list)

        # ---- free run, capture RAW emitted tokens ----
        ctx = build_context_ids(model, ip, image, args.prompt, None)
        emitted = raw_generate(model, ctx, args, H)
        groups = parse_groups(emitted)
        if len(groups) < H:
            print(f"[rec {idx}] free-run malformed ({len(groups)} groups) -- skip")
            continue

        emitted_prefix_ids = [t for g in groups[:k] for t in g]
        emitted_tail_ids = [t for g in groups[k:H] for t in g]

        # ---- re-encode the model's own first-k waypoints ----
        free_wp = decode_groups_to_metres(model, ip, groups[:H], D)
        reencoded_prefix_ids = tokenize_waypoint_prefix(ip, free_wp[:k])

        layout_ok = (emitted_prefix_ids == reencoded_prefix_ids)

        print(f"\n[rec {idx}] prefix layout {'OK' if layout_ok else 'MISMATCH'}")
        print(f"  emitted   prefix ids : {emitted_prefix_ids}")
        print(f"  reencoded prefix ids : {reencoded_prefix_ids}")
        if not layout_ok:
            n_layout_mismatch += 1
            # show the first differing position
            for j, (a, b) in enumerate(zip(emitted_prefix_ids, reencoded_prefix_ids)):
                if a != b:
                    print(f"  first diff at pos {j}: emitted {a} vs reencoded {b}")
                    break
            continue

        # ---- prefix identical: does the continuation still diverge? ----
        prefix_ctx = build_context_ids(model, ip, image, args.prompt, free_wp[:k])
        tail_emitted2 = raw_generate(model, prefix_ctx, args, H - k)
        tail_groups = parse_groups(tail_emitted2)
        tail_ids2 = [t for g in tail_groups[:H - k] for t in g]

        if tail_ids2 == emitted_tail_ids:
            n_clean += 1
            print("  continuation IDENTICAL -> fully reproducible (clean)")
        else:
            n_numeric_only += 1
            print("  prefix identical but continuation DIVERGES -> NUMERIC "
                  "(bf16 prefill-vs-decode), not a layout bug")
            # locate first divergent token
            for j, (a, b) in enumerate(zip(emitted_tail_ids, tail_ids2)):
                if a != b:
                    print(f"  tail first diff at pos {j}: free {a} vs prefixed {b}")
                    break

    print("\n=== summary ===")
    print(f"  layout mismatch (real bug) : {n_layout_mismatch}")
    print(f"  numeric-only divergence    : {n_numeric_only}")
    print(f"  fully reproducible         : {n_clean}")
    print("\nverdict:")
    if n_layout_mismatch:
        print("  -> LAYOUT bug: re-encoded prefix tokens differ from emitted. Fix "
              "tokenize_waypoint_prefix / process_action round-trip.")
    else:
        print("  -> NO layout bug. (1) PLUMBING 'failure' is bf16 numeric "
              "non-determinism; the <1e-6 threshold is too strict. Segment-wise "
              "GoT plumbing is correct -- proceed, and re-judge (3) VALUE SIGNAL.")


if __name__ == "__main__":
    main()
