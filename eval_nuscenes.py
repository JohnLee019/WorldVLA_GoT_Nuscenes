"""
Open-loop trajectory-planning evaluation on nuScenes.

Feeds each val record's camera image + planning prompt to the model, decodes the
generated action tokens back into ego-frame waypoints, and reports the standard
nuScenes planning metric: L2 displacement error at 1 s / 2 s / 3 s.

Two conventions are reported because the literature uses both:
    L2@t      -- displacement at exactly t   (UniAD-style)
    avgL2@t   -- mean displacement over all steps up to t (VAD-style)
Papers are not always explicit about which one they quote, so compare like
with like before claiming parity with a published number.

Usage
-----
    python eval_nuscenes.py \
        --resume_path ./output/nuscenes_mini/epoch19 \
        --tokenizer_path ~/ckpts/models--Alpha-VLLM--Lumina-mGPT-7B-768 \
        --records_json ./data/nuscenes_records/nuscenes_v1.0-mini_val.json \
        --norm_path ./data/nuscenes_records/nuscenes_norm.json \
        --output_dir ./results/nuscenes_eval

IMPORTANT
---------
* --norm_path MUST be the same file used for training. The waypoints come back
  from the model in the normalized [-1, 1] bin space and are un-normalized with
  these ranges; a mismatch silently produces wrong metres.
* A caveat on the metric itself: open-loop L2 on nuScenes is known to be weak --
  models that ignore the image and extrapolate ego motion still score well. Treat
  it as a regression check on the pipeline, not as proof of driving ability.
* On v1.0-mini specifically the val split is NOT comparable to train (train has
  zero right turns, val is ~29% right turns). Mini numbers are for smoke-testing
  only.
"""

import argparse
import csv
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import GenerationConfig

from model import ChameleonXLLMXForConditionalGeneration_ck_action_head
from data.item_processor import FlexARItemProcessor_Action_NuScenes
from data.dataset_nuscenes import DEFAULT_PROMPT
from got_drive.eval_crop import crop_for_eval


def set_seed(seed):
    """Fix all RNGs so a stochastic (do_sample) run is reproducible and the
    GoT-vs-baseline comparison is fair across runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_mean_trajectories(records, time_horizon):
    """Per-command (and overall) mean GT trajectory -- a model-free prior baseline.

    nuScenes open-loop L2 is famously weak: a predictor that ignores the image
    and just returns the average path per command scores well. If VLA+GoT does not
    clearly beat this, the model is not using the image. Compute the means from a
    TRAIN split (pass --train_records_json); using the eval split itself is
    slightly optimistic (peeks at the eval GT distribution)."""
    acc = defaultdict(list)
    for r in records:
        wp = np.array(r["waypoints"], dtype=np.float64)
        if wp.shape[0] < time_horizon:
            continue
        acc[r.get("command", "__all__")].append(wp[:time_horizon])
        acc["__all__"].append(wp[:time_horizon])
    return {k: np.mean(np.stack(v), axis=0) for k, v in acc.items() if v}


def get_args():
    p = argparse.ArgumentParser("nuScenes open-loop planning eval")
    p.add_argument("--resume_path", required=True, help="trained checkpoint dir")
    p.add_argument("--tokenizer_path", required=True)
    p.add_argument("--records_json", required=True, help="val records from preprocess_nuscenes.py")
    p.add_argument("--norm_path", default=None, help="nuscenes_norm.json used at TRAINING time")
    p.add_argument("--output_dir", default="./results/nuscenes_eval")
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--max_seq_len", type=int, default=4096)
    p.add_argument("--action_dim", type=int, default=2)
    p.add_argument("--time_horizon", type=int, default=6)
    p.add_argument("--decoder", choices=["token", "head"], default="token",
                   help="token = generate_dis_ma (255-bin autoregressive, what every "
                        "number in the paper was produced with). head = the L1 "
                        "regression action head that ships in the checkpoint and has "
                        "never been evaluated. Both regress onto the SAME quantised "
                        "target, so a difference is about the estimator, not about "
                        "removing quantisation. Compare the two arms PAIRED "
                        "(compare_base_ckpts.py joins on sample_token).")
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--load_in_4bit", action="store_true", default=False)
    p.add_argument("--limit", type=int, default=0, help="evaluate only the first N records (0 = all)")
    p.add_argument("--seed", type=int, default=42, help="RNG seed (reproducible sampling)")
    p.add_argument("--legacy_random_crop", action="store_true", default=False,
                   help="reproduce pre-fix numbers: let process_image() random-crop the frame "
                        "on every forward call (training augmentation leaking into eval). "
                        "Default is a deterministic centre crop -- see got_drive/eval_crop.py. "
                        "Never mix the two inside one table.")
    p.add_argument("--train_records_json", default=None,
                   help="records to compute the mean-trajectory baseline from (ideally the TRAIN "
                        "split). If unset, the eval split is used (slightly optimistic).")
    return p.parse_args()


def load_model(args):
    common = dict(
        action_dim=args.action_dim,
        time_horizon=args.time_horizon,
        max_position_embeddings=args.max_seq_len,
        mask_image_logits=True,
        dropout=0.0,
        z_loss_weight=0.0,
    )
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig
        model = ChameleonXLLMXForConditionalGeneration_ck_action_head.from_pretrained(
            args.resume_path,
            quantization_config=BitsAndBytesConfig(load_in_4bit=True),
            device_map={"": args.device},
            **common,
        )
    else:
        model = ChameleonXLLMXForConditionalGeneration_ck_action_head.from_pretrained(
            args.resume_path,
            torch_dtype=torch.bfloat16,
            **common,
        ).to(torch.device(f"cuda:{args.device}"))
    model.eval()
    return model


def unnorm_waypoints(norm_wp, wp_min, wp_max):
    """[-1, 1] bin space -> metres. Inverse of item_processor.norm_action."""
    return (norm_wp + 1) / 2 * (wp_max - wp_min + 1e-8) + wp_min


@torch.no_grad()
def predict_waypoints(model, item_processor, image, prompt, args):
    """Returns (n_future, 2) waypoints in metres, or None if generation was malformed."""
    conv = {
        "conversations": [{"from": "human", "value": prompt + "<|image|>"}],
        "image": [image],
        "action": [],
    }
    tokens = item_processor.process_item(conv, training_mode=False)
    input_ids = torch.tensor(tokens, dtype=torch.int64, device=model.device).unsqueeze(0)

    # each waypoint emits [start, x, y, end]; leave headroom for a stray token
    max_new = args.time_horizon * (args.action_dim + 2) + 8
    generation_config = GenerationConfig(
        max_new_tokens=max_new,
        max_length=model.config.max_position_embeddings,
        temperature=1.0,
        top_k=None,
        do_sample=False,
        eos_token_id=[8710],
    )

    # list of per-group tensors, values already decoded to bin centers in [-1, 1].
    #
    # generate_dis_ma parses the token stream with an uninitialized `start_index`,
    # so a stream whose first action marker is an end token (15004) before any
    # start token (10004) raises UnboundLocalError instead of returning []. An
    # undertrained model emits exactly that kind of malformed stream, which is the
    # normal case for a mini smoke test, so treat any parse failure as a malformed
    # generation rather than letting it kill the whole eval run. The LIBERO eval
    # solver takes the same approach (broad except -> count the episode as failed).
    try:
        groups = model.generate_dis_ma(input_ids, generation_config)
    except Exception:
        return None

    if len(groups) != args.time_horizon:
        return None
    norm_wp = []
    for g in groups:
        g = g.detach().float().cpu().numpy()
        if g.shape[0] != args.action_dim:
            return None
        norm_wp.append(g)
    norm_wp = np.stack(norm_wp, axis=0)  # (n_future, action_dim)

    return unnorm_waypoints(norm_wp, item_processor.wp_min, item_processor.wp_max)


@torch.no_grad()
def predict_waypoints_head(model, item_processor, image, prompt, args):
    """Same contract as predict_waypoints, but decoded by the CONTINUOUS action head.

    The checkpoint carries an L1-regression head (`loss_ct`) that has never been used
    at evaluation: eval has always gone through generate_dis_ma (255-bin AR tokens).
    This is the other decoder for the same weights -- one forward, no autoregression,
    so it is FASTER than the discrete path rather than slower.

    !! WHAT THIS DOES AND DOES NOT TEST (read before interpreting the number).
    The head's training target is
        labels_action_ct = decode_token_ids_to_actions(labels_action_dis)
    i.e. the BIN CENTRES of the same 255-bin grid, not the true waypoints. The head
    therefore regresses onto the quantised label and CANNOT recover the quantisation
    error (RMS 0.065 m); both decoders sit on the same grid. What differs is the
    predictor -- token cross-entropy + argmax against L1 regression -- so read any
    change as "a different estimator of the same target", never as "continuous
    representation removes quantisation noise".

    Returns metres, or None when the head could not be applied.
    """
    conv = {
        "conversations": [{"from": "human", "value": prompt + "<|image|>"}],
        "image": [image],
        "action": [],
    }
    tokens = item_processor.process_item(conv, training_mode=False)
    input_ids = torch.tensor(tokens, dtype=torch.int64, device=model.device).unsqueeze(0)

    # generate_action_head clamps this to a single new token; the values below only
    # have to match the discrete path's sampling settings so the two arms differ in
    # the decoder alone
    generation_config = GenerationConfig(
        max_new_tokens=1,
        max_length=model.config.max_position_embeddings,
        temperature=1.0,
        top_k=None,
        do_sample=False,
        eos_token_id=[8710],
    )

    try:
        norm_wp = model.generate_action_head(input_ids, generation_config)
    except Exception:
        return None
    if norm_wp is None:
        return None

    norm_wp = norm_wp.detach().float().cpu().numpy()
    # the head's horizon comes from the checkpoint, the metric's from --time_horizon;
    # a mismatch must not be reshaped away
    if norm_wp.shape != (args.time_horizon, args.action_dim):
        return None

    return unnorm_waypoints(norm_wp, item_processor.wp_min, item_processor.wp_max)


def l2_metrics(pred, gt, hz_idx):
    """pred/gt: (n_future, 2). hz_idx maps a horizon label -> waypoint index."""
    per_step = np.linalg.norm(pred - gt, axis=-1)  # (n_future,)
    out = {}
    for label, idx in hz_idx.items():
        out[f"L2@{label}"] = float(per_step[idx])
        out[f"avgL2@{label}"] = float(per_step[: idx + 1].mean())
    return out, per_step


def main():
    args = get_args()
    set_seed(args.seed)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    with open(args.records_json) as f:
        records = json.load(f)
    if args.limit:
        records = records[: args.limit]

    # model-free prior baseline: per-command mean GT trajectory
    if args.train_records_json:
        with open(args.train_records_json) as f:
            mean_src = json.load(f)
    else:
        print("[eval][warn] --train_records_json unset; mean-trajectory baseline uses the "
              "eval split itself (slightly optimistic).")
        mean_src = records
    mean_trajs = compute_mean_trajectories(mean_src, args.time_horizon)

    item_processor = FlexARItemProcessor_Action_NuScenes(
        tokenizer=args.tokenizer_path,
        target_size=args.resolution,
        norm_path=args.norm_path,
    )
    print(f"[eval] waypoint un-norm range: min={item_processor.wp_min.tolist()} "
          f"max={item_processor.wp_max.tolist()}")

    model = load_model(args)

    # waypoints are 0.5 s apart: index 1 -> 1 s, 3 -> 2 s, 5 -> 3 s
    hz_idx = {"1s": 1, "2s": 3, "3s": 5}
    hz_idx = {k: v for k, v in hz_idx.items() if v < args.time_horizon}

    rows = []
    all_metrics = []
    n_failed = 0
    n_short_gt = 0             # records with fewer than time_horizon GT waypoints
    per_step_all = []
    base_metrics = []          # mean-trajectory baseline (model-free)

    # started AFTER load_model, so the number is decoding cost per record and not
    # the one-off checkpoint load. sec.7.2's footnote reports s/record next to
    # forward_calls, and this script never recorded it -- the 0.97 and 14.65 in
    # that table both came from eval_got_nuscenes.
    t_start = time.time()

    for i, rec in enumerate(records):
        # Crop ONCE, deterministically. process_image() would otherwise re-crop at a
        # random offset on every forward call (got_drive/eval_crop.py); on an
        # already-cropped frame it is a no-op.
        image = crop_for_eval(Image.open(rec["images"][0]).convert("RGB"),
                              item_processor, args.legacy_random_crop)
        gt = np.array(rec["waypoints"], dtype=np.float64)
        if gt.shape[0] < args.time_horizon:
            # predict_waypoints always returns exactly time_horizon waypoints, so a
            # short GT would raise a broadcast error inside l2_metrics and kill the
            # whole run hours in. Skip the record and surface the count instead.
            n_short_gt += 1
            continue
        gt_h = gt[: args.time_horizon]

        # model-free baseline: mean GT trajectory for this record's command.
        # Evaluated on EVERY record (independent of whether the model succeeds).
        mean_pred = mean_trajs.get(rec.get("command"), mean_trajs.get("__all__"))
        if mean_pred is not None:
            bm, _ = l2_metrics(mean_pred, gt_h, hz_idx)
            base_metrics.append(bm)

        decode = predict_waypoints_head if args.decoder == "head" else predict_waypoints
        pred = decode(model, item_processor, image, args.prompt, args)
        if pred is None:
            n_failed += 1
            rows.append({"sample_token": rec["sample_token"], "scene": rec["scene"],
                         "command": rec["command"], "status": "malformed_generation"})
            continue

        m, per_step = l2_metrics(pred, gt_h, hz_idx)
        all_metrics.append(m)
        per_step_all.append(per_step)
        rows.append({
            "sample_token": rec["sample_token"], "scene": rec["scene"],
            "command": rec["command"], "status": "ok",
            **{k: round(v, 4) for k, v in m.items()},
            "pred": np.round(pred, 3).tolist(), "gt": np.round(gt_h, 3).tolist(),
        })

        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(records)}  (failed={n_failed})")

    # ---- summary ----
    csv_path = os.path.join(args.output_dir, "per_sample.csv")
    if rows:
        keys = sorted({k for r in rows for k in r})
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)

    elapsed = time.time() - t_start
    summary = {
        "seed": args.seed,
        "decoder": args.decoder,
        "n_records": len(records),
        "n_evaluated": len(all_metrics),
        "n_malformed_generation": n_failed,
        "n_skipped_short_gt": n_short_gt,
        "elapsed_s": round(elapsed, 1),
        "s_per_record": round(elapsed / max(len(records), 1), 4),
    }
    if all_metrics:
        for k in all_metrics[0]:
            summary[k] = round(float(np.mean([m[k] for m in all_metrics])), 4)
        summary["per_step_L2"] = np.round(np.mean(np.stack(per_step_all), axis=0), 4).tolist()

    # mean-trajectory baseline: the model must clearly beat this to prove it uses
    # the image (nuScenes ego-extrapolation pitfall).
    if base_metrics:
        summary["baseline_mean_traj"] = {
            "n_evaluated": len(base_metrics),
            **{k: round(float(np.mean([m[k] for m in base_metrics])), 4) for k in base_metrics[0]},
        }

    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== nuScenes open-loop planning ===")
    print(json.dumps(summary, indent=2))
    print(f"\nper-sample -> {csv_path}")
    if n_failed:
        print(f"WARNING: {n_failed}/{len(records)} generations were malformed "
              f"(wrong number of action groups) and are EXCLUDED from the averages above.")


if __name__ == "__main__":
    main()
