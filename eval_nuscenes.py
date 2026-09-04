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
        --norm_path ./data/nuscenes_records/nuscenes_norm_v1.0-trainval.json \
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
from transformers import GenerationConfig, LogitsProcessor, LogitsProcessorList

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
    p.add_argument("--constrained", action="store_true", default=False,
                   help="mask the logits to the [start, dims..., end] action grammar so ANY "
                        "checkpoint emits a well-formed trajectory. Use it to score a model "
                        "that was never trained on this token grammar (a raw backbone), where "
                        "free generation returns 100%% malformed. Discrete decoder only. "
                        "★A constrained run is a DIFFERENT ARM -- compare it to the model-free "
                        "baselines, not to the free-generation numbers.")
    p.add_argument("--legacy_random_crop", action="store_true", default=False,
                   help="reproduce pre-fix numbers: let process_image() random-crop the frame "
                        "on every forward call (training augmentation leaking into eval). "
                        "Default is a deterministic centre crop -- see got_drive/eval_crop.py. "
                        "Never mix the two inside one table.")
    p.add_argument("--with_state", action="store_true",
                   help="feed causal ego status (<|state|>) alongside the image. MUST match "
                        "how the checkpoint was trained -- a state-trained checkpoint "
                        "evaluated without it (or vice versa) is off-distribution and the "
                        "L2 is meaningless. Records need a `state` field "
                        "(data/preprocess_nuscenes.py) and the norm json needs `state_min`.")
    p.add_argument("--allow_mini_grid", action="store_true",
                   help="run even when --norm_path is the v1.0-mini action grid. ★Never right "
                        "for a new number -- the incumbent scores 3.7889 instead of 3.5557 on it "
                        "and nothing else fails (§9). Exists only to reproduce a historical "
                        "mini-grid run on purpose.")
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


# The action grid the incumbent (`_cont2/epoch1`) was TRAINED on -- a trainval fit.
# Stated four times in the handoff (sec.1.15, sec.1.16f, sec.9) and asserted by
# scripts/run_state.sh's pre-flight gate.
INCUMBENT_GRID = ([-3.0241, -16.6451], [69.786, 16.6451])
# The v1.0-mini fit that was once written over the norm json on disk. Evaluating
# the incumbent on it does not crash -- it returns avgL2@3s 3.7889 instead of
# 3.5557 (sec.9). A plausible wrong number is the whole danger.
MINI_GRID = ([-1.0193, -15.1637], [56.432, 15.1637])


def check_action_grid(item_processor, norm_path, allow_mini=False):
    """Say out loud which action grid this run is on, and REFUSE if it is the mini one.

    ⚠️★IT REFUSES, IT DOES NOT MERELY WARN -- and that is a correction to this
    guard's own first version. An independent measurement audit caught it
    `print()`-ing and continuing, while every real run is launched as
    `... > run.log`, where a warning scrolls past unread hours before anyone opens
    the summary. That is sec.9's lesson landing on the fix written to satisfy it:
    *"'가드를 넣었다'와 '가드가 이 파일에 걸린다'는 다른 진술이다."* A guard the
    operator never sees is not a guard. `--allow_mini_grid` exists only to
    reproduce a historical mini-grid run on purpose; it is never right for a new
    number.

    WHY THIS EXISTS. The norm filename carries no version but the record filenames
    do, so `preprocess_nuscenes.py --version v1.0-mini` (both defaults!) replaces
    the action grid while leaving the trainval records alone. It happened, and it
    cost two sessions: sec.1.15 inherited the mini grid and reported a 3.7889
    regression that was not a regression. The overwrite guard added afterwards only
    fires on files that already carry a `version`, and the file that caused the
    accident predated the field (sec.9, session 20).

    So this is the LAST line of defence, on the consumer side, where it cannot be
    slept through: the grid is printed every run and named when it is known-wrong.
    """
    lo = [round(v, 4) for v in item_processor.wp_min.tolist()]
    hi = [round(v, 4) for v in item_processor.wp_max.tolist()]
    if (lo, hi) == (MINI_GRID[0], MINI_GRID[1]):
        print(f"\n[grid] !! THIS IS THE v1.0-mini ACTION GRID ({norm_path}).\n"
              f"[grid]    The incumbent checkpoint was trained on the trainval fit\n"
              f"[grid]    min={INCUMBENT_GRID[0]} max={INCUMBENT_GRID[1]}.\n"
              f"[grid]    Evaluating it here yields avgL2@3s ~3.7889 instead of 3.5557 and\n"
              f"[grid]    NOTHING ELSE WILL FAIL. Point --norm_path at the trainval norm json\n"
              f"[grid]    (data/nuscenes_records/nuscenes_norm_v1.0-trainval.json) unless you\n"
              f"[grid]    are deliberately reproducing the mini-grid run.\n")
        if not allow_mini:
            raise SystemExit(
                "refusing to run on the v1.0-mini action grid. This is not a "
                "warning you can scroll past: the incumbent scores 3.7889 instead "
                "of 3.5557 on it and NOTHING ELSE FAILS (sec.9). Point --norm_path "
                "at data/nuscenes_records/nuscenes_norm_v1.0-trainval.json, or pass "
                "--allow_mini_grid if you are deliberately reproducing an old run.")
    elif (lo, hi) == (INCUMBENT_GRID[0], INCUMBENT_GRID[1]):
        print(f"[grid] trainval fit -- matches the incumbent's training grid")
    else:
        print(f"[grid] non-standard grid min={lo} max={hi} -- neither the incumbent's "
              f"trainval fit nor the known mini fit. Make sure this matches the checkpoint.")


def build_planning_conv(prompt, image, state):
    """The one place the observation conversation is assembled.

    Training builds this in data/dataset_nuscenes.py; if the two ever disagree on
    placeholder ORDER the same record tokenizes differently at train and eval
    time and the model is evaluated off-distribution without anything failing.
    Keep the two in sync -- image first, then state, both in the human turn.
    """
    human = prompt + "<|image|>"
    conv = {"conversations": None, "image": [image], "action": []}
    if state is not None:
        human += "<|state|>"
        conv["state"] = [list(map(float, state))]
    conv["conversations"] = [{"from": "human", "value": human}]
    return conv


class WaypointGrammar(LogitsProcessor):
    """Force `[start, x_bin, y_bin, end] x time_horizon` onto the generated stream.

    WHY THIS EXISTS. A checkpoint that was never trained on this grammar emits
    none of these tokens -- `<reserved10000>`/`<reserved15000>` and the 256 bin
    slots after them are RESERVED vocabulary entries with no pretrained meaning,
    so a raw backbone has no reason to produce them. `generate_dis_ma` then parses
    zero action groups and every record comes back malformed (measured: 600/600 on
    ../ckpts/Lumina-mGPT-7B-768). That number says the model does not know the
    OUTPUT PROTOCOL. It says nothing about whether the model knows where to drive.

    Masking the logits to the grammar forces a well-formed trajectory out of ANY
    checkpoint, so the resulting L2 measures the prior the model actually holds
    over the waypoint bins -- which is the question "is there any driving
    knowledge in there" actually asks.

    !! READ THE RESULT AGAINST THE MODEL-FREE BASELINES, NOT AGAINST THE
    UNCONSTRAINED RUNS. This is a different decoding rule, so it is a different
    arm. The finetuned model's 3.5557 was produced by free generation; putting a
    constrained number next to it in the same column compares two things that
    differ in more than the checkpoint.

    Token ids are derived from the item processor, not hardcoded: `digitize`
    returns 1..256 over norm in [-1, 1], and process_action adds
    `token2id(action_start_token) + 1`, so the ids the TRAINING data actually
    contained are start_id+2 .. start_id+1+len(bins) (measured: 10006..10261).
    """

    def __init__(self, prompt_len, start_id, end_id, value_ids, action_dim):
        self.prompt_len = int(prompt_len)
        self.period = action_dim + 2          # [start, dims..., end]
        self.start_id, self.end_id = int(start_id), int(end_id)
        self.value_ids = list(value_ids)

    def __call__(self, input_ids, scores):
        k = (input_ids.shape[1] - self.prompt_len) % self.period
        if k == 0:
            allowed = [self.start_id]
        elif k == self.period - 1:
            allowed = [self.end_id]
        else:
            allowed = self.value_ids
        out = torch.full_like(scores, float("-inf"))
        idx = torch.as_tensor(allowed, device=scores.device, dtype=torch.long)
        out[:, idx] = scores[:, idx]
        return out


def build_grammar(item_processor, prompt_len, action_dim):
    start_id = item_processor.token2id(item_processor.action_start_token)
    end_id = item_processor.token2id(item_processor.action_end_token)
    n_bins = len(item_processor.bins)
    value_ids = [start_id + 1 + d for d in range(1, n_bins + 1)]
    return LogitsProcessorList([
        WaypointGrammar(prompt_len, start_id, end_id, value_ids, action_dim)])


@torch.no_grad()
def predict_waypoints(model, item_processor, image, prompt, args, state=None):
    """Returns (n_future, 2) waypoints in metres, or None if generation was malformed."""
    conv = build_planning_conv(prompt, image, state)
    tokens = item_processor.process_item(conv, training_mode=False)
    input_ids = torch.tensor(tokens, dtype=torch.int64, device=model.device).unsqueeze(0)

    # each waypoint emits [start, x, y, end]; leave headroom for a stray token.
    # Under --constrained the stream cannot stray, so ask for exactly the grammar's
    # length -- a longer budget would only append a 7th group the parser rejects.
    constrained = getattr(args, "constrained", False)
    max_new = args.time_horizon * (args.action_dim + 2) + (0 if constrained else 8)
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
    lp = None
    if constrained:
        lp = build_grammar(item_processor, input_ids.shape[1], args.action_dim)

    try:
        groups = model.generate_dis_ma(input_ids, generation_config, logits_processor=lp)
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
def predict_waypoints_head(model, item_processor, image, prompt, args, state=None):
    """Same contract as predict_waypoints, but decoded by the CONTINUOUS action head.

    The checkpoint carries an L1-regression head (`loss_ct`) that has never been used
    at evaluation: eval has always gone through generate_dis_ma (255-bin AR tokens).
    This is the other decoder for the same weights -- one forward, no autoregression,
    so it is FASTER than the discrete path rather than slower.

    !! WHAT THIS DOES AND DOES NOT TEST (read before interpreting the number).
    The head's training target is
        labels_action_ct = decode_token_ids_to_actions(labels_action_dis)
    i.e. the BIN CENTRES of the same 255-level grid, not the true waypoints. The head
    therefore regresses onto the QUANTISED LABEL, so it cannot recover the
    quantisation error even though its own outputs are continuous. What differs is
    the predictor -- token cross-entropy + argmax against L1 regression -- so read
    any change as "a different estimator of the same target", never as "continuous
    representation removes quantisation noise".

    That is not an argument, it is measured (`analysis/measure_action_grid.py`):
    this arm's predictions sit 25.1% / 24.6% of a grid step from the nearest level
    (the discrete arm sits at 0.1%, i.e. exactly on it), so the head really is off
    the lattice -- and it still lands at 3.6008 vs the discrete 3.5557 (sec.1.11).
    Leaving the grid bought nothing, which is the cleaner form of the same point.

    !! An earlier version of this docstring put the quantisation error at
    "RMS 0.065 m". That was computed from the mini `nuscenes_norm.json` on disk,
    not the trainval range the checkpoint was trained with (the same overwritten
    file that bit sec.1.15). Measured on the real grid, the round-trip error is
    0.0840 m as a 2-D displacement -- which is the figure commensurable with
    avgL2@3s -- or 0.0770 / 0.0337 m per axis. See sec.1.16(f).

    Returns metres, or None when the head could not be applied.
    """
    conv = build_planning_conv(prompt, image, state)
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

    if args.constrained and args.decoder != "token":
        raise SystemExit(
            "--constrained masks the logits of the autoregressive token stream, which the "
            "continuous head does not produce. Use --decoder token, or drop --constrained.")

    if args.with_state:
        # Refuse to run rather than silently evaluate a state-trained checkpoint
        # on stateless records: nothing downstream would fail, the L2 would just
        # be quietly wrong.
        missing = sum(1 for r in records if "state" not in r)
        if missing:
            raise SystemExit(
                f"--with_state given but {missing}/{len(records)} records carry no "
                f"`state`. Rebuild {args.records_json} with data/preprocess_nuscenes.py."
            )
        n_invalid = sum(1 for r in records if not r.get("state_valid", 1))
        print(f"ego status ON -- {n_invalid}/{len(records)} records have a zeroed "
              f"(scene-start) state")

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
    check_action_grid(item_processor, args.norm_path,
                      allow_mini=args.allow_mini_grid)

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
        pred = decode(model, item_processor, image, args.prompt, args,
                      state=rec["state"] if args.with_state else None)
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
        # ★THE RUN'S OWN CONFIGURATION, IN ITS OWN ARTIFACT.
        # A measurement audit found that summary.json recorded no arm settings at
        # all -- `headline/ref` and `headline/temp_tight` were distinguishable
        # only by directory name. That is exactly why the `--planner aggregate`
        # bug (which silently ran the control arm) could not be detected from the
        # artifacts, and it is the same class as §9's "the runner only overwrites
        # the arm it runs, so an old run stays in the table". Writing argv here
        # closes it: every number now carries the flags that produced it.
        "args": {k: (list(v) if isinstance(v, tuple) else v)
                 for k, v in sorted(vars(args).items())},
        "decoding": "constrained" if args.constrained else "free",
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

    # A constrained run cannot produce a malformed group -- the grammar admits no
    # other token at any position. If one appears, the mask is wrong (or the
    # checkpoint's action_dim/time_horizon disagree with the flags), and the L2
    # above is computed on a silently-selected subset. Say so instead of printing
    # a clean-looking mean.
    if args.constrained:
        if n_failed:
            print(f"\n[fatal-ish] --constrained but {n_failed} generations are still "
                  f"malformed. The grammar admits exactly one token shape, so this means "
                  f"the mask or --action_dim/--time_horizon is wrong. DO NOT quote the "
                  f"numbers above.")
        else:
            print(f"\n[constrained] grammar held on all {len(records) - n_short_gt} records "
                  f"(0 malformed), so this L2 is measured on the FULL set, not a subset.")
            print(f"[constrained] read this against the model-free baselines "
                  f"(mean-trajectory avgL2@3s "
                  f"{summary.get('baseline_mean_traj', {}).get('avgL2@3s', float('nan')):.4f}), "
                  f"NOT against free-generation runs -- different decoding rule, different arm.")


if __name__ == "__main__":
    main()
