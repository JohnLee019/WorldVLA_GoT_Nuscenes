#!/usr/bin/env bash
# --- always run from the repo root ---------------------------------------
cd "$(dirname "$0")/.." || exit 1
# -------------------------------------------------------------------------
#
# E8: the ego-status arm (sec.1.15). Trains the incumbent for 2 more epochs with a
# causal ego-status channel, then evaluates it on the frozen 600-record set.
#
# WHY RESUME FROM THE INCUMBENT RATHER THAN THE LUMINA BASE
#   sec.1.13 already ran the exact control this design needs: 2 more epochs from
#   `_cont2/epoch1` with NOTHING changed gave +0.0494 (p_sc 0.5561) -- a null.
#   So training 2 more epochs from the same checkpoint with ONLY the state
#   channel added attributes any movement to the channel, not to more training.
#   Starting from the raw Lumina base would instead re-run the whole recipe and
#   confound "state" with "a different training trajectory".
#
# THE DECISION RULE, FIXED HERE BEFORE THE NUMBERS ARRIVE
#   judge on avgL2@3s over data/nuscenes_records/nuscenes_val_scenespread.json
#   (600 records / 150 scenes), the set every number in sec.1 uses.
#     incumbent bar ......... 3.5557
#     sec.1.15 ceiling ...... ~2.02  (-1.54, ci_sc [-1.79, -1.28])
#   Pre-registered readings:
#     <= 2.5   the channel works and lands near its measured ceiling -> report
#     2.5-3.2  it works but far under ceiling -> the model is underusing the
#              channel; do NOT quietly accept, investigate before reporting
#     >  3.4   null. The plumbing is verified (16/16), so a null here means the
#              model is ignoring the channel, not that the wiring failed
#   sec.1.10(b) already settled the other question: the same information given to
#   the SELECTOR is +0.0205 (ns). This run cannot make GoT beat greedy and is
#   not evidence about that. It is an ABSOLUTE-PERFORMANCE result.
#
# WHAT MUST NOT HAPPEN
#   * do NOT merge these numbers into the sec.1 / sec.7.2 tables. Different setup,
#     no paired comparison. They live in results/egostate/.
#   * do NOT report this without the constant-velocity baseline (sec.7.2 "보류"):
#     with ego status in the input, the mean-trajectory gate no longer shows the
#     model uses the image, because ego status alone clears it.

set -euo pipefail

# Allocator-only setting, no effect on numerics. sec.6.2 measured 20,148 / 24,576 MB
# per rank for this exact configuration, i.e. ~4 GB of headroom -- and the N=2
# image-memory probe showed this workload can strand 3.6 GB in reserved-but-
# unallocated blocks, because var_center_crop varies the sequence length every
# iteration. That is enough to lose a 35 h run at hour 30 for no reason.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --num_workers 4 (default 8) is HOST-RAM insurance, not a speed setting.
# The box has 31 GB, not the 62 GB sec.4 claimed for months. ckpt.py gathers the whole
# 7B onto rank 0's host RAM to save (~14 GB in bf16), and workers are per-rank, so
# the default 8 means 24 image-decoding processes competing with that gather. The
# N=2 probe died exactly there: kernel OOM killer, rank 0, anon-rss 11.7 GB, at the
# epoch0-iter3 save 98 s in. Cost of halving workers is ~nil -- this job is
# comm-bound (sec.6.2) and `data:` was 0.52 s of a 21 s iteration.

REC=./data/nuscenes_records
RECS=./data/nuscenes_records_state
CKPT_IN=./output/nuscenes_trainval_full_r256_cont2/epoch1
OUT=./output/nuscenes_state_r256
TOK=../ckpts/Lumina-mGPT-7B-768

# --- gate: the incumbent must still reproduce 3.5557 on this grid ----------
# sec.9: a mini-fitted norm json silently replaced the trainval action grid once and
# the incumbent scored 3.7889 instead. Refuse to burn 35 h on an unverified grid.
python - <<'PY'
import json, sys
n = json.load(open("./data/nuscenes_records_state/nuscenes_norm.json"))
want_min, want_max = [-3.0241, -16.6451], [69.786, 16.6451]
if [round(v, 4) for v in n["min"]] != want_min or [round(v, 4) for v in n["max"]] != want_max:
    sys.exit(f"norm grid is not the incumbent's:\n  {n['min']} {n['max']}\n  expected {want_min} {want_max}")
if "state_min" not in n:
    sys.exit("norm json carries no state ranges -- rerun data/preprocess_nuscenes.py")
print(f"[gate] action grid matches the incumbent; state dims {len(n['state_min'])}")
PY

# --- the frozen 600-record eval set, with state attached -------------------
# The set in sec.1 predates the state field. Rebuild it by joining on
# sample_token so the eval set stays byte-identical in records and ordering --
# rebuilding it from scratch would risk a different 600.
EVAL_STATE=$RECS/nuscenes_val_scenespread_state.json
python - <<PY
import json
frozen = json.load(open("$REC/nuscenes_val_scenespread.json"))
state = {r["sample_token"]: r for r in json.load(open("$RECS/nuscenes_v1.0-trainval_val.json"))}
out, missing = [], 0
for r in frozen:
    s = state.get(r["sample_token"])
    if s is None:
        missing += 1
        continue
    assert s["waypoints"] == r["waypoints"], f"waypoints moved for {r['sample_token']}"
    out.append({**r, "state": s["state"], "state_valid": s["state_valid"]})
assert missing == 0, f"{missing} frozen records have no state counterpart"
assert len(out) == len(frozen) == 600, f"{len(out)} != 600"
json.dump(out, open("$EVAL_STATE", "w"))
print(f"[eval set] {len(out)} records, "
      f"{sum(1 for r in out if not r['state_valid'])} with a zeroed state -> $EVAL_STATE")
PY

# --- train ----------------------------------------------------------------
# Same knobs as sec.6.3 (batch 2 x accum 4 x 3 GPUs, ~16 h/epoch) plus --with_state.
# --ft true is mandatory with --resume_path (sec.6.4: weights-only, no optimizer
# state, otherwise the 8bit optimizer state pushes 19.6 -> 23 GB and OOMs).
torchrun --nproc_per_node=3 train_nuscenes.py \
  --resume_path "$CKPT_IN" --tokenizer_path "$TOK" \
  --data_config_train  $RECS/nuscenes_v1.0-trainval_train.json \
  --data_config_val_ind $RECS/nuscenes_v1.0-trainval_val.json \
  --data_config_val_ood $RECS/nuscenes_v1.0-trainval_val.json \
  --norm_path $RECS/nuscenes_norm.json \
  --output_dir "$OUT" \
  --trainable full --optimizer paged_adamw8bit --with_state true \
  --batch_size 2 --accum_iter 4 --resolution 256 --grad_precision bf16 \
  --num_workers 4 \
  --save_iteration_interval 1000000 --ckpt_max_keep 4 \
  --epochs 2 --lr 2e-5 --precision bf16 --checkpointing --ft true

# --- eval both epochs -----------------------------------------------------
# sec.9: never run an eval next to a 3-GPU FSDP job -- it OOMs the training. This
# runs after training exits, so it is safe.
for ep in 0 1; do
  python eval_nuscenes.py \
    --resume_path "$OUT/epoch$ep" --tokenizer_path "$TOK" \
    --records_json "$EVAL_STATE" --with_state \
    --norm_path $RECS/nuscenes_norm.json \
    --train_records_json $RECS/nuscenes_v1.0-trainval_train.json \
    --output_dir ./results/egostate/state_ep$ep \
    --resolution 256 --device 0 --seed 42
done

echo
echo "incumbent bar: avgL2@3s 3.5557   |   sec.1.15 ceiling ~2.02"
python -c "
import json
for ep in (0, 1):
    d = json.load(open(f'./results/egostate/state_ep{ep}/summary.json'))
    print(f\"  epoch{ep}: avgL2@3s {d['avgL2@3s']:.4f}  \"
          f\"malformed {d['n_malformed_generation']}  mean_traj {d['baseline_mean_traj']['avgL2@3s']:.4f}\")
"
