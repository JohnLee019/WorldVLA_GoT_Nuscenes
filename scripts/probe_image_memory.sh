#!/usr/bin/env bash
# --- always run from the repo root ---------------------------------------
cd "$(dirname "$0")/.." || exit 1
# -------------------------------------------------------------------------
#
# How many camera images fit in a training step? MEASURE it, do not estimate.
#
# FSDP shards parameters, gradients and the optimizer across the 3 GPUs, but NOT
# activations: every rank runs the whole model on its own microbatch, so a second
# image lengthens the sequence on all three ranks equally. Roughly 13 of the
# 20,148 MB per rank (sec.6.2) is that fixed sharded cost and does not move with the
# image count -- only the activation slice does, and how big that slice is has
# never been measured here.
#
# This duplicates CAM_FRONT N times per record (same pixels, N times the image
# tokens) and runs a few real training iterations in the real regime
# (--trainable full, 3-GPU FSDP, batch 2). The point is the `max mem` field of
# the first log line. Ctrl-C once it appears.
#
#   N=1 bash scripts/probe_image_memory.sh    # sanity: should land near 20,148
#   N=2 bash scripts/probe_image_memory.sh
#   N=3 bash scripts/probe_image_memory.sh
#
# NOTE this measures MEMORY ONLY. Whether more cameras are worth having is a
# separate, already-answered question (sec.8): the collision oracle ceiling is 23%,
# and side/rear views do not observe ego speed, which is 88% of the error.
set -euo pipefail

# ★Fragmentation, not activations, is what kills this at N=2. The first OOM here
# reported 19.04 GiB actually allocated but 3.60 GiB "reserved but unallocated" --
# the caching allocator holding blocks it cannot reuse. The trigger is variable
# sequence length: var_center_crop draws a different crop size per sample, so
# tensor shapes change every iteration, and N images multiply that variation.
# expandable_segments lets the allocator grow a segment instead of stranding it.
# This changes allocation only, never numerics.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ⚠️Read the max mem of a LATER iteration, not iteration 0. Peak varies with the
# drawn crop size, and iteration 0 under-reports: the run that OOMed had logged
# `max mem: 19763` one iteration earlier. MetricLogger prints every 10 iterations,
# so let it reach [10/32] (~3 min) before judging.
N=${N:-2}
RECS=./data/nuscenes_records_state
OUT=./output/imgmem_probe_n$N
SMOKE=$RECS/imgmem_n$N.json

python - <<PY
import json
n = $N
src = json.load(open("$RECS/nuscenes_v1.0-trainval_train.json"))
sub = src[::max(1, len(src)//200)][:200]
for r in sub:
    r["images"] = r["images"] * n          # same frame, n times -> n x image tokens
json.dump(sub, open("$SMOKE", "w"))
print(f"[probe] {len(sub)} records x {n} images "
      f"(~{240*n} image tokens/record, max_seq_len is 4096)")
PY

rm -rf "$OUT"
torchrun --nproc_per_node=3 train_nuscenes.py \
  --resume_path ./output/nuscenes_trainval_full_r256_cont2/epoch1 \
  --tokenizer_path ../ckpts/Lumina-mGPT-7B-768 \
  --data_config_train "$SMOKE" \
  --data_config_val_ind "$SMOKE" --data_config_val_ood "$SMOKE" \
  --norm_path $RECS/nuscenes_norm.json \
  --output_dir "$OUT" \
  --trainable full --optimizer paged_adamw8bit --with_state true \
  --batch_size 2 --accum_iter 32 --resolution 256 --grad_precision bf16 \
  --num_workers 2 \
  --save_iteration_interval 1000000 --ckpt_max_keep 1 \
  --epochs 1 --lr 2e-5 --precision bf16 --checkpointing --ft true

# ★--accum_iter 32 exists to PUSH THE CHECKPOINT SAVE TO THE END, not to change
# the effective batch.
#
# sec.6.6: the epoch0 save at `data_iter_step + 1 == accum_iter` cannot be switched
# off. At the default 4 it fires on micro-step 4, ~90 s in -- and that killed an
# earlier N=2 probe: `exitcode -9` (SIGKILL by the kernel OOM killer, anon-rss
# 11.7 GB, NOT a CUDA OOM) on RANK 0, the rank where util.ckpt.save gathers the
# whole 7B into host RAM. Moving it to micro-step 32 of 33 means every `max mem`
# line we care about is printed first.
#
# ⚠️Do NOT raise it above the micro-step count (200 records / (2 x 3 GPUs) = 33).
# Tried 9999: the run finishes all 33 iterations and then dies with
# `ZeroDivisionError` in the epoch summary, because metrics that only update on a
# gradient-accumulation boundary end with count 0. The max mem lines are still
# printed before that, but the exit is confusing.
