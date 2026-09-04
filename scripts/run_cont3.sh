#!/usr/bin/env bash

# --- 리포 루트에서 실행되도록 고정 -----------------------------------------
# 이 스크립트는 2026-08-21에 루트에서 scripts/ 로 옮겨졌다. 안의 경로가 전부
# 루트 기준(results/, data/, eval_*.py)이라, 어디서 호출되든 루트로 이동한다.
cd "$(dirname "$0")/.." || exit 1
# -------------------------------------------------------------------------
# E1: continue base training from _cont2/epoch1  (3 GPUs, ~16 h/epoch, 2 epochs)
#
# WHY a new output_dir: every number in PROJECT_HANDOFF §1 sits on
# _cont2/epoch1. This run must not be able to overwrite it. Adopt the new
# checkpoint only if eval_nuscenes L2 on the 600-record scene-spread set says so.
#
# Read before running:
#   §6.4  resume MUST be --ft true (weights only) or the 8-bit optimizer state
#         loads onto the GPUs and the run OOMs at 23 GB.
#   §6.6  ckpt_max_keep shares its rotation slots with iteration saves, which is
#         how _cont2/epoch0 and the WM's epoch0 were destroyed. Iteration saving
#         is pushed out of range below so only the 2 epoch checkpoints rotate.
#   §9    NEVER judge this by val closs. Measured: val closs 1.3749 -> ~1.43
#         (worse) while L2 improved 16.6%. Judge by eval_nuscenes L2 only.
#   §9    Do not start an eval while this runs: 3-GPU FSDP sits at 19.6/24 GB and
#         a second 7 B model makes TRAINING OOM, not the eval.
#
# lr: 1e-5, half of what produced _cont2. --ft resets Adam momentum and
# re-warms the lr, so hitting a converged model with the original 2e-5 risks an
# initial degradation. If epoch0 comes back WORSE than the current checkpoint,
# lr is the first thing to halve again.

set -euo pipefail
# (no cd here: the header already cd'd to the repo root, and every path below is
# root-relative. Re-entering scripts/ broke all of them -- see handoff sec.9.)

SRC=./output/nuscenes_trainval_full_r256_cont2/epoch1
OUT=./output/nuscenes_trainval_full_r256_cont3
REC=./data/nuscenes_records
VAL_OOD=$REC/nuscenes_val_stride200.json

[ -d "$SRC" ] || { echo "[fatal] source checkpoint missing: $SRC"; exit 1; }
[ -e "$OUT" ] && { echo "[fatal] $OUT exists -- pick a new name, do not resume into it"; exit 1; }

# The solver always builds an OOD validation set, and passing the full val.json
# to both _ind and _ood runs epoch validation TWICE (~2.5 h wasted per epoch,
# §9). A uniform stride slice keeps the signal and drops the duplicate cost.
if [ ! -f "$VAL_OOD" ]; then
  echo "[setup] building $VAL_OOD (uniform stride, 200 records)"
  python - "$REC/nuscenes_v1.0-trainval_val.json" "$VAL_OOD" <<'EOF'
import json, sys
src, dst = sys.argv[1], sys.argv[2]
recs = json.load(open(src))
step = max(1, len(recs) // 200)
out = recs[::step][:200]
json.dump(out, open(dst, "w"))
print(f"  {len(recs)} -> {len(out)} records, stride {step}")
EOF
fi

nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
echo "[check] all three cards must be free. Ctrl-C now if another job is on them."
sleep 5

torchrun --nproc_per_node=3 train_nuscenes.py \
  --resume_path "$SRC" --ft true \
  --tokenizer_path ../ckpts/Lumina-mGPT-7B-768 \
  --data_config_train $REC/nuscenes_v1.0-trainval_train.json \
  --data_config_val_ind $REC/nuscenes_v1.0-trainval_val.json \
  --data_config_val_ood "$VAL_OOD" \
  --norm_path $REC/nuscenes_norm_v1.0-trainval.json \
  --output_dir "$OUT" \
  --trainable full --optimizer paged_adamw8bit \
  --batch_size 2 --accum_iter 4 --resolution 256 --grad_precision bf16 \
  --save_iteration_interval 1000000 --ckpt_max_keep 4 \
  --epochs 2 --lr 1e-5 --precision bf16 --checkpointing
