#!/usr/bin/env bash

# --- 리포 루트에서 실행되도록 고정 -----------------------------------------
# 이 스크립트는 2026-08-21에 루트에서 scripts/ 로 옮겨졌다. 안의 경로가 전부
# 루트 기준(results/, data/, eval_*.py)이라, 어디서 호출되든 루트로 이동한다.
cd "$(dirname "$0")/.." || exit 1
# -------------------------------------------------------------------------
# NAVSIM navtrain base training (3 GPUs). The pivot's training run.
#
# Records come from data/preprocess_navsim.py: 151,778 scenarios -> 143,916
# train / 7,862 val, split by LOG (session 15). Schema is identical to the
# nuScenes one, so dataset_nuscenes.py and item_processor.py run unchanged.
#
# WHAT IS DIFFERENT FROM run_cont3.sh, and why
#
#   1. --time_horizon 8   (4 s at 0.5 s, handoff sec.8)
#      This RESHAPES the action head, and train_nuscenes.py already handles it:
#      _model_func passes ignore_mismatched_sizes=True, which is exactly how the
#      original port went from (7 dim x 5 steps) to (2 dim x 6 steps). The
#      backbone loads; the head is re-initialised and learns from scratch.
#      No code change is needed -- verified by reading _model_func, not assumed.
#
#   2. --norm_path is the RECOMPUTED one. 4 s at highway speed reaches ~65 m
#      where nuScenes' 3 s reached ~45 m, so reusing nuscenes_norm.json would
#      clip the action tokenizer's range (Step 2.12a).
#
#   3. Validation is a SLICE, not the full val split. On nuScenes a full-val
#      pass cost ~2.5 h per epoch (sec.9) and here val is 7,862 records, so a
#      full pass would be ~4 GPU-h. It would also be spent on the wrong thing:
#      THIS RUN IS NOT JUDGED BY VAL LOSS. The judgment is PDMS from
#      run_pdm_score.py on navtest (Step 2.13c). The slices exist to draw a
#      sanity curve and to catch divergence, nothing more.
#      ⚠️ val_ood is VESTIGIAL here: NAVSIM gives us no separate OOD split, so
#      both slices come from the same held-out logs. Do not read _ood as OOD.
#
# COST, measured not guessed. E1 ran at 14.77 s/iter with 6 records per iter
# (3 GPUs x batch 2). 143,916 / 6 = 23,986 iter, so ONE epoch is ~98 h = 4.1
# days. That is why --epochs defaults to 1: one navtrain epoch already shows
# the model 143,916 DISTINCT records, about 6x a nuScenes epoch, and E1 just
# measured that this model converges quickly on this task (sec.1.13, 2 extra
# epochs = null). Seeing new data once beats seeing old data twice.
#
# CRASH INSURANCE (this is the real reason for the iteration saves). A 4-day
# run with checkpoints only at the end is one power cut away from losing
# everything. sec.6.6 warns that iteration saves share rotation slots with
# epoch saves -- that is what destroyed _cont2/epoch0 -- so the interval and
# ckpt_max_keep are chosen together here: a save every 5,000 iters (~20 h) and
# 3 slots means there is always a recent restart point and the disk holds at
# most ~81 GB of this run.
#
# LR 2e-5, not run_cont3.sh's 1e-5. That halving was for a CONVERGED model with
# an intact head, where --ft's lr re-warm risked degrading what was already
# there (E1's epoch0 came back worse than epoch1, which is that risk showing).
# Here the action head is re-initialised and has to learn from nothing, and the
# data distribution is new, so the original 2e-5 is the right starting point.
# If the first iteration checkpoint looks worse than the source on PDMS, LR is
# the first thing to halve.
#
# Overridable: SRC=... EPOCHS=... LR=... bash run_navsim.sh

set -euo pipefail
cd "$(dirname "$0")"

# WARM START from the nuScenes model by default. Same task family, same image
# tokenizer, same action_dim; only the head shape differs and that is re-init.
# Set SRC=../ckpts/Lumina-mGPT-7B-768 to train from the original base instead.
SRC=${SRC:-./output/nuscenes_trainval_full_r256_cont2/epoch1}
OUT=${OUT:-./output/navsim_navtrain_r256}
REC=./data/navsim_records
EPOCHS=${EPOCHS:-1}
LR=${LR:-2e-5}

TRAIN=$REC/navsim_navtrain_train.json
VAL=$REC/navsim_navtrain_val.json
NORM=$REC/navsim_norm.json
VAL_IND=$REC/navsim_val_slice500.json
VAL_OOD=$REC/navsim_val_slice200.json

for f in "$TRAIN" "$VAL" "$NORM"; do
  [ -f "$f" ] || { echo "[fatal] missing $f -- run 'python -m data.preprocess_navsim' first"; exit 1; }
done
[ -d "$SRC" ] || [ -f "$SRC/config.json" ] || { echo "[fatal] source checkpoint missing: $SRC"; exit 1; }
[ -e "$OUT" ] && { echo "[fatal] $OUT exists -- pick a new name, do not resume into it"; exit 1; }

# The norm file decides what the action tokenizer can represent at all. If it
# was built for a 6-step horizon the run would train on silently clipped
# targets, so this is checked rather than trusted.
python - "$NORM" <<'EOF'
import json, sys
n = json.load(open(sys.argv[1]))
if int(n.get("n_future", 0)) != 8:
    sys.exit(f"[fatal] {sys.argv[1]} has n_future={n.get('n_future')}, expected 8. "
             "Rebuild with data/preprocess_navsim.py --n_future 8.")
print(f"[check] norm ok: n_future=8, x={n['min'][0]:.1f}..{n['max'][0]:.1f} m, "
      f"y=+/-{n['max'][1]:.1f} m")
EOF

for spec in "$VAL_IND 500" "$VAL_OOD 200"; do
  set -- $spec
  if [ ! -f "$1" ]; then
    echo "[setup] building $1 ($2 records, uniform stride)"
    python - "$VAL" "$1" "$2" <<'EOF'
import json, sys
src, dst, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
recs = json.load(open(src))
step = max(1, len(recs) // n)
out = recs[::step][:n]
json.dump(out, open(dst, "w"))
print(f"  {len(recs)} -> {len(out)} records, stride {step}")
EOF
  fi
done

python - "$TRAIN" <<'EOF'
import json, sys
recs = json.load(open(sys.argv[1]))
iters = len(recs) // 6            # 3 GPUs x batch 2, measured on E1
print(f"[plan] {len(recs)} train records -> ~{iters} iter/epoch "
      f"-> ~{iters * 14.77 / 3600:.0f} h/epoch at E1's measured 14.77 s/iter")
EOF

echo "[plan] SRC=$SRC"
echo "[plan] OUT=$OUT  EPOCHS=$EPOCHS  LR=$LR"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
echo "[check] all three cards must be free. Ctrl-C now if another job is on them."
sleep 10

torchrun --nproc_per_node=3 train_nuscenes.py \
  --resume_path "$SRC" --ft true \
  --tokenizer_path ../ckpts/Lumina-mGPT-7B-768 \
  --data_config_train "$TRAIN" \
  --data_config_val_ind "$VAL_IND" \
  --data_config_val_ood "$VAL_OOD" \
  --norm_path "$NORM" \
  --output_dir "$OUT" \
  --action_dim 2 --time_horizon 8 \
  --trainable full --optimizer paged_adamw8bit \
  --batch_size 2 --accum_iter 4 --resolution 256 --grad_precision bf16 \
  --save_iteration_interval 5000 --ckpt_max_keep 3 \
  --epochs "$EPOCHS" --lr "$LR" --precision bf16 --checkpointing
