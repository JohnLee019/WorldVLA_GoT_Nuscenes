#!/usr/bin/env bash

# --- 리포 루트에서 실행되도록 고정 -----------------------------------------
# 이 스크립트는 2026-08-21에 루트에서 scripts/ 로 옮겨졌다. 안의 경로가 전부
# 루트 기준(results/, data/, eval_*.py)이라, 어디서 호출되든 루트로 이동한다.
cd "$(dirname "$0")/.." || exit 1
# -------------------------------------------------------------------------
# E1 judgement: is _cont3 actually better than the incumbent _cont2/epoch1?
#
# Run this AFTER run_cont3.sh finishes (or after epoch0 lands, if you want the
# curve early -- the checkpoints are written before validation, §9, so epoch0 is
# usable while epoch1 is still training... but NOT on these GPUs: an eval next to
# 3-GPU FSDP makes the TRAINING OOM. Wait, or stop the run.)
#
# The decision rule, fixed here before the numbers arrive:
#   judge on avgL2@3s over the 600-record scene-spread set, the same set every
#   number in §1 uses. NOT val closs (§9: it rose while L2 improved 16.6%).
#   The incumbent's bar on this set is greedy free-run avgL2@3s = 3.5557.
#   Scene-cluster CI half-width is ~0.031, so a gain under ~0.05 m is not a gain.
#
# ADOPTION COSTS, decided before seeing the result so it cannot be rationalised:
#   adopting a new base invalidates every number in §1. Re-run chain:
#     headline 3 seeds ~7.5 GPU-h  +  fusion arms  +  WM main ~6 GPU-h
#   So adopt only on a gain that is worth that, and re-run everything if you do.
#   A smaller-but-real gain is still a RESULT (report it), just not an adoption.

set -euo pipefail
# (no cd here: the header already cd'd to the repo root, and every path below is
# root-relative. Re-entering scripts/ broke all of them -- see handoff sec.9.)

REC=./data/nuscenes_records
SET=$REC/nuscenes_val_scenespread.json          # 600 records / all 150 scenes
GPU=${GPU:-0}

run_one () {                                     # $1 = ckpt dir, $2 = run name
  echo "=== $2  ($1) ==="
  python eval_nuscenes.py \
    --resume_path "$1" \
    --tokenizer_path ../ckpts/Lumina-mGPT-7B-768 \
    --records_json "$SET" \
    --norm_path $REC/nuscenes_norm_v1.0-trainval.json \
    --train_records_json $REC/nuscenes_v1.0-trainval_train.json \
    --output_dir ./results/base_ckpt/$2 \
    --resolution 256 --device "$GPU" --seed 42
}

# the incumbent goes through the SAME script and the SAME set, so the comparison
# is paired on sample_token rather than against a remembered number
run_one ./output/nuscenes_trainval_full_r256_cont2/epoch1 incumbent_cont2_ep1
for ep in 0 1; do
  CK=./output/nuscenes_trainval_full_r256_cont3/epoch$ep
  [ -d "$CK" ] && run_one "$CK" cont3_ep$ep
done

ARMS=()
for ep in 0 1; do
  [ -d ./results/base_ckpt/cont3_ep$ep ] && ARMS+=(./results/base_ckpt/cont3_ep$ep)
done

if [ ${#ARMS[@]} -gt 0 ]; then
  python analysis/compare_base_ckpts.py \
    --ref ./results/base_ckpt/incumbent_cont2_ep1 "${ARMS[@]}"
else
  echo "[warn] no _cont3 checkpoints evaluated yet -- nothing to compare"
fi
