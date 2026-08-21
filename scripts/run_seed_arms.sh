#!/usr/bin/env bash
# Seed replication for the two single-seed claims (PROJECT_HANDOFF §11, action 3b).
# ref_s43 / ref_s44 already exist from run_seeds.sh, so these pair directly.
set -u
CKPT=./output/nuscenes_trainval_full_r256_cont2/epoch1
TOK=../ckpts/Lumina-mGPT-7B-768
VAL=./data/nuscenes_records/nuscenes_val_scenespread.json
TRAIN=./data/nuscenes_records/nuscenes_v1.0-trainval_train.json
NORM=./data/nuscenes_records/nuscenes_norm.json
COLL=./data/nuscenes_records/nuscenes_collision_v1.0-trainval_val.json
OUT=./results/seed_arms
run () { local gpu=$1 tag=$2 seed=$3; shift 3
  mkdir -p "$OUT/$tag"; echo "[launch] gpu$gpu $tag seed=$seed $*"
  nohup env PYTHONUNBUFFERED=1 TRANSFORMERS_VERBOSITY=error \
    python eval_got_nuscenes.py \
    --resume_path "$CKPT" --tokenizer_path "$TOK" \
    --records_json "$VAL" --train_records_json "$TRAIN" --norm_path "$NORM" \
    --collision_json "$COLL" \
    --output_dir "$OUT/$tag" --seeds "$seed" --limit 0 --device "$gpu" \
    --k_candidates 4 --beam_width 2 "$@" > "$OUT/$tag.log" 2>&1 &
}
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
echo "^ all three cards must be free. Ctrl-C within 10 s."
sleep 10
run 0 temp_tight_s43 43 --temperatures 1.0 1.1 1.2 1.3
run 1 temp_tight_s44 44 --temperatures 1.0 1.1 1.2 1.3
run 2 lik_full_s43   43 --w_likelihood 1
wait
echo "===== seed 43 (ref_s43 기준) ====="
python compare_arms.py --ref results/headline/ref_s43/per_sample.csv \
  $OUT/temp_tight_s43/per_sample.csv $OUT/lik_full_s43/per_sample.csv
echo "===== seed 44 (ref_s44 기준) ====="
python compare_arms.py --ref results/headline/ref_s44/per_sample.csv \
  $OUT/temp_tight_s44/per_sample.csv
