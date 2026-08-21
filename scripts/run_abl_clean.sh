#!/usr/bin/env bash
# Session-7 re-measurement of the two ablations the paper actually needs
# (PROJECT_HANDOFF §11 action 4), on the scene-spread set so they pair directly
# against the existing results/headline/ref (150 scenes, not the turn set's 70).
#
#   ref_nocmd        --w_command 0     : the GT-leakage defence (§3). Reviewers
#                                        always ask; the current number is
#                                        random-crop data.
#   cmd_prune_only   --final_weights 1 0 : command leads the beam but drops out of
#                                        the final pick. Pruning is untouched, so
#                                        d_pool MUST come out exactly +0.0000 --
#                                        that is the implementation check, and a
#                                        non-zero value means the arm is void.
set -u
CKPT=./output/nuscenes_trainval_full_r256_cont2/epoch1
TOK=../ckpts/Lumina-mGPT-7B-768
VAL=./data/nuscenes_records/nuscenes_val_scenespread.json
TRAIN=./data/nuscenes_records/nuscenes_v1.0-trainval_train.json
NORM=./data/nuscenes_records/nuscenes_norm.json
COLL=./data/nuscenes_records/nuscenes_collision_v1.0-trainval_val.json
OUT=./results/abl_clean

run () { local gpu=$1 tag=$2; shift 2
  mkdir -p "$OUT/$tag"; echo "[launch] gpu$gpu $tag $*"
  nohup env PYTHONUNBUFFERED=1 TRANSFORMERS_VERBOSITY=error \
    python eval_got_nuscenes.py \
    --resume_path "$CKPT" --tokenizer_path "$TOK" \
    --records_json "$VAL" --train_records_json "$TRAIN" --norm_path "$NORM" \
    --collision_json "$COLL" \
    --output_dir "$OUT/$tag" --seeds 42 --limit 0 --device "$gpu" \
    --k_candidates 4 --beam_width 2 "$@" > "$OUT/$tag.log" 2>&1 &
}
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
echo "^ cards 0 and 1 must be free. Ctrl-C within 10 s."
sleep 10
run 0 ref_nocmd      --w_command 0
run 1 cmd_prune_only --final_weights 1 0
wait
echo "===== paired vs the clean ref (seed 42) ====="
python compare_arms.py --ref results/headline/ref/per_sample.csv \
  $OUT/ref_nocmd/per_sample.csv $OUT/cmd_prune_only/per_sample.csv
echo
echo "===== turn-conditional, no GPU needed ====="
python analyze_got_csv.py $OUT/ref_nocmd/per_sample.csv \
  --records_json $VAL --command left right | head -8
