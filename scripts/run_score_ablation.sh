#!/usr/bin/env bash

# --- 리포 루트에서 실행되도록 고정 -----------------------------------------
# 이 스크립트는 2026-08-21에 루트에서 scripts/ 로 옮겨졌다. 안의 경로가 전부
# 루트 기준(results/, data/, eval_*.py)이라, 어디서 호출되든 루트로 이동한다.
cd "$(dirname "$0")/.." || exit 1
# -------------------------------------------------------------------------
# Score / candidate-pool ablation for eval_got_nuscenes, 3 GPUs in parallel.
#
# Ordering rationale (from the 500-record seed-42 analysis):
#   - on the random->oracle scale the score recovers 60.5% (random pick 4.0518,
#     oracle 2.3164, score pick 3.0018) and its position-wise means are strictly
#     monotone, so it is a GOOD ranker. Greedy already sits at 57.5% of that same
#     range, which is why the margin is 1.7% -- the room above the incumbent is
#     structurally narrow, not the score's competence.
#   - so the lever is the POOL, not the weights: either fewer junk candidates
#     (temperature) or more/better ones (k, beam). Weight arms are kept because
#     w_command 0 is needed anyway as the GT-leakage ablation (handoff section 3).
#   - per-step L2 vs greedy was +0.055 at 0.5 s and -0.10..-0.12 from 1.5 s, so
#     the first segment is where the score costs more than it earns.
#   - ALWAYS re-read each arm per command: the aggregate is 82% 'straight', where
#     GoT is a measured no-op (p=0.61). The signal lives in the turns (p=0.029)
#     and an arm can improve the turns while the aggregate barely moves.
#
# Everything here is CLI-only except seg_weight_scale, which needs the patched
# got_pipeline_drive.py + eval_got_nuscenes.py.
#
# Usage:  bash run_score_ablation.sh            # wave 1 and 2, ~2 h total
#         bash run_score_ablation.sh 1          # wave 1 only
set -u

CKPT=./output/nuscenes_trainval_full_r256_cont2/epoch1
TOK=../ckpts/Lumina-mGPT-7B-768
VAL=./data/nuscenes_records/nuscenes_v1.0-trainval_val.json
TRAIN=./data/nuscenes_records/nuscenes_v1.0-trainval_train.json
NORM=./data/nuscenes_records/nuscenes_norm_v1.0-trainval.json
COLL=./data/nuscenes_records/nuscenes_collision_v1.0-trainval_val.json
LIMIT=200          # ~53 min/config at 15.8 s/record
SEED=42
OUT=./results/abl

run () {   # run <gpu> <tag> <extra flags...>
  local gpu=$1 tag=$2; shift 2
  mkdir -p "$OUT/$tag"
  echo "[launch] gpu$gpu  $tag  $*"
  nohup python eval_got_nuscenes.py \
    --resume_path "$CKPT" --tokenizer_path "$TOK" \
    --records_json "$VAL" --train_records_json "$TRAIN" --norm_path "$NORM" \
    --collision_json "$COLL" \
    --output_dir "$OUT/$tag" --seeds $SEED --limit $LIMIT --device "$gpu" \
    --k_candidates 4 --beam_width 2 \
    "$@" > "$OUT/$tag.log" 2>&1 &
}

# ---- preflight: an occupied card kills the run (and any training on it) ----
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
echo "^ every card must be near-empty before continuing (handoff section 9). Ctrl-C to abort."
sleep 10

WAVE=${1:-all}

if [ "$WAVE" = "1" ] || [ "$WAVE" = "all" ]; then
  echo "===== wave 1 ====="
  # reference: current config at THIS limit. The 500-record numbers are not a
  # valid comparator -- the first 200 records are a different (scene-skewed)
  # sample, so every arm needs its own paired baseline at the same limit.
  run 0 ref
  # pool quality: drop the 1.4/1.6 samples that produce the 5.30 tail
  run 1 temp_tight   --temperatures 1.0 1.1 1.2 1.3
  # first segment abstains -> beam keeps the greedy candidate there
  run 2 seg011       --seg_weight_scale 0 1 1
  wait
  echo "wave 1 done"
fi

if [ "$WAVE" = "2" ] || [ "$WAVE" = "all" ]; then
  echo "===== wave 2 ====="
  # GT-leakage ablation (required for the paper) = kinematic only
  run 0 kin_only     --w_kinematic 1 --w_command 0
  # the other half: is kinematic contributing anything to the ranking at all?
  run 1 cmd_only     --w_kinematic 0 --w_command 1
  # is the bottleneck pool SIZE rather than pool cleanliness? More samples can
  # only lower minADE_C (a min over a superset), so this is the arm that buys
  # oracle headroom outright. ~42 forward calls/record vs 20, so budget ~1.7 h.
  # 6 temperatures given explicitly: the config reuses the LAST entry past the
  # end of the tuple, which would make candidates 5 and 6 duplicates of 4 and
  # get them deduped away.
  run 2 wide         --k_candidates 6 --beam_width 3 \
                     --temperatures 1.0 1.1 1.2 1.3 1.4 1.5
  wait
  echo "wave 2 done"
fi

echo
echo "===== compare ====="
for d in "$OUT"/*/; do
  t=$(basename "$d")
  [ -f "$d/summary.json" ] || { echo "$t: no summary.json (check $OUT/$t.log)"; continue; }
  python - "$d/summary.json" "$t" <<'PY'
import json, sys
s = json.load(open(sys.argv[1])); tag = sys.argv[2]
g = list(s["got_per_seed"].values())[0]
o, p, b = g["oracle_selection"], g["vs_baseline_paired"], s["baseline_free_run"]
print(f"{tag:<12} GoT {g['avgL2@3s']:.4f}  base {b['avgL2@3s']:.4f}  "
      f"diff {g['avgL2@3s']-b['avgL2@3s']:+.4f}  win {p['win_rate']:.3f}  "
      f"p {p['wilcoxon_p']:.4f}  minADE_C {o['minADE_C']:.4f}  "
      f"gap {o['selection_gap_avgL2@3s']:.4f}  top1 {o['selection_top1']:.3f}  "
      f"spread {o['candidate_spread']:.3f}")
PY
done
echo
cat <<EOF

How to read the table above:
  * selection_gap shrinks for TWO opposite reasons -- the score got better, or
    the pool got worse (a pool with no good candidate has no gap to lose). Always
    read it next to minADE_C. A real win is diff down AND minADE_C flat or lower.
  * tighter temperatures should raise the pool mean but also RAISE minADE_C
    (less diversity = a worse best candidate). The net is not predictable from
    either number alone; that trade-off is the whole point of the temp arms.
  * the aggregate diff is 82% 'straight', where GoT is a measured no-op. Re-read
    every arm per command before concluding anything:

      for d in $OUT/*/; do echo "== \$d"; \\
        python analysis/analyze_got_csv.py "\$d/per_sample.csv" --records_json $VAL \\
        | sed -n '/turn vs straight/,/^\$/p'; done
EOF
