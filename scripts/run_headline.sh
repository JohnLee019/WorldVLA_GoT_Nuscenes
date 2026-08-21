#!/usr/bin/env bash

# --- 리포 루트에서 실행되도록 고정 -----------------------------------------
# 이 스크립트는 2026-08-21에 루트에서 scripts/ 로 옮겨졌다. 안의 경로가 전부
# 루트 기준(results/, data/, eval_*.py)이라, 어디서 호출되든 루트로 이동한다.
cd "$(dirname "$0")/.." || exit 1
# -------------------------------------------------------------------------
# Headline numbers on the scene-spread eval set (600 records, ALL 150 val
# scenes, built by make_turn_subset.py --per_scene 4).
#
# This is the set the paper's planning table comes from. Every earlier run used
# --limit N on a scene-ordered file, which meant 6 scenes at N=200 and 15 at
# N=500; two conclusions died of it (a "significant" turn advantage, and a
# "significant" wide-pool regression). Nothing with --limit goes in the paper.
#
# Three arms, one per card, same wall clock as one:
#   ref         default GoT. The GoT row of the main table, against the greedy
#               free-run and mean-trajectory baselines the script computes anyway.
#   temp_tight  the only arm that ever moved the pool significantly
#               (minADE_C -0.0787, p=0.0006 on the turn split). Running it here
#               tests whether the absorption result -- pool improves, selection
#               gap widens by 71% of it, output does not move -- reproduces on a
#               scene-diverse MIXED sample rather than on turns only. That is
#               the paper's central claim, so it should not rest on one split.
#   seg011      the contrast case: it improved the output by the same amount as
#               temp_tight while barely touching the pool, which is what shows
#               the absorption is triggered by large changes to the candidate
#               distribution rather than by improvement as such.
#
# ~600 x 12 s = ~2 h.
set -u

CKPT=./output/nuscenes_trainval_full_r256_cont2/epoch1
TOK=../ckpts/Lumina-mGPT-7B-768
VAL=./data/nuscenes_records/nuscenes_val_scenespread.json
TRAIN=./data/nuscenes_records/nuscenes_v1.0-trainval_train.json
NORM=./data/nuscenes_records/nuscenes_norm.json
COLL=./data/nuscenes_records/nuscenes_collision_v1.0-trainval_val.json
OUT=./results/headline

run () {   # run <gpu> <tag> <extra flags...>
  local gpu=$1 tag=$2; shift 2
  mkdir -p "$OUT/$tag"
  echo "[launch] gpu$gpu  $tag  $*"
  nohup env PYTHONUNBUFFERED=1 TRANSFORMERS_VERBOSITY=error \
    python eval_got_nuscenes.py \
    --resume_path "$CKPT" --tokenizer_path "$TOK" \
    --records_json "$VAL" --train_records_json "$TRAIN" --norm_path "$NORM" \
    --collision_json "$COLL" \
    --output_dir "$OUT/$tag" --seeds 42 --limit 0 --device "$gpu" \
    --k_candidates 4 --beam_width 2 \
    "$@" > "$OUT/$tag.log" 2>&1 &
}

if [ ! -f "$VAL" ]; then
  echo "missing $VAL -- build it first:"
  echo "  python scripts/make_turn_subset.py \\"
  echo "    --records ./data/nuscenes_records/nuscenes_v1.0-trainval_val.json \\"
  echo "    --out $VAL --keep left right straight --per_scene 4"
  exit 1
fi

nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
echo "^ all three cards must be free. Ctrl-C within 10 s to abort."
sleep 10

run 0 ref
run 1 temp_tight --temperatures 1.0 1.1 1.2 1.3
run 2 seg011     --seg_weight_scale 0 1 1
wait

cat <<EOF

===== done. The two things to read =====

1) the main table (GoT vs greedy free-run vs mean-trajectory):

  for d in $OUT/*/; do t=\$(basename "\$d"); python - "\$d/summary.json" "\$t" <<'PY'
import json, sys
s = json.load(open(sys.argv[1])); tag = sys.argv[2]
g = list(s["got_per_seed"].values())[0]
o, p = g["oracle_selection"], g["vs_baseline_paired"]
b, m = s["baseline_free_run"], s["baseline_mean_traj"]
print(f"{tag:<12} GoT {g['avgL2@3s']:.4f} | greedy {b['avgL2@3s']:.4f} | "
      f"meantraj {m['avgL2@3s']:.4f} | L2@3s {g['L2@3s']:.4f} | "
      f"coll@3s {g['collision']['coll@3s_pct']:.2f}% | minADE_C {o['minADE_C']:.4f} | "
      f"gap {o['selection_gap_avgL2@3s']:.4f} | s/rec {g['cost']['sec_per_record']:.2f}")
PY
  done

2) whether absorption reproduces off the turn split -- read d_pool against
   d_gap and d_output. On turns it was -0.0787 / +0.0560 / -0.0227:

  python analysis/compare_arms.py --ref $OUT/ref/per_sample.csv $OUT/*/per_sample.csv

  python analysis/analyze_got_csv.py $OUT/ref/per_sample.csv \\
    --records_json $VAL

All p-values and CIs in both tools are scene-clustered (150 scenes here, vs 70
on the turn split), so they are the ones that can be reported as-is.
EOF
