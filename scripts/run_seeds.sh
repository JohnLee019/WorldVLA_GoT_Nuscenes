#!/usr/bin/env bash

# --- 리포 루트에서 실행되도록 고정 -----------------------------------------
# 이 스크립트는 2026-08-21에 루트에서 scripts/ 로 옮겨졌다. 안의 경로가 전부
# 루트 기준(results/, data/, eval_*.py)이라, 어디서 호출되든 루트로 이동한다.
cd "$(dirname "$0")/.." || exit 1
# -------------------------------------------------------------------------
# Close the two single-seed gaps in the paper's claims. Everything reported so
# far -- the headline (GoT vs greedy) and the mechanism (absorption) -- rests on
# seed 42 alone, and GoT's candidate sampling is stochastic, so neither number
# currently has a spread attached.
#
#   ref  seed 43 / 44   the headline gets mean +- std over three seeds. The
#                       greedy baseline is deterministic, so all three runs must
#                       report the same base_avgL2@3s (3.5839) -- a free wiring
#                       check that costs nothing to look at.
#   wide seed 43        the mechanism claim (pool -0.2840 p<1e-4, gap +0.2962
#                       p<1e-4, output null, 104% absorbed) is the strongest
#                       result in the project and is currently one seed deep.
#                       This is the arm whose intervention is large enough to
#                       clear the noise floor, so it is the one worth repeating.
#                       42 forward calls/record, hence ~2x the wall clock.
#
# ~2 h for the two ref seeds, ~4 h for wide; all three in parallel.
set -u

CKPT=./output/nuscenes_trainval_full_r256_cont2/epoch1
TOK=../ckpts/Lumina-mGPT-7B-768
VAL=./data/nuscenes_records/nuscenes_val_scenespread.json
TRAIN=./data/nuscenes_records/nuscenes_v1.0-trainval_train.json
NORM=./data/nuscenes_records/nuscenes_norm.json
COLL=./data/nuscenes_records/nuscenes_collision_v1.0-trainval_val.json

run () {   # run <gpu> <outdir> <seed> <extra flags...>
  local gpu=$1 out=$2 seed=$3; shift 3
  mkdir -p "$out"
  echo "[launch] gpu$gpu  $out  seed=$seed  $*"
  nohup env PYTHONUNBUFFERED=1 TRANSFORMERS_VERBOSITY=error \
    python eval_got_nuscenes.py \
    --resume_path "$CKPT" --tokenizer_path "$TOK" \
    --records_json "$VAL" --train_records_json "$TRAIN" --norm_path "$NORM" \
    --collision_json "$COLL" \
    --output_dir "$out" --seeds "$seed" --limit 0 --device "$gpu" \
    --k_candidates 4 --beam_width 2 \
    "$@" > "$out.log" 2>&1 &
}

nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
echo "^ all three cards must be free. Ctrl-C within 10 s to abort."
sleep 10

run 0 ./results/headline/ref_s43 43
run 1 ./results/headline/ref_s44 44
# NOTE (session 7): no --temperatures here, on purpose. got_pipeline_drive:313 does
# temperatures[min(i, len-1)], so the default 4-value schedule clamps candidates 4
# and 5 to 1.6 rather than erroring. The seed-42 `wide` was run that way, and a
# seed arm must match the arm it replicates -- passing a 6-value schedule here
# would change the intervention, not just the seed.
run 2 ./results/headline/wide_s43 43 --k_candidates 6 --beam_width 3
wait

cat <<'EOF'

===== 1. headline across seeds =====
EOF
python merge_seeds.py ./results/headline/ref ./results/headline/ref_s43 \
  ./results/headline/ref_s44 2>/dev/null || python - <<'PY'
# merge_seeds.py lives only on gpu-server; fall back to reading the summaries
import json, statistics as st
runs = ["./results/headline/ref", "./results/headline/ref_s43", "./results/headline/ref_s44"]
got, base = [], []
for r in runs:
    s = json.load(open(f"{r}/summary.json"))
    got.append(list(s["got_per_seed"].values())[0])
    base.append(s["baseline_free_run"])
print("  greedy base_avgL2@3s per run (must be identical):",
      [round(b["avgL2@3s"], 4) for b in base])
for k in ("L2@1s", "L2@2s", "L2@3s", "avgL2@3s"):
    v = [g[k] for g in got]
    print(f"  GoT {k:<10} {st.mean(v):.4f} +- {st.stdev(v):.4f}   {[round(x,4) for x in v]}")
for k in ("minADE_C", "selection_gap_avgL2@3s", "selection_top1"):
    v = [g["oracle_selection"][k] for g in got]
    print(f"      {k:<22} {st.mean(v):.4f} +- {st.stdev(v):.4f}")
d = [g["avgL2@3s"] - b["avgL2@3s"] for g, b in zip(got, base)]
print(f"  GoT - greedy: {st.mean(d):+.4f} +- {st.stdev(d):.4f}   {[round(x,4) for x in d]}")
PY

cat <<'EOF'

===== 2. does absorption replicate on a second seed? =====
  d_pool and d_gap must both stay clear of zero and still cancel.
EOF
python analysis/compare_arms.py --ref ./results/headline/ref_s43/per_sample.csv \
  ./results/headline/wide_s43/per_sample.csv
