#!/usr/bin/env bash

# --- 리포 루트에서 실행되도록 고정 -----------------------------------------
# 이 스크립트는 2026-08-21에 루트에서 scripts/ 로 옮겨졌다. 안의 경로가 전부
# 루트 기준(results/, data/, eval_*.py)이라, 어디서 호출되든 루트로 이동한다.
cd "$(dirname "$0")/.." || exit 1
# -------------------------------------------------------------------------
# Model self-likelihood as a candidate score, on the scene-spread eval set.
#
# Why this experiment. Every score tried so far reads the waypoints and nothing
# else: score_kinematic is trajectory geometry, score_command is a terminal
# lateral offset. None of them looks at the image, so none can say whether a
# smooth, command-consistent trajectory suits THIS scene. Six interventions on
# that family were all null on the output, and the absorption result showed why
# improving the pool cannot help (minADE_C -0.2840 p<1e-4, gap +0.2962 p<1e-4,
# output unchanged): selection is the only stage left. The model's own
# conditional log-likelihood of a candidate given the image is the one
# scene-grounded signal available without training a world model.
#
# Expected outcome, stated up front so a null is not read as a surprise: the
# candidates were sampled from this same model at temperature > 1, so ranking
# them by its likelihood partly undoes the sampling and may collapse towards
# greedy. That result is still worth having -- it says the generator cannot
# identify which of its own samples are good -- and it closes the obvious
# reviewer question ("did you try the model's own likelihood?"), which currently
# has no answer.
#
# IMPLEMENTATION CHECK for the likelihood arm: its re-rank runs AFTER pruning, so
# the pool is untouched and lik_full's minADE_C must equal ref's exactly -- read
# it as d_pool = -0.0000 in compare_arms rather than as an absolute, since ref
# itself moved (2.8646 -> 3.0053) when the eval-time random crop was fixed in
# session 7. Anything else is a bug, not a result. The --score_norm arms are
# different -- they change pruning as well, so their minADE_C is expected to move.
#
# SESSION 7 WARNING: every number quoted in the comments below (minADE_C -0.2840,
# gap +0.2962, "six interventions all null", rho +0.55..+0.81) was measured with
# the random crop active. The clean values are d_pool -0.2618 / d_gap +0.2696 /
# absorbed 103%, and temp_tight -- previously null -- is now significant
# (d_output -0.0408, p_sc 0.0049). So do NOT read a null here as confirmed in
# advance: the whole reason to re-run this sweep is that noise was hiding effects.
#
# The smoke run also turned up a second, possibly larger finding; see arm (b).
#
# Usage:  bash run_likelihood.sh smoke   # 5 records, ~2 min -- run this FIRST
#         bash run_likelihood.sh         # 3 arms x 600 records, ~2.5 h
set -u

CKPT=./output/nuscenes_trainval_full_r256_cont2/epoch1
TOK=../ckpts/Lumina-mGPT-7B-768
VAL=./data/nuscenes_records/nuscenes_val_scenespread.json
TRAIN=./data/nuscenes_records/nuscenes_v1.0-trainval_train.json
NORM=./data/nuscenes_records/nuscenes_norm.json
COLL=./data/nuscenes_records/nuscenes_collision_v1.0-trainval_val.json
OUT=./results/lik

run () {   # run <gpu> <tag> <limit> <extra flags...>
  local gpu=$1 tag=$2 lim=$3; shift 3
  mkdir -p "$OUT/$tag"
  echo "[launch] gpu$gpu  $tag  limit=$lim  $*"
  nohup env PYTHONUNBUFFERED=1 TRANSFORMERS_VERBOSITY=error \
    python eval_got_nuscenes.py \
    --resume_path "$CKPT" --tokenizer_path "$TOK" \
    --records_json "$VAL" --train_records_json "$TRAIN" --norm_path "$NORM" \
    --collision_json "$COLL" \
    --output_dir "$OUT/$tag" --seeds 42 --limit "$lim" --device "$gpu" \
    --k_candidates 4 --beam_width 2 \
    "$@" > "$OUT/$tag.log" 2>&1 &
}

# ---------------------------------------------------------------- smoke ----
# trajectory_logprob has passed its offline self-test but has never executed
# against the real 7B: the training-path forward is called with list-of-lists
# input_ids and labels, and its scalar c_loss is reused as the log-likelihood.
# Five records is enough to catch a shape, dtype or return-type error, and cheap
# enough to throw away. Do not skip to the full run.
if [ "${1:-}" = "smoke" ]; then
  mkdir -p "$OUT"
  run 0 smoke 5 --w_likelihood 1 --verbose_plan
  wait
  echo
  echo "===== smoke: what to check ====="
  grep -E "self-likelihood final re-rank ON|final re-rank w=" "$OUT/smoke.log" | head -8
  echo "--- errors/tracebacks (should be empty) ---"
  grep -nE "Traceback|Error|error:" "$OUT/smoke.log" | head -10
  echo "--- per-candidate likelihoods must be finite and DIFFER across candidates ---"
  python - "$OUT/smoke/per_sample.csv" <<'PY'
import ast, csv, math, sys
rows = list(csv.DictReader(open(sys.argv[1], newline="", encoding="utf-8")))
ok = bool(rows)
for r in rows[:5]:
    v = r.get("got_cand_lik") or ""
    if not v:
        print("  got_cand_lik column MISSING -> the scorer never ran"); ok = False; break
    lik = ast.literal_eval(v)
    fin = [x for x in lik if math.isfinite(x)]
    spread = (max(fin) - min(fin)) if len(fin) > 1 else 0.0
    print(f"  n={len(lik)} finite={len(fin)} spread={spread:.4f}  {[round(x,3) for x in lik]}")
    # all-nan means every call failed; zero spread means the score cannot rank
    ok &= len(fin) == len(lik) and spread > 1e-6
print("SMOKE:", "PASS -> run the full sweep" if ok else "FAIL -> fix before the sweep")
PY
  exit 0
fi

if [ ! -f "$OUT/smoke/summary.json" ]; then
  echo "run the smoke test first:  bash run_likelihood.sh smoke"
  exit 1
fi

nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
echo "^ all three cards must be free. Ctrl-C within 10 s to abort."
sleep 10

# ---- two independent findings, one sweep ----
# (a) the scene signal: does the model's own likelihood add anything at selection?
run 0 lik_full  0 --w_likelihood 1
# (b) THE NORMALISATION BUG found in the smoke csv, which may matter more.
# score_kinematic spans ~8 orders of magnitude within one pool, so z-norm hands
# the whole std to the single worst candidate: on a measured pool the three best
# candidates got z = +0.3851 / +0.3851 / +0.3828 while their true errors were
# 0.035 / 0.170 / 1.312 m. The term that ranks best in raw form (rho +0.55..+0.81)
# thus contributes no ordering at all, which would explain why reweighting it six
# different ways changed nothing. Rank-norm keeps the ordering and drops the
# magnitude -- safe because the feasibility gate vetoes catastrophes absolutely,
# outside the normalisation. This arm changes PRUNING too, so unlike (a) its
# d_pool is expected to move.
run 1 ranknorm  0 --score_norm rank
# (c) both, since they act on different stages
run 2 rank_lik  0 --score_norm rank --w_likelihood 1
wait

echo
echo "===== results ====="
python analysis/compare_arms.py --ref results/headline/ref/per_sample.csv \
  "$OUT"/lik_full/per_sample.csv "$OUT"/ranknorm/per_sample.csv \
  "$OUT"/rank_lik/per_sample.csv

cat <<EOF

Reading order:
  1. lik_full only: minADE_C must be -0.0000. The likelihood re-rank is
     downstream of pruning, so a non-zero d_pool means the pool moved and the arm
     is void. The two --score_norm arms DO change pruning, so their d_pool is
     expected to move and is part of the result, not a bug.
  2. d_gap is the whole result. Negative and clear of zero = a scene-grounded
     signal can do what reweighting could not. Straddling zero = the model
     cannot rank its own samples either, and the 20% headroom needs information
     from outside both the trajectory and the generator.
  3. Then the component correlation, which now includes the likelihood row:

       python analysis/analyze_got_csv.py $OUT/lik_full/per_sample.csv \\
         --records_json $VAL | tail -12

     Compare rho(likelihood) against rho(kinematic)=+0.55..+0.81. A likelihood
     rho near zero would explain a null d_gap; a high rho with a null d_gap
     would instead mean the top candidates are too close to separate, which is
     what killed cmd_prune_only (pool positions 0 and 1 differ by 0.08).
EOF
