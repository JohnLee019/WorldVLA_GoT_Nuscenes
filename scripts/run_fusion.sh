#!/usr/bin/env bash

# --- 리포 루트에서 실행되도록 고정 -----------------------------------------
# 이 스크립트는 2026-08-21에 루트에서 scripts/ 로 옮겨졌다. 안의 경로가 전부
# 루트 기준(results/, data/, eval_*.py)이라, 어디서 호출되든 루트로 이동한다.
cd "$(dirname "$0")/.." || exit 1
# -------------------------------------------------------------------------
# Fusion wave 1 -- "combine the candidates" instead of "select one of them".
#
# Why this arm exists, in one paragraph. Eleven interventions asked the score to
# rank better and every one is null: weight sweeps, z-norm, kinematic-only
# selection, self-likelihood, segment weighting, pool width, temperature, a
# learned linear rule, a learned nonlinear rule, a learned fallback detector,
# and a hand-made confidence proxy. The learned ones bound the ceiling -- a
# fitted nonlinear combination of every logged signal reaches within-pool rho
# 0.524 against 0.519 for the best single component, and selecting with it still
# lands 0.029 m worse than not deliberating at all. Fusion is the only remaining
# idea that does not need ranking ability:
#
#   * avgL2 is a DISTANCE. The estimator minimising expected distance is the
#     median of the predictive distribution, not its mode -- and greedy decoding
#     returns approximately the mode. The gap needs no discrimination to collect.
#   * GoT adds near-symmetric zero-mean noise around greedy (42.7% worse / 38.7%
#     better / 18.7% identical). Median is the operation that cancels that.
#   * The absorption law is an identity over output = minADE_C + selection_gap.
#     No selection, no gap term, nothing to absorb.
#
# ARMS. `segment` is the arm the argument is about: it fuses every segment, so
# every waypoint is averaged, and it costs 12 forward calls instead of 20 --
# CHEAPER than the arm it is compared against. `final` is the dilution control:
# it leaves the pipeline alone (20 calls) and fuses only at the end, where the
# eight candidates come from just two distinct first-two-segment paths. Running
# both is what makes a null interpretable -- without `final` a null in `segment`
# could be blamed on the changed pipeline, and without `segment` a null in
# `final` could be blamed on dilution.
#
# ~1.5 h for the two segment arms, ~2.4 h for final; all three in parallel.
set -u

CKPT=./output/nuscenes_trainval_full_r256_cont2/epoch1
TOK=../ckpts/Lumina-mGPT-7B-768
VAL=./data/nuscenes_records/nuscenes_val_scenespread.json
TRAIN=./data/nuscenes_records/nuscenes_v1.0-trainval_train.json
NORM=./data/nuscenes_records/nuscenes_norm_v1.0-trainval.json
COLL=./data/nuscenes_records/nuscenes_collision_v1.0-trainval_val.json
OUT=./results/fusion

run () {   # run <gpu> <outdir> <extra flags...>
  local gpu=$1 out=$2; shift 2
  mkdir -p "$out"
  echo "[launch] gpu$gpu  $out  $*"
  nohup env PYTHONUNBUFFERED=1 TRANSFORMERS_VERBOSITY=error \
    python eval_got_nuscenes.py \
    --resume_path "$CKPT" --tokenizer_path "$TOK" \
    --records_json "$VAL" --train_records_json "$TRAIN" --norm_path "$NORM" \
    --collision_json "$COLL" \
    --output_dir "$out" --seeds 42 --limit 0 --device "$gpu" \
    --k_candidates 4 --beam_width 2 \
    "$@" > "$out.log" 2>&1 &
}

nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
echo "^ all three cards must be free. Ctrl-C within 10 s to abort."
sleep 10

# the arm the argument is about: every waypoint fused, beam gone, 12 calls
run 0 "$OUT/seg_all" --fuse median --fuse_scope segment
# same but only the two best-scored candidates per segment: more averaging is
# more variance cancellation AND more risk of averaging two different
# manoeuvres, so the m sweep is itself informative about which force dominates
run 1 "$OUT/seg_top2" --fuse median --fuse_scope segment --fuse_top_m 2
# dilution control: pipeline untouched (20 calls), fusion only at the end
run 2 "$OUT/final_top3" --fuse median --fuse_scope final --fuse_top_m 3
wait

cat <<'EOF'

===================== READ IN THIS ORDER =====================

1. WIRING. Every arm's greedy baseline must be 3.5557. It is deterministic and
   fusion cannot touch it, so anything else means a different condition got
   mixed in (PROJECT_HANDOFF sec 0 -- the stale-arm trap).

2. FEASIBILITY, BEFORE ANY L2. A fused trajectory is one no candidate proposed,
   so it never faced the veto every candidate had to clear. summary.json ->
   fusion.n_infeasible_output must be 0. If it is not, an L2 gain may be
   carried by physically impossible paths, which is a bug and not a result.

3. THE PRE-REGISTERED RULE (fixed before the run; wave 5 is why). Advance an arm
   to seeds 43/44 ONLY if |d_output| >= 0.06 m at seed 42. Below that it is
   under the resolution (scene-clustered CI half-width ~0.031) and more seeds
   cannot resolve it. Two of this project's claims already died by ignoring
   this: temp_tight and lik_full were significant at seed 42 and vanished at 43
   and 44.

4. MODE AVERAGING, PER COMMAND. Averaging "turn left" and "turn right" gives a
   path belonging to neither. Aggregate numbers hide it. Check turns separately
   -- if straights improve while turns get worse, that is the failure mode, not
   a mixed result.

5. selection_gap / selection_rank / selection_top1 are ABSENT from these arms by
   design: nothing was selected. minADE_C stays comparable in the `final` arm
   (untouched pool); in `segment` it is the last segment's pool only and means
   something different -- do not put the two in one column.
EOF

echo
echo "===== 1. baselines + headline ====="
for d in "$OUT"/*/; do
  python - "$d" <<'PY'
import json, sys, os
p = os.path.join(sys.argv[1], "summary.json")
if not os.path.exists(p):
    print(f"  {sys.argv[1]}: NO SUMMARY (crashed? see the .log)"); raise SystemExit
s = json.load(open(p))
got = list(s["got_per_seed"].values())[0]
base = s.get("baseline_free_run", {})
fus = s.get("fusion", {})
d = got["avgL2@3s"] - base.get("avgL2@3s", float("nan"))
flag = "ADVANCE" if abs(d) >= 0.06 else "below resolution -> STOP"
print(f"  {os.path.basename(os.path.normpath(sys.argv[1])):14s} "
      f"GoT {got['avgL2@3s']:.4f}  greedy {base.get('avgL2@3s', float('nan')):.4f}  "
      f"d {d:+.4f}  [{flag}]   infeasible={fus.get('n_infeasible_output')}  "
      f"calls={got.get('cost', {}).get('forward_calls_per_record')}  "
      f"s/rec={got.get('cost', {}).get('sec_per_record')}")
PY
done

echo
echo "===== 2. paired against the selecting arm (results/headline/ref) ====="
for d in "$OUT"/*/; do
  echo "--- $d"
  python analysis/compare_arms.py --ref ./results/headline/ref/per_sample.csv \
    "$d/per_sample.csv" 2>/dev/null || echo "  (compare_arms failed; run it by hand)"
done

echo
echo "===== 3. mode-averaging check: turns vs straights ====="
for d in "$OUT"/*/; do
  echo "--- $d"
  python analysis/analyze_got_csv.py "$d/per_sample.csv" \
    --records_json "$VAL" 2>/dev/null | sed -n '/command/,/^$/p' | head -12
done
