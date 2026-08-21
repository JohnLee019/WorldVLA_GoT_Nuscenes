#!/usr/bin/env bash

# --- 리포 루트에서 실행되도록 고정 -----------------------------------------
# 이 스크립트는 2026-08-21에 루트에서 scripts/ 로 옮겨졌다. 안의 경로가 전부
# 루트 기준(results/, data/, eval_*.py)이라, 어디서 호출되든 루트로 이동한다.
cd "$(dirname "$0")/.." || exit 1
# -------------------------------------------------------------------------
# Wave 3: confirm the two wave-2 winners at a representative scale, and test
# whether they compose.
#
# Why 500 and not 200. The val records are ordered by scene, so --limit N is the
# first N, not a sample. On the first 200 the greedy baseline scores 1.7156; on
# the first 500 it scores 3.0538. The 200-record slice is a much easier stretch
# of driving -- and the 500-record breakdown already showed GoT is a measured
# no-op on easy/straight scenes (p=0.61) and only wins on turns (p=0.029) and
# fast driving (p=0.023). Wave 2 therefore compared arms in precisely the regime
# where none of them can show an effect, which is why every p came back 0.14-0.85.
# 500 also makes these directly comparable to the existing ref run
# (results/got_ep1_s42: GoT 3.0018, base 3.0538, diff -0.052).
#
# NOTE even 500 is the first 500 of 5119, not a random sample. Final paper
# numbers should use the full val split or an explicit random subsample.
#
# Run on the TURN subset (make_turn_subset.py). The wave-2 per-command split
# showed the whole effect lives in turns -- default GoT actually LOSES on
# straight (+0.0460) and wins on turns (-0.1780) -- and turns are only ~18% of
# val, so 500 mixed records buy just ~92 turn samples. 500 turn records buy ~5x
# the statistics where the effect is, for the same GPU time. The straight side
# stays reportable from the existing full-split runs; nothing is discarded.
#
# Arms:
#   temp_tight    pool: drop the 1.4/1.6 samples. Best on BOTH sides in wave 2
#                 (straight -0.0266, turn -0.2107) and the best turn minADE_C
#                 (1.4177 vs ref 1.4888) -- against prediction, narrowing the
#                 temperature range improved the oracle, so the high-temperature
#                 candidates were noise rather than usable diversity.
#   seg011        first segment abstains. Best turn p in wave 2 (0.046, win
#                 0.700), neutral on straight.
#   tight_seg011  both. They act on different stages (which candidates exist vs
#                 which segment gets to vote), so they should compose.
#
# NOT here, deliberately: kin_only (--w_command 0). Wave 2 made it look like the
# best arm on the aggregate, but per-command it destroys 75% of the turn gain
# (-0.178 -> -0.045) and WORSENS the turn pool (minADE_C 1.4888 -> 1.5742,
# i.e. the command term is load-bearing during beam pruning, not just at final
# selection). It is still required as the GT-leakage ablation -- but that
# ablation has to be run against the FINAL config, not against ref, so it
# belongs after this wave settles which config that is.
#
# ~500 x 11.5 s = ~1.6 h per arm, three cards in parallel.
set -u

CKPT=./output/nuscenes_trainval_full_r256_cont2/epoch1
TOK=../ckpts/Lumina-mGPT-7B-768
VAL=./data/nuscenes_records/nuscenes_v1.0-trainval_val_turns.json
# the mean-trajectory prior stays fitted on the FULL train split (it is built
# per command, so a turn-only eval still reads the same left/right means)
TRAIN=./data/nuscenes_records/nuscenes_v1.0-trainval_train.json
NORM=./data/nuscenes_records/nuscenes_norm.json
COLL=./data/nuscenes_records/nuscenes_collision_v1.0-trainval_val.json
# 0 = every record. The turn split is 619 records (259 left / 360 right) across
# 70 scenes, so taking all of them costs only ~24% more than --limit 500 and
# removes the truncation bias for good: records are scene-ordered, so a limit
# would cover ~53 of the 70 scenes. Getting burned by exactly this is what made
# wave 2 unreadable (its first 200 records were a much easier stretch of
# driving, baseline 1.7156 vs 3.0538 on the first 500).
LIMIT=0
SEED=42
OUT=./results/abl3

run () {   # run <gpu> <tag> <extra flags...>
  local gpu=$1 tag=$2; shift 2
  mkdir -p "$OUT/$tag"
  echo "[launch] gpu$gpu  $tag  $*"
  # PYTHONUNBUFFERED so the progress lines appear as they happen: print() goes
  # to stdout, which block-buffers under a redirect, while the HF warnings go to
  # stderr and flush immediately -- in wave 2 that made a healthy run look dead.
  # TRANSFORMERS_VERBOSITY=error drops the per-generate attention-mask warning
  # that otherwise dominates the log (~8k lines/arm).
  nohup env PYTHONUNBUFFERED=1 TRANSFORMERS_VERBOSITY=error \
    python eval_got_nuscenes.py \
    --resume_path "$CKPT" --tokenizer_path "$TOK" \
    --records_json "$VAL" --train_records_json "$TRAIN" --norm_path "$NORM" \
    --collision_json "$COLL" \
    --output_dir "$OUT/$tag" --seeds $SEED --limit $LIMIT --device "$gpu" \
    --k_candidates 4 --beam_width 2 \
    "$@" > "$OUT/$tag.log" 2>&1 &
}

if [ ! -f "$VAL" ]; then
  echo "missing $VAL -- build it first:"
  echo "  python scripts/make_turn_subset.py \\"
  echo "    --records ./data/nuscenes_records/nuscenes_v1.0-trainval_val.json \\"
  echo "    --out $VAL"
  exit 1
fi

nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
echo "^ all three cards must be free (wave 2's 'wide' must have finished). Ctrl-C to abort."
sleep 10

PASS=${1:-all}

if [ "$PASS" = "1" ] || [ "$PASS" = "all" ]; then
echo "===== pass 1 ====="
# ref on the turn subset: the existing 500-record run is a MIXED split, so it is
# not a valid comparator here. Every arm needs its own paired reference on the
# same records, and the baseline is deterministic, so the base numbers must come
# out identical across arms -- a free wiring check. (One caveat: wave 2's 'wide'
# arm did NOT match the others, 1.6720 vs 1.7156, almost certainly the bf16
# argmax tie-flip nondeterminism already documented in handoff section 7.2,
# surfacing because k=6/beam=3 changes the allocation pattern. Arms that share
# k and beam should still agree exactly.)
run 0 ref_turn      # default config, turn subset
run 1 temp_tight    --temperatures 1.0 1.1 1.2 1.3
run 2 seg011        --seg_weight_scale 0 1 1
wait
echo "pass 1 done"
fi

if [ "$PASS" = "2" ] || [ "$PASS" = "all" ]; then
echo "===== pass 2 ====="

nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
sleep 5
# THE arm this wave exists for. Component correlations say command is noise at
# selection time -- path_score ranks worse than kinematic alone on turns (+0.747
# vs +0.811) and straights (+0.543 vs +0.546) -- yet deleting command outright
# degrades the turn POOL (minADE_C 1.4888 -> 1.5742), which selection cannot
# explain. So command earns its keep during pruning and costs at selection.
# --final_weights splits those two jobs: normal pruning, kinematic-only pick.
run 0 cmd_prune_only --final_weights 1 0
# how much of the TURN gain is the command term? (ref_turn minus this).
# Required regardless as the GT-leakage ablation, now at n=619 rather than 30.
run 1 ref_nocmd      --w_command 0
# push the wide result to its conclusion. k=6/beam=3 gave the selector a 13.8%
# better pool (minADE_C 1.2750 -> 1.0990) and it did significantly WORSE
# (diff +0.0124 -> +0.1059, p=0.032, top1 halved) -- the selector degrades as
# the candidate set grows. temp_tight, the best arm, is also the one that HALVED
# candidate spread. If less really is more, fewer candidates should be better
# AND cheaper: 15 forward calls vs 20.
run 2 narrow         --k_candidates 3 --beam_width 2 --temperatures 1.0 1.1 1.2
wait
echo "pass 2 done"
fi

echo
echo "===== wave 3 (turn subset) ====="
for d in "$OUT"/*/; do
  t=$(basename "$d"); [ -f "$d/summary.json" ] || continue
  python - "$d/summary.json" "$t" <<'PY'
import json, sys
s = json.load(open(sys.argv[1])); tag = sys.argv[2]
g = list(s["got_per_seed"].values())[0]
o, p, b = g["oracle_selection"], g["vs_baseline_paired"], s["baseline_free_run"]
print(f"{tag:<14} GoT {g['avgL2@3s']:.4f}  base {b['avgL2@3s']:.4f}  "
      f"diff {g['avgL2@3s']-b['avgL2@3s']:+.4f}  win {p['win_rate']:.3f}  "
      f"p {p['wilcoxon_p']:.4f}  minADE_C {o['minADE_C']:.4f}  "
      f"gap {o['selection_gap_avgL2@3s']:.4f}  top1 {o['selection_top1']:.3f}  "
      f"spread {o['candidate_spread']:.3f}")
PY
done

cat <<EOF

Every 'base' above must be identical (greedy is deterministic on identical
records). If they differ, stop and find out why before reading anything else.

Full per-arm breakdown, now including the score-component correlations that
wave 2 could not produce -- these runs carry got_cand_kin / got_cand_cmd /
got_cand_total. On a turn-only split the command term should show a clearly
POSITIVE rho against -true_error; wave 2 inferred it was load-bearing on turns
only indirectly, from the damage that removing it did:

  for d in $OUT/*/; do echo "== \$(basename \$d)"; \\
    python analysis/analyze_got_csv.py "\$d/per_sample.csv" --records_json $VAL; done

Then the required GT-leakage ablation, run against whichever config wins here
(not against ref) -- the point is what the FINAL system loses without command:

  ... --output_dir $OUT/<winner>_nocmd <winner flags> --w_command 0
EOF
