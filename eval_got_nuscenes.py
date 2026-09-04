"""
Open-loop Graph-of-Thought (GoT) trajectory-planning evaluation on nuScenes.

This is the GoT counterpart of `eval_nuscenes.py`. For each val record it runs the
segment-wise GoT planner (`got_drive.got_pipeline_drive.DriveGoTPipeline`) on the
real trained model and reports the same nuScenes planning metric (L2@1/2/3s), so
the GoT trajectory can be compared directly against the plain free-run baseline.

What it wires together
----------------------
    model  --make_model_generate_fn-->  generate_fn(image, prefix_wp, ...)
                                              |
    DriveGoTPipeline.plan(command)  ----------+--> merged (time_horizon, 2) metres
                                              |
    scoring_driving.rank_candidates (kinematic + command)

Two modes (mirrors got_pipeline_drive):
  * Mode A (default): fixed image, each segment conditioned on the running prefix
    of decided waypoints. This is the (A) building block -- run this the moment a
    base checkpoint exists.
  * Mode B (--wm_path given): the world model predicts the next front frame
    between segments (Context Update). Needs a WM trained by train_nuscenes_wm.py,
    loaded with mask_image_logits=False.

Baseline comparison
-------------------
Unless --no_baseline is passed, every record is ALSO evaluated with the plain
free-run generator (`eval_nuscenes.predict_waypoints`, greedy). The summary
reports GoT vs baseline mean L2 and the per-record win rate, i.e. the question
that actually matters: does the GoT skeleton improve the trajectory over just
decoding all 6 waypoints in one shot on this checkpoint?

What the summary contains
-------------------------
L2@t (UniAD strategy, exact timestep) and avgL2@t (ST-P3/VAD strategy, mean over
0..t) for GoT / free-run baseline / mean-trajectory prior, plus:

  oracle_selection   minADE_C, oracle_L2@t, selection_gap_*, selection_rank,
                     selection_top1, candidate_spread, worst_candidate.
                     Computed from the candidate pool the GoT score chose from
                     (DriveGoTPipeline.final_candidates) at ZERO extra model
                     cost. This is what separates "the generator had nothing
                     good" from "the score failed to pick it" -- the question a
                     single L2 number cannot answer.
  tail               P50/P90/P95/max and frac(L2 > --catastrophe_m) of the L2
                     distribution. The feasibility gate acts on the tail, so its
                     effect is invisible in the mean.
  vs_baseline_paired Wilcoxon signed-rank p and a bootstrap CI on the paired
                     mean difference. GoT vs baseline is an exactly paired design
                     with a small effect; the mean alone cannot separate it from
                     sampling noise.
  collision          coll@t / meanColl@t / cumColl@t, see --collision_parity.
  cost               model forward calls and seconds per record (GoT buys its
                     accuracy with extra forwards; report the price).

With --seeds A B C the whole thing is repeated per seed and `got_across_seeds`
carries mean/std. The greedy baselines are deterministic and computed ONCE.

Usage
-----
    python eval_got_nuscenes.py \
        --resume_path ./output/nuscenes_trainval_full_r256/epoch4 \
        --tokenizer_path ../ckpts/Lumina-mGPT-7B-768 \
        --records_json ./data/nuscenes_records/nuscenes_v1.0-trainval_val.json \
        --train_records_json ./data/nuscenes_records/nuscenes_v1.0-trainval_train.json \
        --norm_path ./data/nuscenes_records/nuscenes_norm_v1.0-trainval.json \
        --collision_json ./data/nuscenes_records/nuscenes_collision_v1.0-trainval_val.json \
        --output_dir ./results/nuscenes_got \
        --k_candidates 4 --beam_width 2 --seeds 42 43 44 --limit 500

Caveats
-------
* --norm_path / --resolution MUST match training (same as eval_nuscenes).
* Open-loop L2 on nuScenes is a weak metric (ego-motion extrapolation scores
  well); treat it as a pipeline regression check, not proof of driving skill.
* GoT is only meaningful once the base checkpoint continues a partial trajectory
  coherently -- gate this behind verify_got_prefix_nuscenes.py (the (A) check).
* The RNG stream is now derived per (seed, record index) rather than seeded once
  for the whole run, so a record's plan no longer depends on --limit or on where
  it sits in the file. Numbers will NOT be bit-identical to runs from before
  this change; re-measure rather than comparing across the boundary.
"""

import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from data.item_processor import FlexARItemProcessor_Action_NuScenes
from data.dataset_nuscenes import DEFAULT_PROMPT
from got_drive.eval_crop import crop_for_eval
from got_drive.got_pipeline_drive import DriveGoTConfig, DriveGoTPipeline, make_model_generate_fn
from got_drive.graph import GraphPlanner
# The --planner -> graph_kind map and the per-arm kwargs. Imported rather than
# rebuilt here: an inline if-chain is where the aggregate/improve arms silently
# became the control arm, and it could not be unit-tested because this module
# needs torch. See got_drive/graph/planner.py.
from got_drive.graph.planner import (AGGREGATE_PLANNERS, GRAPH_PLANNERS,
                                     INTENT_PLANNERS, planner_kwargs)
from got_drive.planning_metrics import (across_seeds, oracle_and_selection,
                                        paired_comparison, tail_stats)
# feasibility predicate, reused to audit FUSED outputs: a fused trajectory is
# one no candidate proposed, so it never faced the veto the candidates cleared.
from got_drive.scoring_driving import _feasible

# Reuse the base eval's model loader, metric, free-run generator, seeding and
# mean-trajectory baseline verbatim so GoT and baselines share identical code.
from eval_nuscenes import (load_model, l2_metrics, predict_waypoints,
                           set_seed, compute_mean_trajectories, check_action_grid)


def get_args():
    p = argparse.ArgumentParser("nuScenes open-loop GoT planning eval")
    # --- shared with eval_nuscenes ---
    p.add_argument("--resume_path", required=True, help="trained checkpoint dir")
    p.add_argument("--tokenizer_path", required=True)
    p.add_argument("--records_json", required=True, help="val records from preprocess_nuscenes.py")
    p.add_argument("--norm_path", default=None, help="nuscenes_norm.json used at TRAINING time")
    p.add_argument("--output_dir", default="./results/nuscenes_got")
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--max_seq_len", type=int, default=4096)
    p.add_argument("--action_dim", type=int, default=2)
    p.add_argument("--time_horizon", type=int, default=6)
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--load_in_4bit", action="store_true", default=False)
    p.add_argument("--limit", type=int, default=0, help="evaluate only the first N records (0 = all)")
    p.add_argument("--seed", type=int, default=42, help="RNG seed (reproducible GoT sampling)")
    p.add_argument("--legacy_random_crop", action="store_true", default=False,
                   help="reproduce pre-fix numbers: let process_image() random-crop the frame on "
                        "every forward call. That is what made the deterministic greedy baseline "
                        "move whenever --k_candidates/--beam_width changed (§9): the baseline is "
                        "not re-seeded, so it inherits the RNG state left by the previous record's "
                        "GoT, whose draw count is k + 2*beam*k. See got_drive/eval_crop.py. "
                        "Never mix the two inside one table.")
    p.add_argument("--intent_speed_scales", type=float, nargs="+", default=None,
                   help="--planner intent only: multipliers on the model's own implied first-step "
                        "displacement. Default (0.0, 0.5, 1.0). ★These are prescribed by "
                        "measurement, not taste: Step 2.11 found our pool at GAP_dep -0.0942 "
                        "(worse than random) with the speed ladder at v x0.60-1.50 when the room "
                        "is at v x0.00-0.45, and Step 2.10 put the optimum at v x0.45-0.60 with "
                        "the conservative end MANDATORY. Quote any GAP with its N (Step 2.8).")
    p.add_argument("--intent_curvatures", type=float, nargs="+", default=None,
                   help="--planner intent/aggregate/improve: constant curvatures in 1/m. "
                        "Default is intent.DEFAULT_CURVATURES = (0.0,) -- SPEED-ONLY, and that "
                        "is a measurement, not a simplification. ★This help text used to claim "
                        "(-0.01, 0.0, +0.01), i.e. NAVSIM's values, which this package asserts "
                        "in its selftest CANNOT separate on this horizon: lateral offset grows "
                        "as k*L^2/2, so k=0.01 is 0.02 m over a 2 m first step and only ~0.72 m "
                        "over the whole 3 s horizon -- both under §1.5's 1 m floor, below which "
                        "0/600 records flipped a token. Those values were fitted to a 4 s nuPlan "
                        "horizon. For a real curvature axis use intent.USABLE_CURVATURES "
                        "(-0.05, 0.0, +0.05) AND --intent_anchor_steps >= 2, then confirm on "
                        "`got_intent_sep_max` per record. ★Keep them SYMMETRIC about 0 -- the "
                        "'do not do' list rejects the k=+0.005 cell because it sits on an "
                        "unexplained left bias (Step 2.10d), which Step 2.7e had already "
                        "discarded once.")
    p.add_argument("--intent_anchor_steps", type=int, default=1,
                   help="--planner intent only: how many waypoints the intent commits before "
                        "the model continues. 1 expresses SPEED and essentially not curvature: "
                        "an arc's lateral offset grows as k*L^2/2, so k=0.01 over a 2 m step is "
                        "0.02 m and even over the whole 3 s horizon only ~0.72 m -- both under "
                        "§1.5's 1 m floor. ★A curvature axis therefore needs BOTH k=±0.05 and "
                        "anchor_steps>=2. Every committed step is one the model no longer "
                        "chooses, so raise it only for the axis that needs it.")
    p.add_argument("--intent_variants", type=int, default=2,
                   help="--planner intent only: realisations per intent. Cost is "
                        "1 + n_intents*variants forward calls, NOT the incumbent's 20 -- report "
                        "the call count alongside the L2, and note that §1.5's free pairing only "
                        "holds between arms with equal call counts.")
    p.add_argument("--aggregate_method", choices=["median", "mean"], default="median",
                   help="--planner aggregate only: how one intent's realisations are combined. "
                        "Median is the estimator matched to an L2 metric (got_drive/fusion.py) "
                        "and is what §1.5 used.")
    p.add_argument("--aggregate_keep_inputs", action="store_true",
                   help="--planner aggregate only: keep each intent's realisations in the pool "
                        "alongside its aggregate. ★Without this the pool handed downstream is "
                        "just the aggregates, so minADE_C is computed over a DIFFERENT pool than "
                        "the other arms and `d_pool` stops being comparable with §1 (where "
                        "fusion left the pool at exactly 0). Say which one a number came from.")
    p.add_argument("--improve_tries", type=int, default=2,
                   help="--planner improve only: repair attempts per infeasible candidate. "
                        "Each attempt clamps the violating waypoint to the limit it broke and "
                        "regenerates the tail from there. ★A repair smaller than §1.5's ~1 m "
                        "floor cannot change the sampled tokens (0/600 flipped), so the csv "
                        "logs every repair's prefix delta and counts the ones below it.")
    p.add_argument("--improve_no_aggregate", action="store_true",
                   help="--planner improve only: skip the Aggregate fan-out so ValidateAndImprove "
                        "runs directly on the realisations. Isolates Improve's contribution from "
                        "Aggregate's -- run both if you want to attribute either.")
    p.add_argument("--planner",
                   choices=["pipeline", "graph", "intent", "aggregate", "improve"],
                   default="pipeline",
                   help="'pipeline' = got_drive.got_pipeline_drive.DriveGoTPipeline, the "
                        "incumbent every number in §1 was produced by. 'graph' = the same "
                        "search declared as a Graph of Operations and run by the upstream-shaped "
                        "FIFO controller (got_drive/graph/). ★They are meant to be IDENTICAL: "
                        "the graph arm's only job at this stage is to reproduce the incumbent "
                        "bit-for-bit, which is the control that licenses everything built on it "
                        "later (same idea as --fuse_top_m 1). A difference here is a bug, not a "
                        "result. Not supported by 'graph': --fuse, --wm_path scoring, "
                        "--final_weights, --w_likelihood -- each raises rather than running a "
                        "silently different arm. "
                        "★'intent' branches on DRIVING HYPOTHESES (speed profile x curvature) "
                        "instead of on time -- §1.17 measured 1.52 live options at stage 1 with "
                        "48%% of records having exactly one, so the incumbent deliberates late "
                        "while the decisive longitudinal error is fixed early. This is NOT a "
                        "control arm: it cannot reproduce 3.5557 and is not meant to. Put it in "
                        "results/graph/ and never pair it against the §1 tables. ⚠️Read "
                        "`got_intent_separated` in the csv BEFORE the L2 -- if the intents did "
                        "not separate by >=1 m the model ignored them (§1.5: sub-metre prefix "
                        "perturbations flipped 0 of 600 records) and the arm measured nothing.")
    p.add_argument("--graph_keep_valid", action="store_true",
                   help="graph planner only: run KeepValid as a real filtering operation "
                        "instead of a pass-through. ★This is a DIFFERENT ARM, not the control -- "
                        "the incumbent's feasibility veto still lives inside rank_candidates, so "
                        "this filters twice. It exists because §1.7(b)3 traced the z-norm veto "
                        "bug to validity being folded into the score; separating the operation "
                        "is the structural fix, and removing the in-score veto is the other half "
                        "of that change (not done yet).")
    p.add_argument("--graph_trace_dir", default=None,
                   help="graph planner only: write each record's executed operation graph "
                        "(topology + every thought's trajectory and provenance) as JSON here. "
                        "★Costs no model calls. §1.10(a2) could not separate 'the model did not "
                        "generate it' from 'the beam pruned it' because the pre-prune pool was "
                        "not retained; §1.17 was possible only because the surviving candidates' "
                        "trajectories were. Record it from the start.")
    p.add_argument("--with_state", action="store_true",
                   help="feed causal ego status (<|state|>) alongside the image, in BOTH the "
                        "GoT candidate generator and the greedy free-run baseline. MUST match "
                        "how the checkpoint was trained -- a state-trained checkpoint evaluated "
                        "without it (or vice versa) is off-distribution and the L2 is meaningless. "
                        "Records need a `state` field (data/preprocess_nuscenes.py) and the norm "
                        "json needs `state_min`. ★Results belong in results/egostate/, never "
                        "paired against the §1 tables -- different setup (§8).")
    p.add_argument("--allow_mini_grid", action="store_true",
                   help="run even when --norm_path is the v1.0-mini action grid. ★Never right "
                        "for a new number -- the incumbent scores 3.7889 instead of 3.5557 on it "
                        "and nothing else fails (§9). Exists only to reproduce a historical "
                        "mini-grid run on purpose.")
    p.add_argument("--seeds", type=int, nargs="+", default=None,
                   help="run GoT under several seeds and report mean+-std across them "
                        "(e.g. --seeds 42 43 44). GoT sampling is stochastic and the "
                        "GoT-vs-baseline gap is small, so a single seed cannot separate "
                        "the effect from sampling noise. Defaults to [--seed]. The greedy "
                        "free-run and mean-trajectory baselines are deterministic and are "
                        "computed ONCE, not per seed.")
    p.add_argument("--train_records_json", default=None,
                   help="records for the mean-trajectory baseline (ideally TRAIN split); "
                        "if unset, the eval split is used (slightly optimistic).")
    p.add_argument("--collision_json", default=None,
                   help="obstacle-box records from preprocess_nuscenes_collision.py. If given, "
                        "the UniAD/ST-P3 collision rate is reported next to L2 for GoT, the "
                        "free-run baseline and the mean-trajectory baseline.")
    p.add_argument("--collision_parity", choices=["uniad", "ours"], default="uniad",
                   help="'uniad' reproduces UniAD's PlanningMetric footprint (axis-aligned "
                        "ego box, +0.5 m forward shift) -- use this for any number you put "
                        "next to a published table. 'ours' orients the box along the heading "
                        "(physically right, not comparable). See got_drive/collision_metric.py.")
    p.add_argument("--no_collision_gt_mask", action="store_true", default=False,
                   help="disable UniAD's correction that discounts steps where the GT "
                        "trajectory itself collides. Leaving it ON (default) is what UniAD "
                        "does; turning it off biases every rate HIGH.")
    # --- diagnostic metric knobs ---
    p.add_argument("--rank_key", default="avgL2@3s",
                   help="single L2 key used to (a) pick the oracle candidate and (b) decide "
                        "GoT-vs-baseline head-to-head. One candidate is chosen and reported "
                        "across all horizons, so the oracle is a trajectory that could "
                        "actually have been driven.")
    p.add_argument("--tail_key", default="L2@3s",
                   help="L2 key whose distribution (P50/P90/P95/max, catastrophe rate) is "
                        "reported. The feasibility gate acts on the tail, not the mean.")
    p.add_argument("--catastrophe_m", type=float, default=10.0,
                   help="threshold for the catastrophe rate frac(L2 > X m)")
    p.add_argument("--n_boot", type=int, default=10000,
                   help="bootstrap resamples for the paired mean-difference CI")
    # --- GoT config ---
    p.add_argument("--n_segments", type=int, default=3)
    p.add_argument("--segment_len", type=int, default=2)
    p.add_argument("--k_candidates", type=int, default=4)
    p.add_argument("--beam_width", type=int, default=2)
    p.add_argument("--temperatures", type=float, nargs="+", default=[1.0, 1.2, 1.4, 1.6],
                   help="per-candidate temperature; idx0 is used greedily (do_sample=False)")
    p.add_argument("--w_kinematic", type=float, default=1.0)
    p.add_argument("--w_command", type=float, default=1.0)
    p.add_argument("--seg_weight_scale", type=float, nargs="+", default=None,
                   help="per-segment multiplier on the score weights, e.g. '0 1 1' to stop the "
                        "score from voting on the first segment (its candidates tie, so the beam "
                        "keeps them in generation order, greedy first). Note this makes the "
                        "segment ABSTAIN, it does not lock greedy in: a later voting segment can "
                        "still pick a non-greedy branch. Fewer values than segments reuses the "
                        "last. Default (unset) = uniform 1.0.")
    p.add_argument("--seg_w_command_scale", type=float, nargs="+", default=None,
                   help="per-segment multiplier on --w_command alone, on top of "
                        "--seg_weight_scale. Rebalances the two terms within a segment. NOTE "
                        "this cannot make command 'prune but not select': selection ranks on "
                        "the accumulated path_score, so earlier segments still carry it. Use "
                        "--final_weights for that.")
    p.add_argument("--final_weights", type=float, nargs=2, default=None,
                   metavar=("W_KIN", "W_CMD"),
                   help="re-rank the final full-horizon pool with these (w_kinematic, w_command) "
                        "and pick the winner from that, instead of by accumulated path_score. "
                        "Pruning is unaffected, so the pool and minADE_C do not change -- only "
                        "which surviving candidate is returned. '1 0' selects on kinematic alone "
                        "while still letting command steer the beam.")
    p.add_argument("--score_norm", choices=["zscore", "rank"], default="zscore",
                   help="how each score component is normalised across the candidate set before "
                        "the weighted sum. 'zscore' is the historical default and is destroyed by "
                        "outliers: score_kinematic spans ~8 orders of magnitude within one pool, "
                        "so one wild candidate owns the std and every plausible candidate "
                        "collapses to the same z (measured: top-3 spread 0.0024 while their true "
                        "errors spanned 37x). 'rank' keeps only the ordering, which is safe "
                        "because catastrophes are vetoed absolutely by the feasibility gate, "
                        "outside the normalisation.")
    p.add_argument("--w_likelihood", type=float, default=0.0,
                   help="weight of the MODEL SELF-LIKELIHOOD in the final re-rank (>0 enables it "
                        "and switches the re-rank on). Every other score reads the waypoints "
                        "only; this one is the model's own log-likelihood of the candidate given "
                        "the image, i.e. the only scene-grounded signal available without "
                        "training a world model. Costs one forward per final candidate (~|C| "
                        "extra forwards/record on top of ~20 generation calls). Combine with "
                        "--final_weights, e.g. '--final_weights 0 0 --w_likelihood 1' for "
                        "likelihood-only selection.")
    p.add_argument("--fuse", choices=["median", "mean"], default=None,
                   help="COMBINE the candidates instead of selecting one of them. Every other "
                        "intervention asked the score to rank better and all of them are null, "
                        "including a fitted nonlinear rule that bounds the ceiling (within-pool "
                        "rho 0.524 vs 0.519 for the best single component, still 0.029 m worse "
                        "than not deliberating). Fusion needs no ranking ability: avgL2 is a "
                        "DISTANCE, whose optimal point estimator is the median of the predictive "
                        "distribution rather than its mode, and greedy decoding returns the mode. "
                        "'median' is the default choice -- matched to the metric and far more "
                        "robust than the mean to averaging two different manoeuvres into one that "
                        "belongs to neither. CHECK THAT PER COMMAND, not just in aggregate.")
    p.add_argument("--fuse_top_m", type=int, default=0,
                   help="how many top-scored candidates enter the fusion; 0 = all. The score is "
                        "used only to short-list, never to pick -- it is measurably good at the "
                        "coarse job (random 4.656 -> 3.595) and saturated at the fine one (top1 "
                        "0.249). ** 1 is the no-op control: fusing one candidate IS selecting it, "
                        "so '--fuse median --fuse_top_m 1' must reproduce the unfused arm exactly. "
                        "Run it once per code change.**")
    p.add_argument("--fuse_scope", choices=["final", "segment"], default="final",
                   help="'final' fuses the last segment's beam-expanded pool with the pipeline "
                        "otherwise unchanged (20 calls). Cheap, but the 8 candidates come from "
                        "only 2 distinct first-two-segment paths, so a null is ambiguous between "
                        "'fusion does not work' and 'it was diluted'. 'segment' fuses at every "
                        "segment with no beam (12 calls -- CHEAPER than the selecting arm), so "
                        "every waypoint is fused and a null is a real null.")
    p.add_argument("--verbose_plan", action="store_true", default=False,
                   help="print per-segment GoT trace for every record (noisy)")
    # --- baseline / world model ---
    p.add_argument("--no_baseline", action="store_true", default=False,
                   help="skip the free-run baseline comparison")
    p.add_argument("--wm_path", default=None,
                   help="world-model checkpoint dir. Enables WM use (Context Update AND "
                        "plausibility scoring, each independently toggleable below). "
                        "WM must be trained by train_nuscenes_wm.py. If unset, Mode A.")
    p.add_argument("--no_wm_ctx", action="store_true", default=False,
                   help="with --wm_path: disable WM Context Update (Mode B next-frame prediction)")
    p.add_argument("--no_wm_score", action="store_true", default=False,
                   help="with --wm_path: disable WM plausibility scoring (two-stage rerank)")
    p.add_argument("--w_wm", type=float, default=1.0, help="weight of the WM plausibility score")
    p.add_argument("--wm_rerank_m", type=int, default=4,
                   help="candidates per segment that get a (costly) WM plausibility call")
    p.add_argument("--wm_device", type=int, default=None,
                   help="GPU index for the world model (default: same as --device). Put the WM on a "
                        "SEPARATE card from the base model to avoid 2x7B (28GB) on one 24GB GPU.")
    return p.parse_args()


def load_world_model(args):
    """Load the WM for Mode B. Must use mask_image_logits=False so generate_img
    can emit image tokens (see got_drive.world_model)."""
    from model import ChameleonXLLMXForConditionalGeneration_ck_action_head
    wm = ChameleonXLLMXForConditionalGeneration_ck_action_head.from_pretrained(
        args.wm_path,
        action_dim=args.action_dim,
        time_horizon=args.time_horizon,
        max_position_embeddings=args.max_seq_len,
        mask_image_logits=False,
        dropout=0.0,
        z_loss_weight=0.0,
        torch_dtype=torch.bfloat16,
    ).to(torch.device(f"cuda:{args.wm_device if args.wm_device is not None else args.device}"))
    wm.eval()
    return wm


def summarize(metric_list, per_step_list):
    """mean of each L2 key + mean per-step curve, over the ok records."""
    out = {}
    if not metric_list:
        return out
    for k in metric_list[0]:
        out[k] = round(float(np.mean([m[k] for m in metric_list])), 4)
    out["per_step_L2"] = np.round(np.mean(np.stack(per_step_list), axis=0), 4).tolist()
    return out


def summarize_collision(coll_list):
    """Mean of each collision key -> RATE (fraction of records that collide).

    Each entry is 0.0/1.0 per record, so the mean is the collision rate; also
    emitted as a percentage, the unit UniAD/ST-P3 tables use.
    """
    if not coll_list:
        return {}
    out = {"n_evaluated": len(coll_list)}
    for k in coll_list[0]:
        rate = float(np.mean([c[k] for c in coll_list]))
        out[k] = round(rate, 5)
        out[f"{k}_pct"] = round(rate * 100, 3)
    return out


def summarize_mean(dict_list):
    """Mean of every numeric key across a list of per-record dicts.

    Tolerates ragged keys (a record whose plan had no locatable selected
    candidate contributes oracle_* but no selection_*), so a few odd records
    cannot silently drop a whole metric.
    """
    if not dict_list:
        return {}
    out = {"n": len(dict_list)}
    for k in sorted({k for d in dict_list for k in d}):
        vals = [d[k] for d in dict_list if k in d]
        if vals:
            out[k] = round(float(np.mean(vals)), 4)
    return out


def _flatten(d, prefix=""):
    """Nested summary dict -> {dotted.key: float}, for across-seed aggregation.
    Non-numeric leaves (lists such as per_step_L2, strings, None) are dropped."""
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flatten(v, f"{prefix}{k}."))
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[f"{prefix}{k}"] = float(v)
    return out


class _CountingFn:
    """generate_fn wrapper counting model calls, for the inference-cost metric.

    GoT buys its accuracy with extra forward passes (1 greedy call vs
    beam-expanded resampling), so the cost has to be reported next to the gain.
    """

    def __init__(self, fn):
        self.fn = fn
        self.n = 0

    def __call__(self, *a, **kw):
        self.n += 1
        return self.fn(*a, **kw)

    def reset(self):
        self.n = 0


def main():
    args = get_args()
    # dict.fromkeys: de-duplicate while keeping order. A repeated seed would
    # otherwise iterate the same accumulator twice and double-count every record.
    seeds = list(dict.fromkeys(args.seeds if args.seeds else [args.seed]))
    set_seed(seeds[0])
    assert args.n_segments * args.segment_len == args.time_horizon, (
        f"n_segments*segment_len ({args.n_segments}*{args.segment_len}) != "
        f"time_horizon ({args.time_horizon})")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    with open(args.records_json) as f:
        records = json.load(f)
    if args.limit:
        records = records[: args.limit]

    # Refuse to run rather than silently evaluate a state-trained checkpoint on
    # stateless records: nothing downstream would fail, every arm would just be
    # quietly off-distribution. Same guard as eval_nuscenes.py, on purpose --
    # the GoT path has MORE ways to lose the channel (candidate generator and
    # greedy baseline are separate call sites).
    if args.with_state:
        missing = sum(1 for r in records if "state" not in r)
        if missing:
            raise SystemExit(
                f"--with_state given but {missing}/{len(records)} records carry no "
                f"`state`. Rebuild {args.records_json} with data/preprocess_nuscenes.py.")
        n_invalid = sum(1 for r in records if not r.get("state_valid", 1))
        print(f"[eval-got] ego status ON -- {n_invalid}/{len(records)} records have a "
              f"zeroed (scene-start) state")

    # model-free prior baseline: per-command mean GT trajectory
    if args.train_records_json:
        with open(args.train_records_json) as f:
            mean_src = json.load(f)
    else:
        print("[eval-got][warn] --train_records_json unset; mean-trajectory baseline uses the "
              "eval split itself (slightly optimistic).")
        mean_src = records
    mean_trajs = compute_mean_trajectories(mean_src, args.time_horizon)

    # optional UniAD/ST-P3 collision metric: obstacle boxes keyed by sample_token
    coll_boxes, coll_cfg = None, None
    if args.collision_json:
        from got_drive.collision_metric import CollisionConfig, trajectory_collisions
        with open(args.collision_json) as f:
            coll_boxes = {r["sample_token"]: r["agent_boxes"] for r in json.load(f)}
        coll_cfg = CollisionConfig(uniad_parity=(args.collision_parity == "uniad"))
        n_cov = sum(1 for r in records if r["sample_token"] in coll_boxes)
        print(f"[eval-got] collision metric ON ({args.collision_parity} parity): boxes for "
              f"{n_cov}/{len(records)} records (ego {coll_cfg.ego_length}x{coll_cfg.ego_width} m, "
              f"grid {coll_cfg.resolution} m, yaw={coll_cfg.apply_yaw}, "
              f"x_offset={coll_cfg.ego_x_offset}, "
              f"gt_mask={not args.no_collision_gt_mask})")

    item_processor = FlexARItemProcessor_Action_NuScenes(
        tokenizer=args.tokenizer_path,
        target_size=args.resolution,
        norm_path=args.norm_path,
    )
    print(f"[eval-got] waypoint un-norm range: min={item_processor.wp_min.tolist()} "
          f"max={item_processor.wp_max.tolist()}")
    check_action_grid(item_processor, args.norm_path,
                      allow_mini=args.allow_mini_grid)

    model = load_model(args)

    # generate_fn is image-agnostic (image is passed per plan()); build it once.
    # Wrapped so every model call is counted -> inference-cost metric.
    # `state_holder` carries the current record's ego status into that single fn;
    # the record loop writes it before each plan(). None throughout when
    # --with_state is off, which is byte-identical to the pre-ego-status build.
    state_holder = [None]
    generate_fn = _CountingFn(make_model_generate_fn(
        model, item_processor, args.prompt, args,
        state_holder=state_holder if args.with_state else None))

    # World-model uses (optional, needs a trained WM). Context Update (Mode B
    # next-frame prediction) and plausibility scoring are independent, so any of
    # {neither, ctx only, score only, both} can be ablated.
    context_update_fn = None
    wm_score_fn = None
    if args.wm_path:
        from got_drive.world_model import make_wm_context_update_fn, make_wm_score_fn
        print(f"[eval-got] loading world model from {args.wm_path}")
        wm = load_world_model(args)
        # bind to the WM's OWN device (may differ from the base model's when
        # --wm_device puts them on separate GPUs).
        if not args.no_wm_ctx:
            context_update_fn = make_wm_context_update_fn(wm, item_processor, wm.device)
        if not args.no_wm_score:
            wm_score_fn = make_wm_score_fn(wm, item_processor, wm.device)

    cfg = DriveGoTConfig(
        n_segments=args.n_segments,
        segment_len=args.segment_len,
        k_candidates=args.k_candidates,
        beam_width=args.beam_width,
        time_horizon=args.time_horizon,
        temperatures=tuple(args.temperatures),
        w_kinematic=args.w_kinematic,
        w_command=args.w_command,
        w_wm=args.w_wm,
        wm_rerank_m=args.wm_rerank_m,
        seg_weight_scale=tuple(args.seg_weight_scale) if args.seg_weight_scale else None,
        seg_w_command_scale=(tuple(args.seg_w_command_scale)
                             if args.seg_w_command_scale else None),
        final_weights=tuple(args.final_weights) if args.final_weights else None,
        w_likelihood=args.w_likelihood,
        score_norm=args.score_norm,
        fuse=args.fuse,
        fuse_top_m=args.fuse_top_m,
        fuse_scope=args.fuse_scope,
        verbose=args.verbose_plan,
    )
    if cfg.fuse is not None:
        print(f"[eval-got] FUSION on: {cfg.fuse}/{cfg.fuse_scope} "
              f"top_m={cfg.fuse_top_m or 'all'}  beam_width={cfg.beam_width}"
              + ("  (forced to 1 by per-segment fusion)"
                 if cfg.fuse_scope == "segment" else ""))
        print("[eval-got] this arm SELECTS NOTHING -> selection_gap / "
              "selection_rank / selection_top1 are undefined and will be absent "
              "from the summary. minADE_C stays comparable (the pool is "
              "unchanged in 'final' scope).")

    # ── intent arm: the hypothesis set, built once and printed ───────────────
    intent_set = None
    # The shared tuple, not a fourth hand-written copy of the same list: the arm
    # guards drifting apart is precisely how aggregate/improve lost their kwargs.
    if args.planner in INTENT_PLANNERS:
        from got_drive.graph import make_intent_grid, MIN_SEPARATION_M
        kw = {}
        if args.intent_speed_scales:
            kw["speed_scales"] = tuple(args.intent_speed_scales)
        if args.intent_curvatures:
            kw["curvatures"] = tuple(args.intent_curvatures)
        intent_set = make_intent_grid(**kw)
        n_calls = 1 + len(intent_set) * args.intent_variants
        print(f"[eval-got] INTENT arm: {len(intent_set)} intents x "
              f"{args.intent_variants} variants -> {n_calls} forward calls/record "
              f"(the incumbent makes {args.k_candidates * (1 + 2 * args.beam_width)}). "
              f"Report the call count with the L2.")
        print(f"[eval-got]   {[i.name for i in intent_set]}")
        print(f"[eval-got] ⚠️ this arm is NOT the control -- it cannot reproduce 3.5557. "
              f"Keep it out of the §1 tables.")
        print(f"[eval-got] ⚠️ check got_intent_separated before the L2: intents closer than "
              f"{MIN_SEPARATION_M} m do not move the model (§1.5, 0/600 tokens flipped).")

    # ── the arm's identity, resolved ONCE and printed ────────────────────────
    # ★Built here, used for every record AND written into summary.json, so the
    # summary cannot describe an arm other than the one that ran. The bug this
    # replaces: an inline if-chain gave --planner aggregate/improve an EMPTY kwarg
    # dict, GraphPlanner's graph_kind defaulted to "staged", and both arms ran the
    # incumbent control while the summary said otherwise. Printing it means a
    # mismatch is visible in the log before the run finishes, not after the table
    # is written (§9).
    arm_kw = planner_kwargs(args, intents=intent_set)
    if arm_kw:
        print(f"[eval-got] graph arm: --planner {args.planner} -> graph_kind="
              f"{arm_kw['graph_kind']!r}, keep_valid={arm_kw['keep_valid']}"
              + (f", variants={arm_kw['variants']}, anchor_steps={arm_kw['anchor_steps']}"
                 if "variants" in arm_kw else "")
              + (f", aggregate={arm_kw['aggregate_method']}"
                 f"/keep_inputs={arm_kw['aggregate_keep_inputs']}"
                 if "aggregate_method" in arm_kw else "")
              + (f", improve_tries={arm_kw['num_tries']}"
                 f"/aggregate={arm_kw['improve_aggregate']}"
                 if "num_tries" in arm_kw else ""))
        if arm_kw["keep_valid"]:
            print("[eval-got] ⚠️ --graph_keep_valid filters BEFORE KeepBestN, so "
                  "minADE_C and the selection gap are computed over the FEASIBLE "
                  "SUBSET -- a smaller denominator, not a better generator. "
                  "`got_keep_valid_dropped` in the csv says by how much; do not "
                  "compare minADE_C against an arm run without this flag.")

    # Scene-grounded final re-rank. Built from the SAME model as generation (no
    # extra checkpoint, no extra GPU), so it is only wired when asked for.
    lik_score_fn = None
    if args.w_likelihood != 0.0:
        from got_drive.segment_generation import make_likelihood_score_fn
        lik_score_fn = make_likelihood_score_fn(
            model, item_processor, args.prompt, args,
            state_holder=state_holder if args.with_state else None)
        print(f"[eval-got] model self-likelihood final re-rank ON (w={args.w_likelihood})")
    # accurate label: Mode B iff a context-update fn is active; +WM-score iff the
    # plausibility rerank is active (wm_path with both toggles off -> plain Mode A).
    mode = ("B (ctx-update)" if context_update_fn is not None else "A (prefix)") \
        + (" +WM-score" if wm_score_fn is not None else "")
    print(f"[eval-got] score_norm={cfg.score_norm}")
    print(f"[eval-got] mode={mode}  cfg: {cfg.n_segments}seg x {cfg.segment_len}wp, "
          f"k={cfg.k_candidates}, beam={cfg.beam_width}, temps={args.temperatures}, "
          f"w=(kin {cfg.w_kinematic}, cmd {cfg.w_command}, wm {cfg.w_wm})")

    # waypoints are 0.5 s apart: index 1 -> 1 s, 3 -> 2 s, 5 -> 3 s
    hz_idx = {"1s": 1, "2s": 3, "3s": 5}
    hz_idx = {k: v for k, v in hz_idx.items() if v < args.time_horizon}
    l2_key_names = [f"{p}@{h}" for h in hz_idx for p in ("L2", "avgL2")]
    rank_key = args.rank_key if args.rank_key in l2_key_names else f"avgL2@{list(hz_idx)[-1]}"
    n_infeasible_fused = 0
    tail_key = args.tail_key if args.tail_key in l2_key_names else f"L2@{list(hz_idx)[-1]}"
    if (rank_key, tail_key) != (args.rank_key, args.tail_key):
        print(f"[eval-got][warn] rank/tail key adjusted for time_horizon="
              f"{args.time_horizon}: rank={rank_key} tail={tail_key}")
    print(f"[eval-got] seeds={seeds}  rank_key={rank_key}  tail_key={tail_key}")

    rows = []
    # GoT sampling is stochastic -> one accumulator per seed. The greedy free-run
    # and the mean-trajectory prior are deterministic, so they are computed ONCE
    # per record and shared by every seed (re-running them would burn GPU for
    # bit-identical numbers).
    S = {s: {"metrics": [], "per_step": [], "coll": [], "oracle": [],
             "pair_got": [], "pair_base": [], "tail": [],
             "n_failed": 0, "sec": [], "calls": []} for s in seeds}
    base_metrics, base_per_step, base_coll, base_tail, base_sec = [], [], [], [], []
    mean_metrics, mean_coll = [], []
    n_base_failed = 0
    n_short_gt = 0          # records with fewer than time_horizon GT waypoints

    def _collide(traj, token, gt_traj, tag, row):
        """Collision metrics for one trajectory, or None if no boxes for it."""
        if coll_boxes is None or traj is None:
            return None
        boxes = coll_boxes.get(token)
        if boxes is None:
            return None
        cm, _ = trajectory_collisions(
            traj, boxes, hz_idx, coll_cfg,
            gt_traj=None if args.no_collision_gt_mask else gt_traj)
        row.update({f"{tag}_{k}": v for k, v in cm.items()})
        return cm

    for i, rec in enumerate(records):
        # Crop ONCE, deterministically, before any forward call. Otherwise every
        # generate() re-crops at a random offset and consumes global RNG, which is
        # what coupled the unseeded greedy baseline below to the GoT draw count and
        # made it drift with k/beam (§9). No-op on an already-cropped frame.
        image = crop_for_eval(Image.open(rec["images"][0]).convert("RGB"),
                              item_processor, args.legacy_random_crop)
        gt = np.array(rec["waypoints"], dtype=np.float64)
        command = rec["command"]
        token = rec["sample_token"]
        # Both arms read the ego status from here: the GoT candidate generator via
        # the closure built above, the greedy baseline via its own `state=` below.
        # Setting it in one place is what keeps them from drifting apart.
        state_holder[0] = rec["state"] if args.with_state else None
        if gt.shape[0] < args.time_horizon:
            # every predictor emits exactly time_horizon waypoints, so a short GT
            # would raise a broadcast error deep in l2_metrics after hours of GPU
            # work. Skip the record instead and surface the count.
            n_short_gt += 1
            continue
        gt_h = gt[: args.time_horizon]
        base_row = {"sample_token": token, "scene": rec["scene"], "command": command}

        # ---- model-free baseline: mean GT trajectory for this command ----
        mean_pred = mean_trajs.get(command, mean_trajs.get("__all__"))
        if mean_pred is not None:
            mm, _ = l2_metrics(mean_pred, gt_h, hz_idx)
            mean_metrics.append(mm)
            cm = _collide(mean_pred, token, gt_h, "mean", base_row)
            if cm is not None:
                mean_coll.append(cm)

        # ---- free-run baseline (same checkpoint, greedy single shot) ----
        base_m = None
        if not args.no_baseline:
            t_b = time.perf_counter()
            pred_base = predict_waypoints(model, item_processor, image, args.prompt, args,
                                          state=state_holder[0])
            base_sec.append(time.perf_counter() - t_b)
            if pred_base is None:
                n_base_failed += 1
                base_row["base_status"] = "malformed_generation"
            else:
                base_m, base_ps = l2_metrics(pred_base, gt_h, hz_idx)
                base_metrics.append(base_m)
                base_per_step.append(base_ps)
                base_tail.append(base_m[tail_key])
                base_row["base_status"] = "ok"
                base_row.update({f"base_{k}": round(v, 4) for k, v in base_m.items()})
                # the greedy WAYPOINTS, not just its metrics. Without these any
                # offline post-processing study (E3 smoothing) can only be run
                # on the GoT arm, and "smoothing helps GoT" cannot be told
                # apart from "smoothing helps any trajectory" -- the control is
                # the whole experiment. Same rounding as got_pred.
                base_row["base_pred"] = np.round(pred_base, 3).tolist()
                cm = _collide(pred_base, token, gt_h, "base", base_row)
                if cm is not None:
                    base_coll.append(cm)

        # ---- GoT plan, once per seed ----
        for s in seeds:
            acc = S[s]
            row = dict(base_row, seed=s)
            # derive the RNG stream from (seed, record index) so a record's plan is
            # reproducible regardless of --limit or where it sits in the file.
            # Modulo keeps it inside np.random.seed's [0, 2**32) domain -- without
            # it any --seeds value above ~4294 raises ValueError.
            set_seed((s * 1_000_003 + i) % (2 ** 31 - 1))
            # Same construction for both planners: GraphPlanner deliberately wears
            # DriveGoTPipeline's interface so this loop -- and every metric,
            # csv column and bootstrap downstream of it -- is shared, not forked.
            planner_cls = (GraphPlanner if args.planner in GRAPH_PLANNERS
                           else DriveGoTPipeline)
            # ★The kwargs live in got_drive/graph/planner.py, not inline here.
            # Inline, this was `if args.planner in ("graph", "intent"):` wrapping
            # every branch below it, so --planner aggregate and --planner improve
            # got an EMPTY dict, GraphPlanner defaulted graph_kind to "staged", and
            # both arms silently ran the INCUMBENT CONTROL ARM while summary.json
            # said "planner": "aggregate". The tell was the CALL COUNT: the control arm's
            # k*(1+2*beam) where the intent-based arms make 1 + n_intents*variants
            # (+ repairs). ⚠️those are config-dependent -- do not quote absolute
            # numbers here, quote the formula.
            # As a pure function the mapping is testable without torch, and
            # got_drive.graph.selftest now asserts every --planner value lands on
            # the graph_kind its name promises.
            planner_kw = arm_kw
            pipe = planner_cls(
                cfg, generate_fn,
                initial_image=image,
                context_update_fn=context_update_fn,
                wm_score_fn=wm_score_fn,
                lik_score_fn=lik_score_fn,
                **planner_kw,
            )
            generate_fn.reset()
            t_g = time.perf_counter()
            merged, _ = pipe.plan(command)
            if args.planner in INTENT_PLANNERS:
                # Logged per record, not aggregated: whether the intents separated
                # is a property of THIS scene's speed (a stopped ego makes every
                # multiplier collapse to ~0). An arm-level mean would hide the
                # records where the hypothesis set degenerated to one plan.
                sep = pipe.last_separation or {}
                row["got_intent_sep_min"] = sep.get("min_pairwise")
                row["got_intent_sep_max"] = sep.get("max_pairwise")
                row["got_intent_separated"] = sep.get("separated")
                row["got_intent_v0_step"] = sep.get("v0_step")
            if args.planner in AGGREGATE_PLANNERS:
                # ★WITHOUT THIS THE AGGREGATE ARM HAS NO PER-RECORD EVIDENCE.
                # `fuse_trajectories` returns its single input UNCHANGED, so an
                # intent with one surviving realisation aggregates nothing: an arm
                # where that happens on every intent of every record is a pure
                # identity wearing Aggregate's name, and its L2 would be reported
                # as an Aggregate result. `n_identity` is what says so. Same
                # discipline as got_intent_separated -- read it BEFORE the L2.
                agg = pipe.last_aggregate or {}
                row["got_aggregate_n"] = agg.get("n_aggregates")
                # Per intent, not summed: one intent starved to a single
                # realisation is invisible in a total.
                row["got_aggregate_inputs"] = agg.get("n_inputs")
                row["got_aggregate_identity"] = agg.get("n_identity")
                row["got_aggregate_infeasible_dropped"] = agg.get("n_infeasible_dropped")
            if args.planner in GRAPH_PLANNERS:
                # How many candidates KeepValid removed BEFORE KeepBestN, i.e. how
                # much smaller the pool minADE_C and the selection gap were
                # computed over. Always written (0 when --graph_keep_valid is off)
                # so the column exists in every graph-arm csv and a shrunken
                # denominator can never be mistaken for a better generator.
                kv = pipe.last_keep_valid or {}
                row["got_keep_valid_dropped"] = kv.get("n_dropped")
            if args.planner == "improve":
                # Cost here is DATA-DEPENDENT (only infeasible candidates are
                # repaired), so §1.5's free pairing does not hold for this arm.
                # Log the per-record call count and report its distribution.
                imp = pipe.last_improve or {}
                row["got_improve_invalid"] = imp.get("n_invalid")
                row["got_improve_repaired"] = imp.get("n_repaired")
                row["got_improve_calls"] = imp.get("n_generate_calls")
                row["got_improve_below_floor"] = imp.get("n_repairs_below_floor")
            if args.planner in GRAPH_PLANNERS and args.graph_trace_dir:
                Path(args.graph_trace_dir).mkdir(parents=True, exist_ok=True)
                pipe.last_controller.output_graph(
                    str(Path(args.graph_trace_dir) / f"{token}_seed{s}.json"))
            acc["sec"].append(time.perf_counter() - t_g)
            acc["calls"].append(generate_fn.n)

            if merged is None or merged.shape != (args.time_horizon, args.action_dim):
                acc["n_failed"] += 1
                row["got_status"] = "malformed_plan"
                rows.append(row)
                continue

            got_m, got_ps = l2_metrics(merged, gt_h, hz_idx)
            acc["metrics"].append(got_m)
            acc["per_step"].append(got_ps)
            acc["tail"].append(got_m[tail_key])
            row["got_status"] = "ok"
            row.update({f"got_{k}": round(v, 4) for k, v in got_m.items()})
            row["got_pred"] = np.round(merged, 3).tolist()
            if cfg.fuse is not None:
                # A fused trajectory is one NO candidate proposed, so it never
                # passed the feasibility veto that every candidate had to clear.
                # If fusion "wins" by emitting physically impossible paths that
                # happen to sit closer to the GT, that is a bug, not a result.
                row["got_fuse_n"] = list(pipe.last_fusion_n)
                row["got_fuse_feasible"] = bool(_feasible(merged))
                n_infeasible_fused += 0 if row["got_fuse_feasible"] else 1
            cm = _collide(merged, token, gt_h, "got", row)
            if cm is not None:
                acc["coll"].append(cm)

            # ---- oracle / selection gap: free, scores the pool GoT chose from ----
            cand_trajs, sel_idx = pipe.final_candidates()
            if cand_trajs:
                cand_metrics = [l2_metrics(c, gt_h, hz_idx)[0] for c in cand_trajs]
                os_m = oracle_and_selection(cand_metrics, sel_idx, rank_key=rank_key)
                acc["oracle"].append(os_m)
                row.update({f"got_{k}": round(v, 4) for k, v in os_m.items()})
                # keep the raw pool so the diagnosis can be redone without re-running
                row["got_cand_vals"] = [round(m[rank_key], 3) for m in cand_metrics]
                # ...and the candidates' actual WAYPOINTS. got_cand_vals says how
                # good each candidate was and got_cand_kin what the score thought
                # of it, but neither preserves the trajectory's SHAPE, so any
                # reranker richer than the hand-made scalars (raw geometry, or an
                # image+geometry model) cannot be trained offline without a fresh
                # GPU run. ~1 KB/row; the alternative is 2.4 h of GPU per seed.
                row["got_cand_wps"] = [np.round(t, 4).tolist() for t in cand_trajs]
                # ...and the score that produced the ranking, in the same order.
                # got_cand_vals says WHICH candidate was better; these say what the
                # score thought of it, so corr(component, -true_error) can name the
                # term to fix instead of guessing. Free -- already on the nodes.
                cs = pipe.final_candidate_scores()
                row["got_cand_kin"] = [round(v, 4) for v in cs["kinematic"]]
                row["got_cand_cmd"] = [round(v, 4) for v in cs["command"]]
                row["got_cand_total"] = [round(v, 4) for v in cs["path_score"]]
                if lik_score_fn is not None:
                    row["got_cand_lik"] = [round(v, 4) for v in cs["likelihood"]]
                # once a final re-rank runs, path_score is no longer the selection
                # criterion, so log the total that actually chose the winner
                if any(v == v for v in cs["final_score"]):        # not all-nan
                    row["got_cand_final"] = [round(v, 4) for v in cs["final_score"]]

            # ---- paired head-to-head against the shared greedy baseline ----
            if base_m is not None:
                acc["pair_got"].append(got_m[rank_key])
                acc["pair_base"].append(base_m[rank_key])
                row["got_beats_base"] = bool(got_m[rank_key] < base_m[rank_key])

            rows.append(row)

        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(records)}  (plans ok="
                  f"{sum(len(S[s]['metrics']) for s in seeds)}, got_failed="
                  f"{sum(S[s]['n_failed'] for s in seeds)}, base_failed={n_base_failed})")

    # ---- per-sample csv (one row per record x seed) ----
    csv_path = os.path.join(args.output_dir, "per_sample.csv")
    if rows:
        keys = sorted({k for r in rows for k in r})
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)

    def _round_pc(pc):
        out = {}
        for k, v in pc.items():
            if isinstance(v, list):
                out[k] = [round(x, 5) for x in v]
            elif isinstance(v, float):
                out[k] = round(v, 5)
            else:
                out[k] = v
        return out

    # ---- per-seed GoT summary ----
    per_seed = {}
    for s in seeds:
        acc = S[s]
        blk = {
            "n_evaluated": len(acc["metrics"]),
            "n_malformed_plan": acc["n_failed"],
            **summarize(acc["metrics"], acc["per_step"]),
        }
        if acc["oracle"]:
            blk["oracle_selection"] = summarize_mean(acc["oracle"])
        if acc["tail"]:
            blk["tail"] = {f"{tail_key}_{k}": round(v, 4) for k, v
                           in tail_stats(acc["tail"], args.catastrophe_m).items()}
        if acc["coll"]:
            blk["collision"] = summarize_collision(acc["coll"])
        if acc["pair_got"]:
            blk["vs_baseline_paired"] = _round_pc(paired_comparison(
                acc["pair_got"], acc["pair_base"], n_boot=args.n_boot, seed=s))
        if acc["sec"]:
            blk["cost"] = {
                "forward_calls_per_record": round(float(np.mean(acc["calls"])), 2),
                "sec_per_record": round(float(np.mean(acc["sec"])), 3),
            }
        per_seed[str(s)] = blk

    summary = {
        # ★THE RUN'S OWN CONFIGURATION, IN ITS OWN ARTIFACT.
        # A measurement audit found that summary.json recorded no arm settings at
        # all -- `headline/ref` and `headline/temp_tight` were distinguishable
        # only by directory name. That is exactly why the `--planner aggregate`
        # bug (which silently ran the control arm) could not be detected from the
        # artifacts, and it is the same class as §9's "the runner only overwrites
        # the arm it runs, so an old run stays in the table". Writing argv here
        # closes it: every number now carries the flags that produced it.
        "args": {k: (list(v) if isinstance(v, tuple) else v)
                 for k, v in sorted(vars(args).items())},
        "mode": mode,
        # Which planner produced these numbers. In the summary rather than only in
        # the shell history because §9's failure mode is exactly this: a runner
        # overwrites only the arm it ran, an old directory survives, and two arms
        # end up in one table with nothing in the artefacts saying they differ.
        "planner": args.planner,
        # ★READ OFF THE KWARGS THAT WERE ACTUALLY PASSED, not re-derived from
        # args. Re-derivation is what let "planner": "aggregate" sit next to an
        # arm that ran as "staged", and the same re-derived conditions here were
        # ALSO wrong (graph_keep_valid was reported only for graph/intent, and
        # anchor_steps only for intent, though aggregate and improve use both).
        # Reporting the real kwargs makes the summary an artefact OF the run
        # rather than a second opinion about it.
        "graph_kind": arm_kw.get("graph_kind"),
        "graph_keep_valid": arm_kw.get("keep_valid"),
        # The hypothesis set is part of the arm's identity: Step 2.8 showed GAP and
        # the sign-transition point are both functions of N, so a result quoted
        # without its intent set (and its N) is not interpretable.
        "intents": ([i.name for i in intent_set] if intent_set else None),
        "intent_variants": arm_kw.get("variants"),
        "improve_tries": arm_kw.get("num_tries"),
        "improve_aggregate": arm_kw.get("improve_aggregate"),
        "aggregate_method": arm_kw.get("aggregate_method"),
        "aggregate_keep_inputs": arm_kw.get("aggregate_keep_inputs"),
        "intent_anchor_steps": arm_kw.get("anchor_steps"),
        "ego_status": bool(args.with_state),
        "seeds": seeds,
        "n_records": len(records),
        "n_skipped_short_gt": n_short_gt,
        "rank_key": rank_key,
        "tail_key": tail_key,
        "got_per_seed": per_seed,
    }
    if len(seeds) > 1:
        summary["got_across_seeds"] = across_seeds(
            {s: _flatten(per_seed[str(s)]) for s in seeds})

    # ---- baselines (deterministic, shared across seeds) ----
    if not args.no_baseline:
        bblk = {
            "n_evaluated": len(base_metrics),
            "n_malformed_generation": n_base_failed,
            **summarize(base_metrics, base_per_step),
        }
        if base_tail:
            bblk["tail"] = {f"{tail_key}_{k}": round(v, 4) for k, v
                            in tail_stats(base_tail, args.catastrophe_m).items()}
        if base_coll:
            bblk["collision"] = summarize_collision(base_coll)
        if base_sec:
            bblk["cost"] = {"forward_calls_per_record": 1.0,
                            "sec_per_record": round(float(np.mean(base_sec)), 3)}
        summary["baseline_free_run"] = bblk

        # seed-averaged GoT minus baseline (negative = GoT better)
        deltas = {}
        for k in l2_key_names:
            vals = [per_seed[str(s)][k] for s in seeds if k in per_seed[str(s)]]
            if vals and k in bblk:
                deltas[k] = round(float(np.mean(vals)) - bblk[k], 4)
        summary["got_minus_baseline"] = deltas

    # model-free prior: GoT must clearly beat this to prove it uses the image.
    if mean_metrics:
        mblk = {"n_evaluated": len(mean_metrics),
                **{k: round(float(np.mean([m[k] for m in mean_metrics])), 4)
                   for k in mean_metrics[0]}}
        if mean_coll:
            mblk["collision"] = summarize_collision(mean_coll)
        summary["baseline_mean_traj"] = mblk

    if cfg.fuse is not None:
        summary["fusion"] = {
            "mode": cfg.fuse, "scope": cfg.fuse_scope,
            "top_m": cfg.fuse_top_m, "beam_width": cfg.beam_width,
            "n_infeasible_output": n_infeasible_fused,
            "note": ("the emitted trajectory is a combination, not a pool "
                     "member, so selection_gap/rank/top1 are undefined here. "
                     "n_infeasible_output > 0 means fusion produced paths that "
                     "would have been vetoed as physically impossible -- an L2 "
                     "gain carried by those is a bug, not a result."),
        }

    if coll_cfg is not None:
        summary["collision_config"] = {
            "parity": args.collision_parity,
            "ego_length": coll_cfg.ego_length, "ego_width": coll_cfg.ego_width,
            "resolution": coll_cfg.resolution,
            "apply_yaw": coll_cfg.apply_yaw, "ego_x_offset": coll_cfg.ego_x_offset,
            "gt_collision_mask": not args.no_collision_gt_mask,
            "note": ("coll@t = exactly at t (UniAD strategy); meanColl@t = mean of the "
                     "per-step rates over 0..t (ST-P3/VAD strategy); cumColl@t = any step "
                     "up to t (ours, no published table uses it). Read GoT against BOTH "
                     "baselines -- open-loop collision is weak on its own."),
        }

    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== nuScenes open-loop GoT planning ===")
    print(json.dumps(summary, indent=2))
    print(f"\nper-sample -> {csv_path}")

    # ---- headline read-out ----
    base_rk = summary.get("baseline_free_run", {}).get(rank_key)
    for s in seeds:
        blk = per_seed[str(s)]
        pc = blk.get("vs_baseline_paired", {})
        osel = blk.get("oracle_selection", {})
        print(f"\n[seed {s}] {rank_key}: GoT {blk.get(rank_key)}  baseline {base_rk}  "
              f"| win {pc.get('win_rate')}  p {pc.get('wilcoxon_p')}  "
              f"CI95 {pc.get('mean_diff_ci95')}")
        if osel:
            print(f"          oracle minADE_C {osel.get('minADE_C')}  "
                  f"selection_gap {osel.get(f'selection_gap_{rank_key}')}  "
                  f"top1 {osel.get('selection_top1')}  "
                  f"spread {osel.get('candidate_spread')}  "
                  f"n_cand {osel.get('n_candidates')}")
    # This note was rewritten 2026-08-12. The pre-session-7 version told the
    # reader that a large selection_gap means "the score is the bottleneck,
    # redesign scoring" -- and the data below is permanently in exactly that
    # condition, so it was handing every future reader a prescription that has
    # since been measured and falsified. Do not restore it.
    print("\nHow to read:")
    print("  * The verdict is 'got_minus_baseline' vs greedy (negative = GoT "
          "better), read WITH its CI. minADE_C and selection_gap describe the "
          "POOL; they are diagnostics, never the result.")
    print("  * A large selection_gap does NOT license 'redesign the scoring'. "
          "That was tried on exactly this condition (minADE_C 3.0053 < greedy "
          "3.5557, gap 0.6019): every rule buildable from the logged "
          "components -- hand-weighted, ridge, GBR, learned fallback, raw "
          "geometry -- lands in 62.8-63.8% recovery and ALL of them below "
          "greedy (handoff sec.1.3/1.6). Aggregating instead of selecting is "
          "null as well (sec.1.5).")
    print("  * minADE_C AT the baseline does NOT mean 'base not converged'. "
          "E1 trained 2 further epochs: null, +0.0494 with CI over 0 "
          "(sec.1.13).")
    print("  * ABSORPTION (sec.1.2/1.5): moving the pool moves the gap, not "
          "the output -- 103%/106% absorbed, and it holds in BOTH directions. "
          "A better pool is not evidence until got_minus_baseline moves.")
    print("  * The room in minADE_C is an order statistic over candidates near "
          "the decode, not a distillable target (displacement -0.1238, "
          "sec.1.8).")


if __name__ == "__main__":
    main()
