"""
Open-loop Graph-of-Thought trajectory planner for nuScenes driving.

Driving counterpart of got_vla_v2/got_pipeline.py (LIBERO). Same GoT skeleton --
Generate -> Score -> Aggregate -> Prune -> Merge -- and no env.step / Phase-2
execution (open-loop planning returns the merged trajectory for L2 eval).

Two ways to advance context between segments, selected by whether a
`context_update_fn` is supplied:

  Mode A (default, no world model).  The image is fixed for the whole horizon;
  the next segment is conditioned on the running PREFIX of already-decided
  waypoints (segment_generation.predict_segment). All waypoints live in the
  original ego frame, so merging is a plain concatenation.

  Mode B (world-model Context Update).  After committing a segment, the WM
  predicts the next front-camera frame (got_drive.world_model), and the next
  segment is generated FROM that predicted frame with a reset prefix -- exactly
  the LIBERO "Action -> State -> Action" edge, with the WM standing in for the
  simulator. Each segment's waypoints are then in a NEW ego frame, so the
  pipeline composes the per-segment ego poses to express the merged trajectory
  back in the original frame. WM Context Update runs only for the pruned
  survivors (image generation is expensive), matching the LIBERO pipeline.

Scoring (scoring_driving.rank_candidates) always operates on the cumulative
trajectory in the ORIGINAL frame, so it is identical across both modes.

Testability: `generate_fn` and `context_update_fn` are injected, so the whole
control flow (both modes, including frame composition) runs under
`python -m got_drive.got_pipeline_drive` with pure-numpy dummies -- no GPU.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np

from got_drive.fusion import FUSE_MODES, fuse_trajectories, top_m_indices
from got_drive.scoring_driving import rank_candidates, rank_candidates_wm


# ──────────────────────────────────────────────
# config / node
# ──────────────────────────────────────────────

@dataclass
class DriveGoTConfig:
    n_segments: int = 3
    segment_len: int = 2
    k_candidates: int = 4
    beam_width: int = 2
    time_horizon: int = 6
    temperatures: Tuple[float, ...] = (1.0, 1.2, 1.4, 1.6)  # per-candidate; idx0 greedy
    w_kinematic: float = 1.0
    w_command: float = 1.0
    w_wm: float = 1.0                # weight of the world-model plausibility score
    wm_rerank_m: int = 4             # short-list size for the (costly) WM re-rank
    # Per-segment multiplier on the score weights. Scaling every weight by the
    # same factor cannot reorder candidates WITHIN a segment (the components are
    # z-normalised, so a common positive factor is a monotone transform), but it
    # does change how much that segment contributes to the accumulated
    # path_score the beam sorts on. A 0.0 therefore means "this segment does not
    # vote": its candidates all tie at 0.0 and the stable sort keeps generation
    # order, so the greedy candidate (idx 0, temperature 1.0, do_sample=False)
    # survives the prune. The feasibility veto still applies, so a segment with
    # scale 0.0 still cannot promote a physically impossible candidate.
    # CAVEAT: a 0.0 segment abstains, it does not lock in greedy. The beam keeps
    # `beam_width` candidates from it, and a LATER segment's score still decides
    # between those branches -- so the finally selected trajectory can carry a
    # non-greedy first segment that a voting segment chose in hindsight. To force
    # greedy outright the beam would have to be narrowed for that segment.
    # Motivation: on the 500-record seed-42 run the per-step L2 delta vs greedy
    # was +0.055 at 0.5 s but -0.10 .. -0.12 from 1.5 s on, i.e. the score is a
    # net loss on the FIRST segment (at 0.5 s every candidate is smooth, so
    # kinematic carries no signal) and a net gain later. `(0, 1, 1)` tests that
    # directly. None = uniform 1.0 (unchanged behaviour).
    seg_weight_scale: Optional[Tuple[float, ...]] = None
    # Per-segment multiplier on w_command ALONE, applied on top of
    # seg_weight_scale. Unlike a common factor this DOES change the ranking
    # within a segment, because it shifts the balance between the two
    # z-normalised components.
    # Motivation: the score plays two different roles, and measurement says it
    # is good at one and bad at the other. Regressing each component against the
    # candidates' true error gives kinematic +0.55 on straights and +0.81 on
    # turns (never negative there), while command sits at +0.16..+0.20
    # everywhere -- and the combined path_score ranks WORSE than kinematic alone
    # on both (0.747 vs 0.811 on turns). So at final selection command is noise.
    # But deleting it outright (--w_command 0) made the turn pool itself worse
    # (minADE_C 1.4888 -> 1.5742), which selection cannot explain: command is
    # doing real work during beam PRUNING. The reason is that kinematic rewards
    # smoothness and the smoothest continuation of a turn is to stop turning, so
    # kinematic-only pruning kills the turning branches early and command is the
    # only term that keeps them alive. So command should steer the beam and then
    # stand aside for the final pick -- but note that muting it on the LAST
    # segment does not achieve that (see final_weights): selection ranks on the
    # ACCUMULATED path_score, which still carries command's contribution from the
    # earlier segments. This knob only rebalances the two terms per segment.
    seg_w_command_scale: Optional[Tuple[float, ...]] = None
    # (w_kinematic, w_command) used to re-rank the FINAL full-horizon pool and
    # choose the winner, replacing "highest accumulated path_score". This is what
    # actually separates the score's two jobs: the beam still prunes with the
    # normal weights (so command keeps the turning branches alive and the pool
    # stays good), and the final pick is then made on whatever ranks candidates
    # best. Measurement says that is kinematic alone -- it beats the combined
    # path_score on both turns (+0.811 vs +0.747) and straights (+0.546 vs
    # +0.543) -- so `(1.0, 0.0)` is the arm this exists for. None = unchanged.
    final_weights: Optional[Tuple[float, float]] = None
    # Weight of the model self-likelihood component in the FINAL re-rank
    # (segment_generation.trajectory_logprob, injected as lik_score_fn). > 0
    # switches the final re-rank on even without final_weights.
    # Why only at the final re-rank: the absorption result says improving the
    # POOL buys nothing -- enlarging it lowered minADE_C by 0.2840 (p<1e-4) and
    # the selection gap grew by 0.2962 (p<1e-4), leaving the output unchanged.
    # Selection is therefore the only stage where a new signal can matter, and
    # scoring only the final pool costs |C| forwards per record instead of
    # |C| x n_segments.
    w_likelihood: float = 0.0
    # Component normalisation before the weighted sum: "zscore" (historical) or
    # "rank" (outlier-robust). See scoring_driving._rank_norm -- on a measured
    # pool the three best candidates got z-scores spanning 0.0024 while their
    # true errors spanned 37x, i.e. the kinematic term contributed no ordering at
    # all. Reweighting a component that carries no ordering cannot change the
    # outcome, which makes this a candidate root cause for the null ablation.
    score_norm: str = "zscore"
    # ── fusion: combine candidates instead of selecting one (got_drive.fusion) ──
    # None = unchanged (argmax selection). "median" / "mean" = emit the
    # componentwise combination of the short-listed candidates instead.
    # Rationale in got_drive/fusion.py; in one line: avgL2 is a distance, whose
    # optimal point estimator is the median of the predictive distribution and
    # not its mode, and greedy decoding returns the mode -- so this is the one
    # remaining idea that does not need the score to rank better, which eleven
    # interventions and two learned upper bounds say it cannot.
    fuse: Optional[str] = None
    # how many top-scored candidates enter the fusion. 0 = all. 1 is the no-op
    # control: fusing one candidate IS selecting it, so `--fuse median
    # --fuse_top_m 1` must reproduce the unfused arm bit-for-bit. Run it.
    fuse_top_m: int = 0
    # "final"   fuse the last segment's beam-expanded pool, pipeline unchanged
    #           (20 forward calls). Cheap, but the 8 candidates come from only 2
    #           distinct first-two-segment paths, so only the last third of the
    #           trajectory actually gets averaged -- a null here is ambiguous
    #           between "fusion does not work" and "it was diluted".
    # "segment" fuse at every segment, no beam (12 calls -- CHEAPER than the
    #           selecting arm). Every waypoint is fused, so a null here is a
    #           real null. This is the arm the argument is actually about.
    fuse_scope: str = "final"
    verbose: bool = True

    def __post_init__(self):
        assert self.n_segments * self.segment_len == self.time_horizon, (
            f"n_segments*segment_len ({self.n_segments}*{self.segment_len}) "
            f"!= time_horizon ({self.time_horizon})")
        if self.fuse is not None:
            assert self.fuse in FUSE_MODES, (
                f"fuse must be one of {FUSE_MODES} or None, got {self.fuse!r}")
            assert self.fuse_scope in ("final", "segment"), (
                f"fuse_scope must be 'final' or 'segment', got {self.fuse_scope!r}")
            if self.fuse_scope == "segment" and self.beam_width != 1:
                # per-segment fusion leaves exactly one node alive, so a beam is
                # meaningless; more importantly, in Mode B `segment_local` of
                # nodes under different parents live in different frames and
                # must never be averaged together.
                object.__setattr__(self, "beam_width", 1)


@dataclass
class DrivePathNode:
    segment_local: np.ndarray                 # (segment_len, 2) in this segment's frame
    cum_traj: np.ndarray                      # cumulative waypoints, ORIGINAL frame
    end_pose: Tuple[np.ndarray, float]        # (position, heading) after this segment, orig frame
    image: object = None                      # obs frame used to generate THIS node's children
    segment_score: float = 0.0
    path_score: float = 0.0
    parent: Optional["DrivePathNode"] = None
    depth: int = -1
    kinematic: float = 0.0
    command: float = 0.0
    wm: float = 0.0
    likelihood: float = float("nan")   # model self-likelihood; nan = not scored
    final_score: float = float("nan")  # final re-rank total; nan if no re-rank


# ──────────────────────────────────────────────
# ego-frame composition (Mode B)
# ──────────────────────────────────────────────

def _rot(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def map_local_to_original(seg_local, base_p, base_theta):
    """Waypoints in a segment's local ego frame -> original frame."""
    return np.asarray(seg_local, dtype=np.float64) @ _rot(base_theta).T + base_p


def advance_pose(seg_orig, base_p):
    """New ego pose (position, heading) at the end of a segment, original frame."""
    new_p = seg_orig[-1].copy()
    d = seg_orig[-1] - (seg_orig[-2] if len(seg_orig) >= 2 else base_p)
    return new_p, float(np.arctan2(d[1], d[0]))


def collect_path_segments(node: DrivePathNode) -> np.ndarray:
    """Backtrack root->node; cumulative already holds the merged path (orig frame)."""
    return node.cum_traj if node.cum_traj.size else np.empty((0, 2), dtype=np.float64)


def _seg_weights(cfg: DriveGoTConfig, seg_idx: int):
    """(w_kinematic, w_command, w_wm) for one segment after seg_weight_scale.

    Segments past the end of the tuple reuse its last entry, so a 1-element
    scale applies uniformly and a short tuple never raises mid-run.
    """
    s = 1.0
    if cfg.seg_weight_scale:
        s = float(cfg.seg_weight_scale[min(seg_idx, len(cfg.seg_weight_scale) - 1)])
    cs = 1.0
    if cfg.seg_w_command_scale:
        cs = float(cfg.seg_w_command_scale[min(seg_idx, len(cfg.seg_w_command_scale) - 1)])
    return cfg.w_kinematic * s, cfg.w_command * s * cs, cfg.w_wm * s


# ──────────────────────────────────────────────
# generate_fn adapter (real model)
# ──────────────────────────────────────────────

def make_model_generate_fn(model, item_processor, prompt, args, state_holder=None):
    """generate_fn(image, prefix_wp, n_generate, temperature, do_sample) -> (n,2)|None.

    `state_holder` is a one-element mutable list carrying the CURRENT record's ego
    status, or None for the stateless (incumbent) setup. It has to be a holder and
    not a plain argument because DriveGoTPipeline fixes generate_fn's signature and
    the fn is built once for the whole run, while state is per record: the eval loop
    writes `state_holder[0]` before each plan() call.

    ★Default None keeps every existing caller byte-identical -- `predict_segment`
    then builds the same stateless conversation it always did. That matters: a
    state-trained checkpoint evaluated with GoT but WITHOUT the state channel is
    off-distribution and nothing raises, which is exactly the silent failure
    eval_nuscenes.py refuses to run into (see its --with_state guard).
    """
    from got_drive.segment_generation import predict_segment

    def generate_fn(image, prefix_wp, n_generate, temperature, do_sample):
        return predict_segment(
            model, item_processor, image, prompt, args,
            prefix_wp=prefix_wp, n_generate=n_generate,
            temperature=temperature, do_sample=do_sample,
            state=None if state_holder is None else state_holder[0],
        )

    return generate_fn


# ──────────────────────────────────────────────
# pipeline
# ──────────────────────────────────────────────

class DriveGoTPipeline:
    """Open-loop GoT planner. `plan(command)` returns the merged trajectory.

    context_update_fn(current_image, segment_wp_local) -> next_image enables
    Mode B (world-model Context Update). If None, runs Mode A (fixed image,
    token-prefix conditioning).

    After `plan()`, the last segment's scored candidate pool stays available via
    `final_candidates()`. GoT's whole claim is that its score picks a better
    trajectory than greedy, so the pool the score chose FROM is what separates
    "the generator had nothing good" from "the score failed to pick it"
    (oracle / selection-gap metrics in got_drive.planning_metrics). Keeping the
    pool costs nothing -- it is already generated, just discarded at the prune.
    """

    def __init__(self, cfg: DriveGoTConfig, generate_fn: Callable,
                 initial_image=None, context_update_fn: Optional[Callable] = None,
                 score_fn: Optional[Callable] = None,
                 wm_score_fn: Optional[Callable] = None,
                 lik_score_fn: Optional[Callable] = None):
        self.cfg = cfg
        self.generate_fn = generate_fn
        self.initial_image = initial_image
        self.context_update_fn = context_update_fn
        self.use_ctx_update = context_update_fn is not None
        self.score_fn = score_fn or self._default_score_fn
        # populated by plan(); see final_candidates()
        self.last_final_pool: List[DrivePathNode] = []
        self.last_selected_node: Optional[DrivePathNode] = None
        # how many candidates each fusion consumed, one entry per fused segment.
        # Empty when cfg.fuse is None. Logged so a "fusion did nothing" result
        # can be told apart from "fusion never ran".
        self.last_fusion_n: List[int] = []
        # wm_score_fn(image, segment_wp) -> plausibility float | None. If given,
        # each segment does a two-stage rerank: cheap kinematic+command short-list
        # -> WM plausibility only on the short-list -> combined prune. WM scoring
        # is cleanest in Mode B (segment_local is in the frame matching the
        # conditioning image); in Mode A it is an approximation (fixed t0 image).
        self.wm_score_fn = wm_score_fn
        self.use_wm_score = wm_score_fn is not None
        # lik_score_fn(image, cum_traj) -> float | None. Applied ONLY in the
        # final re-rank (see cfg.w_likelihood), on complete trajectories.
        self.lik_score_fn = lik_score_fn
        self.use_lik_score = lik_score_fn is not None

    @staticmethod
    def _default_score_fn(cum_trajs, command, cfg, seg_idx=0):
        w_kin, w_cmd, _ = _seg_weights(cfg, seg_idx)
        return rank_candidates(cum_trajs, command, weights=(w_kin, w_cmd),
                               norm=cfg.score_norm)

    def _safe_wm_score(self, image, segment_wp):
        """wm_score_fn guarded: any failure -> np.nan (neutral in the ranker)."""
        try:
            v = self.wm_score_fn(image, segment_wp)
        except Exception:
            return np.nan
        return np.nan if v is None else float(v)

    def _score_pool(self, pool, command, seg_idx=0):
        """(totals, components) aligned with pool. Without a WM score this is the
        injected score_fn (kinematic+command). With one, run the two-stage rerank:
        cheaply rank the pool, call the WM only on the top `wm_rerank_m`, then
        combine kinematic+command+WM (candidates without a WM call stay neutral).

        `seg_idx` is passed through so score_fn can weight segments differently
        (cfg.seg_weight_scale). An injected score_fn must accept it."""
        cfg = self.cfg
        cum = [n.cum_traj for n in pool]
        w_kin, w_cmd, w_wm = _seg_weights(cfg, seg_idx)
        if not self.use_wm_score:
            return self.score_fn(cum, command, cfg, seg_idx)

        cheap, _ = rank_candidates(cum, command, weights=(w_kin, w_cmd),
                                   norm=cfg.score_norm)
        m = min(cfg.wm_rerank_m, len(pool))
        top_idx = np.argsort(cheap)[::-1][:m]                # highest cheap score first
        wm = np.full(len(pool), np.nan)
        for i in top_idx:
            node = pool[i]
            # Mode B conditions the WM on the parent's (WM-predicted) frame; in
            # Mode A children carry no per-node frame, so the image is fixed at t0.
            img = (node.parent.image
                   if (self.use_ctx_update and node.parent is not None
                       and node.parent.image is not None)
                   else self.initial_image)
            wm[i] = self._safe_wm_score(img, node.segment_local)
        return rank_candidates_wm(cum, command, wm, weights=(w_kin, w_cmd, w_wm),
                                  norm=cfg.score_norm)

    def _make_node(self, parent: DrivePathNode, seg: np.ndarray) -> DrivePathNode:
        """Build a child node from one segment in the generator's frame.

        Factored out of _expand so that a FUSED segment (which no generator
        produced) goes through the identical frame composition -- otherwise the
        two paths could drift apart and a fusion arm would differ from a
        selecting arm for reasons other than the fusion itself.
        """
        base_p, base_theta = parent.end_pose
        seg_orig = (map_local_to_original(seg, base_p, base_theta)
                    if self.use_ctx_update else seg)
        cum = (seg_orig if parent.cum_traj.size == 0
               else np.vstack([parent.cum_traj, seg_orig]))
        return DrivePathNode(
            segment_local=seg, cum_traj=cum,
            end_pose=advance_pose(seg_orig, base_p),
            parent=parent,
        )

    def _expand(self, parent: DrivePathNode):
        """Generate k candidate segments off one parent -> list of DrivePathNode
        (image not yet set; filled for survivors after prune in Mode B)."""
        cfg = self.cfg
        if self.use_ctx_update:
            image, prefix = parent.image, None            # fresh frame, reset prefix
        else:
            image = self.initial_image
            prefix = parent.cum_traj if parent.cum_traj.shape[0] > 0 else None

        out, seen = [], set()
        for i in range(cfg.k_candidates):
            temp = cfg.temperatures[min(i, len(cfg.temperatures) - 1)]
            seg = self.generate_fn(image, prefix, cfg.segment_len, temp, do_sample=(i > 0))
            if seg is None:
                continue
            seg = np.asarray(seg, dtype=np.float64)
            if seg.shape != (cfg.segment_len, 2):
                continue
            key = seg.round(4).tobytes()
            if key in seen:
                continue
            seen.add(key)
            out.append(self._make_node(parent, seg))
        return out

    def _shortlist(self, pool: List[DrivePathNode], scores: List[float]):
        """Top-m of the pool by score, best first.

        The score is used ONLY to short-list, never to pick a winner -- that is
        the whole point of this arm. Measurement says the score is good at the
        coarse job (random 4.656 -> 3.595, 63% of the way to the oracle) and
        saturated at the fine one (top1 0.249, unmoved by eleven interventions
        and by two learned upper bounds), so it is asked to do the first and
        stood down from the second.
        """
        return [pool[i] for i in top_m_indices(scores, self.cfg.fuse_top_m)]

    @staticmethod
    def _copy_scores(node: DrivePathNode, chosen: List[DrivePathNode]) -> None:
        """Mean of the fused candidates' components onto the fused node.

        Bookkeeping for the csv only -- nothing ranks on these, because after
        fusion there is nothing left to rank.
        """
        node.segment_score = float(np.mean([n.segment_score for n in chosen]))
        node.kinematic = float(np.mean([n.kinematic for n in chosen]))
        node.command = float(np.mean([n.command for n in chosen]))
        node.wm = float(np.mean([n.wm for n in chosen]))
        node.depth = chosen[0].depth

    def _fuse_segment(self, pool: List[DrivePathNode],
                      parent: DrivePathNode) -> Optional[DrivePathNode]:
        """Per-segment fusion: combine this segment and continue from it.

        Fuses `segment_local` and rebuilds the node through _make_node, so the
        fused segment goes through exactly the same frame composition a
        generated one would.
        """
        chosen = self._shortlist(pool, [n.path_score for n in pool])
        if not chosen:
            return None
        seg = fuse_trajectories([n.segment_local for n in chosen], self.cfg.fuse)
        if seg is None:
            return None
        node = self._make_node(parent, seg)
        self._copy_scores(node, chosen)
        node.path_score = parent.path_score + node.segment_score
        self.last_fusion_n.append(len(chosen))
        return node

    def _fuse_final(self, pool: List[DrivePathNode],
                    scores: List[float]) -> Optional[DrivePathNode]:
        """Final-scope fusion: combine the COMPLETE trajectories.

        Fusing cum_traj rather than only the last segment is what makes this a
        test of fusion at all -- the eight candidates come from two distinct
        first-two-segment paths, and fusing the full trajectory is what averages
        across those two. It still only has two distinct values to average over
        the early waypoints, which is the dilution that fuse_scope="segment"
        exists to remove.
        """
        chosen = self._shortlist(pool, scores)
        if not chosen:
            return None
        cum = fuse_trajectories([n.cum_traj for n in chosen], self.cfg.fuse)
        if cum is None:
            return None
        node = DrivePathNode(
            segment_local=cum[-self.cfg.segment_len:].copy(),
            cum_traj=cum,
            end_pose=advance_pose(cum, np.zeros(2)),
            parent=chosen[0].parent,
        )
        self._copy_scores(node, chosen)
        node.path_score = float(np.mean([n.path_score for n in chosen]))
        self.last_fusion_n.append(len(chosen))
        return node

    def _final_pool_nodes(self) -> List[DrivePathNode]:
        """Last plan()'s full-horizon candidate nodes, best first -- by
        accumulated path_score, or by cfg.final_weights when a final re-rank ran.
        Single source of order for final_candidates() and
        final_candidate_scores(), so the trajectories and their score components
        can never drift out of alignment."""
        T = self.cfg.time_horizon
        return [n for n in self.last_final_pool if n.cum_traj.shape[0] == T]

    def final_candidates(self) -> Tuple[List[np.ndarray], Optional[int]]:
        """Full-horizon candidates from the last plan(), and which one GoT chose.

        Returns (trajs, selected_idx). Only candidates that actually reached
        time_horizon waypoints are returned, so oracle/minADE is computed over
        realizable trajectories; selected_idx indexes into `trajs` (None if the
        chosen node was dropped by that filter, or if plan() produced nothing).

        NOTE these are the last segment's beam-expanded candidates: they share
        prefixes with each other (beam search), so this is "minADE over the final
        candidate set", NOT minADE_k over k independent samples. Report it as
        such -- calling it minADE_k would overstate the diversity.
        """
        trajs: List[np.ndarray] = []
        sel: Optional[int] = None
        for node in self._final_pool_nodes():
            if node is self.last_selected_node:
                sel = len(trajs)
            trajs.append(node.cum_traj)
        return trajs, sel

    def final_candidate_scores(self) -> dict:
        """Per-candidate score components, aligned with final_candidates()[0].

        The oracle/selection metrics say the score picks the pool's best
        candidate only ~27% of the time, but they cannot say WHY, because the
        score itself is thrown away at the prune. This returns it, so the true
        error of each candidate (already logged as got_cand_vals) can be
        regressed against the term that produced the ranking:

          kinematic / command  raw, un-z-normalised component values from the
                               FINAL segment's ranking. Raw because the z-norm
                               is per-pool: comparing z-scores across records
                               would compare different normalisations.
          wm                   world-model plausibility, nan when not scored.
          likelihood           model self-likelihood of the complete trajectory
                               given the image (segment_generation.
                               trajectory_logprob), nan when not scored. The
                               only component here that looks at the pixels.
          segment_score        weighted, z-normalised total for the final
                               segment, feasibility veto applied.
          path_score           accumulated total over all segments -- THIS is
                               what the beam sorts on, i.e. the actual selection
                               criterion. Correlate against -true_error first.

        Costs nothing: every value is already on the node. Empty lists if
        plan() produced no full-horizon candidate.
        """
        nodes = self._final_pool_nodes()
        return {
            "kinematic": [float(n.kinematic) for n in nodes],
            "command": [float(n.command) for n in nodes],
            "wm": [float(n.wm) for n in nodes],
            "likelihood": [float(n.likelihood) for n in nodes],
            "final_score": [float(n.final_score) for n in nodes],
            "segment_score": [float(n.segment_score) for n in nodes],
            "path_score": [float(n.path_score) for n in nodes],
        }

    def plan(self, command: str) -> Tuple[Optional[np.ndarray], Optional[DrivePathNode]]:
        cfg = self.cfg
        t0 = time.time()
        # reset per-plan state: a previous record's pool must never leak into
        # this record's oracle/selection metrics.
        self.last_final_pool = []
        self.last_selected_node = None
        mode = "B/ctx-update" if self.use_ctx_update else "A/prefix"
        if cfg.verbose:
            print(f"[DriveGoT|{mode}] command={command}  {cfg.n_segments}seg x "
                  f"{cfg.segment_len}wp, k={cfg.k_candidates}, beam={cfg.beam_width}")

        root = DrivePathNode(
            segment_local=np.empty((0, 2)), cum_traj=np.empty((0, 2)),
            end_pose=(np.zeros(2), 0.0), image=self.initial_image, depth=-1,
        )
        beam: List[DrivePathNode] = [root]

        for seg_idx in range(cfg.n_segments):
            pool: List[DrivePathNode] = []
            for parent in beam:
                pool.extend(self._expand(parent))
            if not pool:
                if cfg.verbose:
                    print(f"  [seg {seg_idx+1}] no candidates -> early stop")
                break

            totals, comp = self._score_pool(pool, command, seg_idx)
            wm_comp = comp.get("wm")
            for i, (node, tscore, kv, cv) in enumerate(
                    zip(pool, totals, comp["kinematic"], comp["command"])):
                node.segment_score = float(tscore)
                node.path_score = node.parent.path_score + float(tscore)
                node.depth = seg_idx
                node.kinematic, node.command = float(kv), float(cv)
                if wm_comp is not None:
                    node.wm = float(wm_comp[i])

            pool.sort(key=lambda n: n.path_score, reverse=True)
            # keep the full scored pool of the DEEPEST segment reached; overwritten
            # each segment so this ends up holding the final-horizon candidates.
            self.last_final_pool = pool

            if cfg.fuse is not None and cfg.fuse_scope == "segment":
                # One node survives each segment, so every parent in `pool` is
                # the same node -- required, because in Mode B segment_local of
                # nodes under different parents live in different frames.
                parent = pool[0].parent
                assert all(n.parent is parent for n in pool), (
                    "per-segment fusion requires a single parent; beam_width "
                    "should have been forced to 1")
                fused = self._fuse_segment(pool, parent)
                if fused is None:
                    if cfg.verbose:
                        print(f"  [seg {seg_idx+1}] fusion produced nothing "
                              f"-> falling back to the top candidate")
                    beam = pool[:1]
                else:
                    beam = [fused]
            else:
                beam = pool[: cfg.beam_width]

            # Context Update: only for survivors, and not needed after the last segment
            if self.use_ctx_update and seg_idx < cfg.n_segments - 1:
                for node in beam:
                    node.image = self.context_update_fn(node.parent.image, node.segment_local)

            if cfg.verbose:
                for r, n in enumerate(beam):
                    print(f"  [seg {seg_idx+1}] keep{r+1}: seg={n.segment_score:+.3f} "
                          f"path={n.path_score:+.3f} kin={n.kinematic:.2f} cmd={n.command:+.2f} "
                          f"end_y={n.cum_traj[-1,1]:+.2f}")

        if not beam or beam[0].depth < 0:
            return None, None

        # Final re-rank: pick the winner by a dedicated weighting of the complete
        # trajectories instead of by accumulated path_score. Pruning is already
        # done, so this changes only WHICH of the surviving candidates is
        # returned -- the pool itself (and therefore minADE_C) is untouched.
        # last_final_pool is re-sorted with it so final_candidates() and
        # final_candidate_scores() keep reporting the selected candidate first.
        if (cfg.final_weights is not None or self.use_lik_score) and self.last_final_pool:
            pool = self.last_final_pool
            cum = [n.cum_traj for n in pool]
            w_kin, w_cmd = (tuple(cfg.final_weights) if cfg.final_weights is not None
                            else (cfg.w_kinematic, cfg.w_command))
            if self.use_lik_score:
                # One forward per candidate. Failures -> nan, which
                # rank_candidates_wm's third slot treats as a neutral 0 rather
                # than letting one bad candidate shift the whole z-score.
                lik = np.full(len(pool), np.nan)
                for i, node in enumerate(pool):
                    try:
                        v = self.lik_score_fn(self.initial_image, node.cum_traj)
                    except Exception:
                        v = None
                    lik[i] = np.nan if v is None else float(v)
                    node.likelihood = lik[i]
                totals, _ = rank_candidates_wm(
                    cum, command, lik,
                    weights=(w_kin, w_cmd, cfg.w_likelihood),
                    norm=cfg.score_norm)
            else:
                totals, _ = rank_candidates(cum, command, weights=(w_kin, w_cmd),
                                            norm=cfg.score_norm)
            # keep it: once a re-rank runs, path_score is no longer the
            # criterion, so without this the csv cannot explain the pick
            for n_, t_ in zip(pool, totals):
                n_.final_score = float(t_)
            order = sorted(range(len(pool)), key=lambda i: -totals[i])
            self.last_final_pool = [pool[i] for i in order]
            beam = self.last_final_pool[: cfg.beam_width]
            if cfg.verbose:
                lw = f", lik {cfg.w_likelihood}" if self.use_lik_score else ""
                print(f"  [final re-rank w=({w_kin}, {w_cmd}){lw}] winner was "
                      f"path_score rank {order[0]} of {len(pool)}")

        # Final-scope fusion: the pipeline ran unchanged (beam, 20 calls) and we
        # now emit the combination of the short-list instead of its argmax. The
        # returned trajectory is deliberately NOT a pool member, so
        # final_candidates() reports selected_idx=None and the eval's
        # selection_gap / selection_rank / top1 are undefined for this arm --
        # correct, because nothing was selected. minADE_C stays meaningful: the
        # pool is untouched, so the generator ceiling is still comparable.
        if cfg.fuse is not None and cfg.fuse_scope == "final" and self.last_final_pool:
            pool = self.last_final_pool
            # short-list on whatever actually ordered the pool: the final
            # re-rank's score when one ran, the accumulated path_score otherwise
            key = ("final_score" if any(n.final_score == n.final_score for n in pool)
                   else "path_score")
            fused = self._fuse_final(pool, [getattr(n, key) for n in pool])
            if fused is not None:
                self.last_selected_node = None      # not a pool member
                if cfg.verbose:
                    print(f"  [fuse {cfg.fuse}/final m={cfg.fuse_top_m or len(pool)}] "
                          f"emitting a combination of {self.last_fusion_n[-1]} "
                          f"candidates (no selection)")
                return fused.cum_traj, fused

        best = beam[0]
        self.last_selected_node = best
        if cfg.verbose:
            print(f"[DriveGoT|{mode}] done {time.time()-t0:.2f}s  "
                  f"path_score={best.path_score:+.3f}  {len(best.cum_traj)} waypoints")
        return best.cum_traj, best


# ──────────────────────────────────────────────
# self-test: both modes with pure-numpy dummies (no GPU)
# ──────────────────────────────────────────────

def _make_dummy_generate_fn(step_x=4.0):
    """Deterministic generator: continues from the frame's local origin,
    advancing ~step_x per waypoint with a per-call lateral drift menu."""
    menu = [0.0, +0.6, -0.6, +1.4, -1.4]
    counter = {"i": 0}

    def gen(image, prefix_wp, n_generate, temperature, do_sample):
        # local frame: if a prefix is given (Mode A) continue from its end,
        # else (Mode B) start from local origin (0,0).
        if prefix_wp is not None and len(prefix_wp) > 0:
            x0, y0 = float(prefix_wp[-1][0]), float(prefix_wp[-1][1])
        else:
            x0, y0 = 0.0, 0.0
        drift = menu[0] if not do_sample else menu[counter["i"] % len(menu)]
        if do_sample:
            counter["i"] += 1
        x, y, seg = x0, y0, []
        for _ in range(n_generate):
            x += step_x
            y += drift
            seg.append([x, y])
        return np.array(seg, dtype=np.float64)

    return gen


def _run(command, use_ctx):
    cfg = DriveGoTConfig(verbose=True)
    ctx_fn = (lambda img, seg: img) if use_ctx else None   # identity dummy WM
    pipe = DriveGoTPipeline(cfg, _make_dummy_generate_fn(),
                            initial_image="frame0" if use_ctx else None,
                            context_update_fn=ctx_fn)
    merged, best = pipe.plan(command)
    shape_ok = merged is not None and merged.shape == (cfg.time_horizon, 2)
    x_mono = shape_ok and np.all(np.diff(merged[:, 0]) > 0)
    return merged, shape_ok, x_mono


def _run_wm(command, use_ctx=True):
    """wm_score_fn two-stage rerank plumbing, in Mode A (use_ctx=False, fixed t0
    image) and Mode B (use_ctx=True). Dummy WM = 'a low-lateral segment gives a
    more plausible future'. The wm_fn requires a non-None image, so this also
    guards the Mode A regression where child nodes carry no per-node frame."""
    cfg = DriveGoTConfig(verbose=True, w_wm=1.0, wm_rerank_m=3)
    def wm_fn(image, seg):
        assert image is not None, "wm_score_fn received image=None"
        return -float(abs(np.asarray(seg, dtype=np.float64)[-1, 1]))
    pipe = DriveGoTPipeline(cfg, _make_dummy_generate_fn(),
                            initial_image="frame0",
                            context_update_fn=(lambda img, seg: img) if use_ctx else None,
                            wm_score_fn=wm_fn)
    merged, best = pipe.plan(command)
    shape_ok = merged is not None and merged.shape == (cfg.time_horizon, 2)
    x_mono = shape_ok and np.all(np.diff(merged[:, 0]) > 0)
    # winner was in the WM short-list -> its wm component is finite (not the
    # neutral nan given to un-scored candidates); confirms WM participated.
    wm_populated = best is not None and np.isfinite(best.wm)
    return shape_ok, x_mono, wm_populated


def _run_pool():
    """final_candidates(): full-horizon pool exposed, selected node locatable,
    and per-plan state reset (a stale pool would silently corrupt oracle stats)."""
    cfg = DriveGoTConfig(verbose=False)
    pipe = DriveGoTPipeline(cfg, _make_dummy_generate_fn(), initial_image=None)

    merged, best = pipe.plan("straight")
    trajs, sel = pipe.final_candidates()
    ok = merged is not None and len(trajs) > 1
    ok &= all(t.shape == (cfg.time_horizon, 2) for t in trajs)
    ok &= sel is not None and np.allclose(trajs[sel], merged)
    # the selected node is the highest path_score, i.e. first in the sorted pool
    ok &= sel == 0
    print(f"  [pool] n_candidates={len(trajs)} selected_idx={sel} "
          f"shapes_ok={all(t.shape == (cfg.time_horizon, 2) for t in trajs)}")

    # score components must be index-aligned with the trajectories (they are
    # written to the same csv row and correlated pairwise downstream), and
    # path_score must reproduce the selection: descending, max at `sel`.
    sc = pipe.final_candidate_scores()
    len_ok = all(len(v) == len(trajs) for v in sc.values())
    ps = sc["path_score"]
    sorted_ok = ps == sorted(ps, reverse=True)
    sel_ok = sel is not None and ps[sel] == max(ps)
    print(f"  [scores] keys={sorted(sc)} aligned={len_ok} "
          f"path_score_desc={sorted_ok} argmax==selected={sel_ok}")
    ok &= len_ok and sorted_ok and sel_ok

    # a plan that produces nothing must clear the pool, not keep the old one
    dead = DriveGoTPipeline(cfg, lambda *a, **k: None, initial_image=None)
    dead.last_final_pool, dead.last_selected_node = list(pipe.last_final_pool), best
    m2, _ = dead.plan("straight")
    t2, s2 = dead.final_candidates()
    reset_ok = m2 is None and t2 == [] and s2 is None
    print(f"  [pool] stale-state reset on failed plan: {reset_ok}")
    return ok and reset_ok


def _run_seg_scale():
    """seg_weight_scale: the score's per-segment vote can be switched off.

    Invariant that makes the flag trustworthy: with EVERY segment scaled to 0
    the score never votes, so the beam can only keep generation order and GoT
    must degenerate to the greedy path -- even under a 'left' command, which
    with uniform weights pulls the trajectory to +y (asserted in _run above).
    If a scale of 0 still moved the result, the weights would be leaking in
    somewhere and every ablation number would be suspect.
    """
    ok = True

    # weight arithmetic, incl. the short-tuple reuse rule
    c = DriveGoTConfig(w_kinematic=2.0, w_command=3.0, w_wm=4.0,
                       seg_weight_scale=(0.0, 0.5), verbose=False)
    w0, w1, w2 = _seg_weights(c, 0), _seg_weights(c, 1), _seg_weights(c, 2)
    arith_ok = (w0 == (0.0, 0.0, 0.0) and w1 == (1.0, 1.5, 2.0) and w2 == w1)
    none_ok = _seg_weights(DriveGoTConfig(w_kinematic=2.0, w_command=3.0,
                                          w_wm=4.0), 0) == (2.0, 3.0, 4.0)
    print(f"  [seg_scale] weights seg0={w0} seg1={w1} seg2(reuse)={w2} ok={arith_ok}; "
          f"None->unscaled={none_ok}")
    ok &= arith_ok and none_ok

    # command-only scale: hits w_command and nothing else, and composes with
    # seg_weight_scale rather than replacing it
    cc = DriveGoTConfig(w_kinematic=2.0, w_command=3.0, w_wm=4.0,
                        seg_w_command_scale=(1.0, 1.0, 0.0), verbose=False)
    c0, c2 = _seg_weights(cc, 0), _seg_weights(cc, 2)
    cmd_ok = c0 == (2.0, 3.0, 4.0) and c2 == (2.0, 0.0, 4.0)
    both = DriveGoTConfig(w_kinematic=2.0, w_command=3.0, w_wm=4.0,
                          seg_weight_scale=(0.5,), seg_w_command_scale=(2.0,),
                          verbose=False)
    compose_ok = _seg_weights(both, 0) == (1.0, 3.0, 2.0)   # cmd: 3*0.5*2
    print(f"  [cmd_scale] seg0={c0} seg2={c2} cmd_only={cmd_ok}; "
          f"composes with seg_weight_scale={compose_ok}")
    ok &= cmd_ok and compose_ok

    # final_weights: the pick changes, the POOL does not. That separation is the
    # whole point -- command must keep steering the beam (it is what keeps the
    # turning branches alive) while the final choice is made on kinematic alone.
    # Also guards the ordering contract: final_candidates() must report the
    # re-ranked winner at index 0, or every selection metric silently misreads.
    base_pipe = DriveGoTPipeline(DriveGoTConfig(verbose=False),
                                 _make_dummy_generate_fn(), initial_image=None)
    m_base, _ = base_pipe.plan("left")
    pool_base = sorted(round(float(t[-1, 1]), 4)
                       for t in base_pipe.final_candidates()[0])

    fw_pipe = DriveGoTPipeline(
        DriveGoTConfig(verbose=False, final_weights=(1.0, 0.0)),
        _make_dummy_generate_fn(), initial_image=None)
    m_fw, _ = fw_pipe.plan("left")
    trajs, sel = fw_pipe.final_candidates()
    pool_fw = sorted(round(float(t[-1, 1]), 4) for t in trajs)

    # lik_score_fn: a scene-grounded component injected into the same final
    # re-rank. Dummy scorer prefers a LEFT-ending trajectory, so under a
    # 'straight' command (which kinematic+command both push towards y=0) a large
    # enough likelihood weight must be able to overturn the pick -- otherwise the
    # component is wired in but inert, which would look exactly like a null
    # result on real data.
    calls = {"n": 0}

    def dummy_lik(image, traj):
        calls["n"] += 1
        return float(np.asarray(traj, dtype=np.float64)[-1, 1])   # reward +y

    lik_pipe = DriveGoTPipeline(
        DriveGoTConfig(verbose=False, w_likelihood=50.0),
        _make_dummy_generate_fn(), initial_image="frame0", lik_score_fn=dummy_lik)
    m_lik, _ = lik_pipe.plan("straight")
    lt, ls = lik_pipe.final_candidates()
    lik_vals = lik_pipe.final_candidate_scores()["likelihood"]
    lik_on = m_lik is not None and m_lik[-1, 1] > 0.5           # pulled off y=0
    lik_scored = calls["n"] == len(lt) and all(np.isfinite(v) for v in lik_vals)
    lik_sel = ls == 0 and np.allclose(lt[0], m_lik)
    # a failing scorer must degrade to neutral, not crash the plan
    bad_pipe = DriveGoTPipeline(
        DriveGoTConfig(verbose=False, w_likelihood=50.0),
        _make_dummy_generate_fn(), initial_image="frame0",
        lik_score_fn=lambda i, t: (_ for _ in ()).throw(RuntimeError("boom")))
    m_bad, _ = bad_pipe.plan("straight")
    bad_ok = m_bad is not None and m_bad.shape == (6, 2)
    print(f"  [likelihood] final_y={m_lik[-1, 1]:+.2f} (want >0, overturned) "
          f"overturned={lik_on}  one_call_per_candidate={lik_scored} "
          f"({calls['n']} calls / {len(lt)} cands)  selected_at_idx0={lik_sel}  "
          f"scorer_exception_survived={bad_ok}")
    ok &= lik_on and lik_scored and lik_sel and bad_ok

    pick_differs = m_fw is not None and not np.allclose(m_fw, m_base)
    pool_same = pool_base == pool_fw
    sel_ok = sel == 0 and np.allclose(trajs[0], m_fw)
    sc_ok = len(fw_pipe.final_candidate_scores()["kinematic"]) == len(trajs)
    print(f"  [final_weights] (1,0) final_y={m_fw[-1, 1]:+.2f} vs default "
          f"{m_base[-1, 1]:+.2f} -> pick_differs={pick_differs}, "
          f"pool_unchanged={pool_same}, selected_at_idx0={sel_ok}, "
          f"scores_aligned={sc_ok}")
    ok &= pick_differs and pool_same and sel_ok and sc_ok

    # all-zero scale must reproduce the greedy path under a 'left' command
    cfg0 = DriveGoTConfig(verbose=False, seg_weight_scale=(0.0, 0.0, 0.0))
    m0, _ = DriveGoTPipeline(cfg0, _make_dummy_generate_fn(),
                             initial_image=None).plan("left")
    greedy_ok = (m0 is not None and m0.shape == (cfg0.time_horizon, 2)
                 and np.allclose(m0[:, 1], 0.0))
    print(f"  [seg_scale] all-zero + command=left -> final_y="
          f"{m0[-1, 1]:+.2f} (want 0.00, greedy) ok={greedy_ok}")
    ok &= greedy_ok

    # the actual experiment shape (0, 1, 1): first segment abstains, later ones
    # vote. Plumbing only -- the outcome is what the ablation measures.
    cfg1 = DriveGoTConfig(verbose=False, seg_weight_scale=(0.0, 1.0, 1.0))
    m1, _ = DriveGoTPipeline(cfg1, _make_dummy_generate_fn(),
                             initial_image=None).plan("left")
    mix_ok = m1 is not None and m1.shape == (cfg1.time_horizon, 2)
    print(f"  [seg_scale] (0,1,1) + command=left -> shape_ok={mix_ok} "
          f"seg1_y={np.round(m1[:2, 1], 2).tolist() if mix_ok else None} "
          f"final_y={m1[-1, 1]:+.2f}" if mix_ok else "  [seg_scale] (0,1,1) FAILED")
    ok &= mix_ok
    return ok


def _run_fusion():
    """Fusion arms: the no-op control, both scopes, and the call-count claim."""
    ok = True
    print("\n########## fusion (combine instead of select) ##########")

    def plan_with(**kw):
        """One plan() with a fresh dummy generator, so call counts are per-run."""
        gen = _make_dummy_generate_fn()
        calls = {"n": 0}

        def counting(image, prefix, n, temp, do_sample):
            calls["n"] += 1
            return gen(image, prefix, n, temp, do_sample)

        cfg = DriveGoTConfig(verbose=False, **kw)
        pipe = DriveGoTPipeline(cfg, counting, initial_image=None)
        traj, _ = pipe.plan("straight")
        return traj, calls["n"], pipe

    # 1) the no-op control: fusing ONE candidate must BE selecting it. If this
    #    ever drifts, every fusion number is uninterpretable, because the arm
    #    would differ from the baseline for reasons other than fusion.
    base, base_calls, _ = plan_with()
    solo, solo_calls, _ = plan_with(fuse="median", fuse_top_m=1, fuse_scope="final")
    same = base is not None and solo is not None and np.allclose(base, solo)
    print(f"  [control] fuse_top_m=1 reproduces plain selection: {same} "
          f"(calls {base_calls} vs {solo_calls})")
    ok &= same and base_calls == solo_calls

    # 2) fusing more than one must actually change the output, and stay inside
    #    the pool's envelope (fusion interpolates, it cannot invent)
    fused, _, pipe = plan_with(fuse="median", fuse_scope="final")
    trajs, sel = pipe.final_candidates()
    moved = fused is not None and not np.allclose(fused, base)
    inside = True
    if trajs:
        lo = np.min(np.stack(trajs), 0)
        hi = np.max(np.stack(trajs), 0)
        inside = bool(np.all(fused >= lo - 1e-9) and np.all(fused <= hi + 1e-9))
    # nothing was selected, so the eval must see selected_idx=None and skip
    # selection_gap / rank / top1 rather than attribute them to a pool member
    print(f"  [final] output differs from selection={moved}  "
          f"inside pool envelope={inside}  selected_idx={sel} (want None)  "
          f"fused_n={pipe.last_fusion_n}")
    ok &= moved and inside and sel is None

    # 3) per-segment fusion: every waypoint gets averaged, the beam is gone, and
    #    the call count drops to n_segments * k (12 vs 20 for the defaults) --
    #    the arm is CHEAPER than the one it is compared against
    seg_t, seg_calls, seg_pipe = plan_with(fuse="median", fuse_scope="segment")
    want = DriveGoTConfig().n_segments * DriveGoTConfig().k_candidates
    shape_ok = seg_t is not None and seg_t.shape == (DriveGoTConfig().time_horizon, 2)
    n_fused = len(seg_pipe.last_fusion_n)
    print(f"  [segment] calls={seg_calls} (want {want}, selecting arm uses "
          f"{base_calls})  shape_ok={shape_ok}  fused_segments={n_fused} "
          f"(want {DriveGoTConfig().n_segments})")
    ok &= seg_calls == want and shape_ok and n_fused == DriveGoTConfig().n_segments
    ok &= seg_calls < base_calls

    # 4) beam_width is forced to 1 for per-segment fusion: averaging
    #    segment_local across different parents would mix ego frames in Mode B
    cfg = DriveGoTConfig(fuse="median", fuse_scope="segment", beam_width=3)
    print(f"  [guard] beam_width 3 -> {cfg.beam_width} for per-segment fusion "
          f"(want 1)")
    ok &= cfg.beam_width == 1

    # 5) a bad mode must raise at construction, not silently fall back to argmax
    try:
        DriveGoTConfig(fuse="geometric")
    except AssertionError:
        print("  [guard] unknown fuse mode rejected at config time")
    else:
        print("  [guard] FAIL: unknown fuse mode accepted")
        ok = False

    # 6) fusion off = byte-identical to before this feature existed
    again, again_calls, _ = plan_with(fuse=None)
    print(f"  [regression] fuse=None unchanged: "
          f"{np.allclose(base, again) and base_calls == again_calls}")
    ok &= bool(np.allclose(base, again)) and base_calls == again_calls
    return ok


def _selftest():
    ok = True
    for use_ctx in (False, True):
        tag = "Mode B (WM ctx-update)" if use_ctx else "Mode A (prefix)"
        print(f"\n########## {tag} ##########")
        for command, check, desc in [
            ("right",    lambda t: t[-1, 1] < -0.5, "final y < 0 (right)"),
            ("left",     lambda t: t[-1, 1] > +0.5, "final y > 0 (left)"),
            ("straight", lambda t: abs(t[-1, 1]) < 0.5, "final y ~ 0"),
        ]:
            print(f"\n===== {tag} | command = {command} =====")
            merged, shape_ok, x_mono = _run(command, use_ctx)
            cmd_ok = shape_ok and check(merged)
            print(f"  shape_ok={shape_ok} x_mono={x_mono} {desc}={cmd_ok} "
                  f"(final_y={merged[-1,1]:+.2f})" if shape_ok else "  (no path)")
            ok &= shape_ok and x_mono and cmd_ok

    print("\n########## WM score (two-stage rerank plumbing) ##########")
    for use_ctx in (False, True):
        tag = "Mode B" if use_ctx else "Mode A"
        for command in ("straight", "right"):
            shape_ok, x_mono, wm_populated = _run_wm(command, use_ctx=use_ctx)
            print(f"  [wm {tag} {command}] shape_ok={shape_ok} x_mono={x_mono} "
                  f"wm_populated={wm_populated}")
            ok &= shape_ok and x_mono and wm_populated

    print("\n########## final_candidates (oracle/selection plumbing) ##########")
    ok &= _run_pool()

    print("\n########## seg_weight_scale (per-segment score vote) ##########")
    ok &= _run_seg_scale()

    ok &= _run_fusion()

    print("\nSELF-TEST:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if _selftest() else 1)
