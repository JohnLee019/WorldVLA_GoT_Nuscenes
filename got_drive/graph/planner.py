"""Adapter: drive the graph through DriveGoTPipeline's interface.

WHY AN ADAPTER RATHER THAN A SECOND EVAL PATH
    eval_got_nuscenes.py reads four things off the planner -- `plan()`,
    `final_candidates()`, `final_candidate_scores()` and `last_fusion_n` -- and
    everything downstream (the csv columns, the oracle/selection-gap metrics, the
    collision parity, the bootstrap) is built on their exact shapes. Forking that
    loop would give the graph arm its own copy of ~200 lines of metric plumbing,
    and the first time the two copies disagreed the difference would be reported
    as a result. So the graph wears the incumbent's interface instead, and the
    eval loop only chooses a class.

    Same reasoning as `_CountingFn` wrapping the generator rather than the call
    sites counting for themselves.

WHAT IS DELIBERATELY NOT SUPPORTED
    fusion, the world-model rerank, and the final re-rank (`--final_weights`,
    `--w_likelihood`) raise instead of being silently ignored. Each of them is a
    published arm in sec.1, and an arm that quietly runs without its defining
    feature is the failure mode sec.9 keeps a whole row for ("the runner only
    overwrites the arm it runs, so an old run stays in the table").
"""

from typing import List, Optional, Tuple

import numpy as np

from got_drive.got_pipeline_drive import DriveGoTConfig
from got_drive.graph.controller import Controller
from got_drive.graph.graph_of_operations import (
    build_aggregate_goo, build_improve_goo, build_intent_goo, build_staged_goo)
from got_drive.graph.operations import (
    AggregateIntent, DrivingContext, KeepValid, RealizeIntent, ValidateAndImprove)
from got_drive.graph.thought import Thought


# ─────────────────────────────────────────────────────────────────────────────
# The --planner -> graph_kind map, and the kwargs each arm needs.
#
# ★WHY THIS LIVES HERE RATHER THAN INLINE IN eval_got_nuscenes.py. It used to be
# an `if args.planner in ("graph", "intent"):` block that wrapped ALL the branches
# below it, so `--planner aggregate` and `--planner improve` fell through with an
# EMPTY kwarg dict. GraphPlanner then defaulted graph_kind to "staged" and both
# arms silently ran the incumbent control arm while summary.json recorded
# "planner": "aggregate" (measured: the aggregate arm made 20 calls -- the
# incumbent's k*(1+2*beam) -- where the real arm makes 10, and improve 11).
# --graph_keep_valid, --aggregate_keep_inputs, --improve_tries,
# --improve_no_aggregate, --intent_variants and --intent_anchor_steps were
# swallowed with it. That is sec.9's failure mode in its purest form: a plausible
# table row from an arm that never ran.
#
# Inlined in the eval the mapping was untestable -- eval_got_nuscenes.py imports
# torch, so the selftest cannot import it. As a pure function it is testable, and
# got_drive/graph/selftest.py now asserts the produced kwargs for EVERY --planner
# value land on the intended graph_kind.
# ─────────────────────────────────────────────────────────────────────────────

#: --planner values that are served by GraphPlanner rather than DriveGoTPipeline.
GRAPH_PLANNERS: Tuple[str, ...] = ("graph", "intent", "aggregate", "improve")

#: The one true --planner -> graph_kind map. "graph" is the odd one out: it names
#: the CONTROL arm, whose graph kind is "staged" (build_staged_goo).
PLANNER_GRAPH_KIND = {
    "graph": "staged",
    "intent": "intent",
    "aggregate": "aggregate",
    "improve": "improve",
}

#: Arms built on intent nodes -- everything except the staged control.
INTENT_PLANNERS: Tuple[str, ...] = ("intent", "aggregate", "improve")

#: Arms that fan out into AggregateIntent operations (improve does, unless
#: --improve_no_aggregate). Used for both the kwargs and the csv columns.
AGGREGATE_PLANNERS: Tuple[str, ...] = ("aggregate", "improve")


def planner_kwargs(args, intents=None) -> dict:
    """The GraphPlanner kwargs for `args.planner`, or {} for the incumbent.

    `args` is eval_got_nuscenes.py's argparse Namespace; `intents` is the intent
    set it built once per run. Returns {} for --planner pipeline so the eval can
    splat the result into DriveGoTPipeline unchanged.

    Every flag an arm owns is set HERE, so "the flag was parsed but never reached
    the planner" cannot happen once per arm in a growing if-chain.
    """
    if args.planner not in GRAPH_PLANNERS:
        return {}
    kw = {
        "keep_valid": bool(args.graph_keep_valid),
        # Required, never defaulted -- see GraphPlanner.__init__.
        "graph_kind": PLANNER_GRAPH_KIND[args.planner],
    }
    if args.planner in AGGREGATE_PLANNERS:
        kw["aggregate_method"] = args.aggregate_method
        kw["aggregate_keep_inputs"] = bool(args.aggregate_keep_inputs)
    if args.planner == "improve":
        kw["num_tries"] = args.improve_tries
        kw["improve_aggregate"] = not args.improve_no_aggregate
    if args.planner in INTENT_PLANNERS:
        kw["intents"] = intents
        kw["variants"] = args.intent_variants
        kw["anchor_steps"] = args.intent_anchor_steps
    return kw


class GraphPlanner:
    """DriveGoTPipeline-shaped front end over GraphOfOperations + Controller.

    A fresh GraphOfOperations is built per plan() call: operations carry their
    executed flag and their thoughts, so reusing one would leak the previous
    record's pool into this record's oracle metrics -- the same stale-state
    failure DriveGoTPipeline.plan() resets against on entry.

    ★`graph_kind` IS REQUIRED AND KEYWORD-ONLY, ON PURPOSE. It used to default to
    "staged". That default is what turned eval_got_nuscenes.py's wrong `if
    args.planner in ("graph", "intent")` guard from a TypeError on record 1 into a
    plausible table row: --planner aggregate and --planner improve constructed this
    class with no kwargs at all and quietly ran the incumbent control arm. A
    default that silently names the CONTROL arm is the worst possible default --
    the control is the one arm whose numbers already exist to be mistaken for.
    With no default the same mistake cannot survive construction. Callers that
    really want the control say `graph_kind="staged"`, which also reads correctly
    at the call site.
    """

    def __init__(self, cfg: DriveGoTConfig, generate_fn, *, graph_kind: str,
                 initial_image=None, context_update_fn=None,
                 score_fn=None, wm_score_fn=None, lik_score_fn=None,
                 keep_valid: bool = False,
                 intents=None, variants: int = 2, anchor_steps: int = 1,
                 aggregate_method: str = "median", aggregate_keep_inputs: bool = False,
                 num_tries: int = 2, improve_aggregate: bool = True):
        if cfg.fuse is not None:
            raise NotImplementedError(
                "fusion is not ported to the graph planner. Aggregate is the "
                "operation it corresponds to and it is not implemented yet "
                "(handoff sec.1.7(a2): upstream's Aggregate is always LLM-driven, "
                "and sec.1.5 tested only a non-LLM analogue). Run --planner "
                "pipeline for fusion arms.")
        if wm_score_fn is not None:
            raise NotImplementedError(
                "the world-model rerank branch of _score_pool is not ported. It "
                "is off by default (sec.8 keeps the WM an offline evaluator); "
                "running without it would silently be a different arm.")
        if lik_score_fn is not None or cfg.final_weights is not None:
            raise NotImplementedError(
                "the final re-rank (--final_weights / --w_likelihood) is not "
                "ported. Both change WHICH candidate is returned, so a graph run "
                "that ignored them would not be the arm its flags name.")
        if context_update_fn is not None:
            raise NotImplementedError(
                "Mode B (context update) is not ported -- it is dormant by sec.8 "
                "decision and its per-node frames need the rotation path that "
                "nothing in the graph selftest exercises.")

        if graph_kind not in ("staged", "intent", "aggregate", "improve"):
            raise ValueError("graph_kind must be one of staged/intent/aggregate/improve, "
                             f"got {graph_kind!r}")

        # Same guard RealizeIntent enforces at execute time, hoisted to
        # construction: an arm misconfigured this way returns None on EVERY record
        # (the model is left 0 waypoints to generate), and a csv of nothing but
        # `malformed_plan` with no stated cause costs a whole run to diagnose.
        # Raise before record 1, not after record N.
        if graph_kind in ("intent", "aggregate", "improve") and int(anchor_steps) >= cfg.time_horizon:
            raise ValueError(
                f"anchor_steps={anchor_steps} >= cfg.time_horizon={cfg.time_horizon}: "
                f"the intent would commit the entire horizon and every record would "
                f"come back malformed. Use anchor_steps <= {cfg.time_horizon - 1}.")

        self.cfg = cfg
        self.generate_fn = generate_fn
        self.initial_image = initial_image
        self.score_fn = score_fn
        self.keep_valid = bool(keep_valid)
        self.graph_kind = graph_kind
        self.intents = intents
        self.variants = int(variants)
        self.anchor_steps = int(anchor_steps)
        self.aggregate_method = aggregate_method
        self.aggregate_keep_inputs = bool(aggregate_keep_inputs)
        self.num_tries = int(num_tries)
        self.improve_aggregate = bool(improve_aggregate)
        # interface parity with DriveGoTPipeline
        self.last_fusion_n: List[int] = []
        self.last_final_pool: List[Thought] = []
        self.last_selected: Optional[Thought] = None
        self.last_controller: Optional[Controller] = None
        # intent arm only: did the intents actually produce distinct plans on this
        # record? See intent.separation_report -- read this BEFORE the L2.
        self.last_separation: dict = {}
        # aggregate arm only: did anything actually get combined?
        self.last_aggregate: dict = {}
        # any arm run with keep_valid=True: how many candidates the filter removed
        # BEFORE KeepBestN, i.e. how much smaller the pool minADE_C and the
        # selection gap are computed over has become. Zero when keep_valid is off.
        # Logged per record so a shrinking denominator can never be read as a
        # better generator -- see KeepBestN's docstring.
        self.last_keep_valid: dict = {}
        # improve arm only: how many candidates failed validation, how many were
        # actually repaired, and how far each repair moved the prefix.
        self.last_improve: dict = {}

    def _build(self):
        if self.graph_kind == "improve":
            return build_improve_goo(self.cfg, intents=self.intents,
                                     variants=self.variants,
                                     num_tries=self.num_tries,
                                     method=self.aggregate_method,
                                     aggregate=self.improve_aggregate,
                                     keep_inputs=self.aggregate_keep_inputs,
                                     keep_valid=self.keep_valid,
                                     anchor_steps=self.anchor_steps)
        if self.graph_kind == "aggregate":
            return build_aggregate_goo(self.cfg, intents=self.intents,
                                       variants=self.variants,
                                       method=self.aggregate_method,
                                       keep_inputs=self.aggregate_keep_inputs,
                                       keep_valid=self.keep_valid,
                                       anchor_steps=self.anchor_steps)
        if self.graph_kind == "intent":
            return build_intent_goo(self.cfg, intents=self.intents,
                                    variants=self.variants,
                                    keep_valid=self.keep_valid,
                                    anchor_steps=self.anchor_steps)
        return build_staged_goo(self.cfg, keep_valid=self.keep_valid)

    def plan(self, command: str) -> Tuple[Optional[np.ndarray], Optional[Thought]]:
        self.last_final_pool = []
        self.last_selected = None
        self.last_separation = {}
        self.last_aggregate = {}
        self.last_keep_valid = {}
        self.last_improve = {}
        goo = self._build()
        ctx = DrivingContext(self.generate_fn, self.cfg, self.initial_image,
                             command, score_fn=self.score_fn)
        ctrl = Controller(goo, ctx).run()
        self.last_controller = ctrl
        for op in goo.operations:
            if isinstance(op, RealizeIntent):
                self.last_separation = dict(op.separation)
        aggs = [op for op in goo.operations if isinstance(op, AggregateIntent)]
        if aggs:
            # Per record, like separation: an aggregate built from one surviving
            # realisation is an identity (fuse_trajectories returns it unchanged),
            # so an arm where that happens everywhere combined nothing.
            self.last_aggregate = {
                "n_aggregates": len(aggs),
                "n_inputs": [a.n_inputs for a in aggs],
                "n_infeasible_dropped": sum(a.n_infeasible_dropped for a in aggs),
                "n_identity": sum(1 for a in aggs if a.n_inputs <= 1),
                "method": self.aggregate_method,
            }
        for op in goo.operations:
            if isinstance(op, ValidateAndImprove):
                self.last_improve = dict(op.stats)
        # ★The pool minADE_C is computed over shrinks when KeepValid filters, and
        # that shrinkage happens BEFORE KeepBestN -- so final_pool() is the
        # feasible subset and a lower minADE_C would read as a better generator
        # when it is only a smaller denominator. Report it per record; a silent
        # denominator change is exactly the kind of artefact sec.9 keeps a row for.
        kvs = [op for op in goo.operations if isinstance(op, KeepValid)]
        if kvs:
            self.last_keep_valid = {
                "enabled": self.keep_valid,
                "n_ops": len(kvs),
                "n_dropped": sum(op.n_dropped for op in kvs),
                "per_op_dropped": [op.n_dropped for op in kvs],
            }
        merged = ctrl.plan()
        pool = ctrl.final_pool()
        self.last_final_pool = pool
        # Emit takes the first survivor, and KeepBestN sorted the pool, so the
        # selected thought is pool[0] whenever anything was produced. Resolved by
        # identity rather than assumed, so it stays correct if Emit ever changes.
        if merged is not None:
            for t in pool:
                if t.cum_traj is merged:
                    self.last_selected = t
                    break
            if self.last_selected is None and pool:
                self.last_selected = pool[0]
        return merged, self.last_selected

    # ── the diagnostics eval_got_nuscenes.py reads ───────────────────────────
    def _final_pool_thoughts(self) -> List[Thought]:
        T = self.cfg.time_horizon
        return [t for t in self.last_final_pool if t.cum_traj.shape[0] == T]

    def final_candidates(self) -> Tuple[List[np.ndarray], Optional[int]]:
        trajs: List[np.ndarray] = []
        sel: Optional[int] = None
        for t in self._final_pool_thoughts():
            if t is self.last_selected:
                sel = len(trajs)
            trajs.append(t.cum_traj)
        return trajs, sel

    def final_candidate_scores(self) -> dict:
        """Aligned with final_candidates()[0]. Same keys as DriveGoTPipeline.

        `wm`, `likelihood` and `final_score` are nan throughout: the operations
        that would populate them are not ported (see __init__'s guards). They are
        present so the csv schema does not change between planners -- a missing
        column would look like a different eval, and sec.9 has a row about arms
        drifting apart inside one table.
        """
        ts = self._final_pool_thoughts()
        comp = [t.state.get("components", {}) for t in ts]
        return {
            "kinematic": [float(c.get("kinematic", np.nan)) for c in comp],
            "command": [float(c.get("command", np.nan)) for c in comp],
            "wm": [float("nan")] * len(ts),
            "likelihood": [float("nan")] * len(ts),
            "final_score": [float("nan")] * len(ts),
            "segment_score": [float(t.state.get("segment_score", np.nan)) for t in ts],
            "path_score": [t.path_score for t in ts],
        }
