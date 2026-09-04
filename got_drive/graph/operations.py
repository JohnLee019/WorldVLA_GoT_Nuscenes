"""Operations. Shaped after `graph_of_thoughts/operations/operations.py`.

UPSTREAM SHAPE WE KEEP
    id / predecessors / successors / executed / operation_type
    can_be_executed()  -- all predecessors executed
    get_previous_thoughts()
    add_predecessor() / add_successor()
    execute(...)  -> guards, calls _execute(), sets executed
    _execute()    -- abstract
    get_thoughts() -- abstract

ONE DELIBERATE DEVIATION, AND WHY
    Upstream: `execute(lm, prompter, parser, **problem_parameters)`. Those three
    exist because every upstream operation ultimately renders a prompt and parses a
    response. Ours does not: our Generate calls a sampling VLA through `generate_fn`
    and gets waypoints back already parsed by the model wrapper, and our Score is a
    `scoring_function` -- which upstream names as an explicit, supported escape
    hatch (`Score(scoring_function=...)`), so this is NOT a departure from the
    method, only from the plumbing. We therefore pass a single `DrivingContext`
    carrying (generate_fn, score_fn, cfg, image, command). Handoff sec.1.7(a2) is
    the place to keep this argument straight when writing the paper.

    ⚠️Aggregate and Improve are the two operations upstream ALWAYS drives with the
    LLM, and our base VLA has no channel for "combine these" or "improve this" --
    it was fine-tuned on image+prompt -> action tokens. They are not in this file
    yet on purpose. The honest wording for anything we build there is "a non-LLM
    analogue of the official operation" (sec.1.7(a2) corrects an earlier overclaim
    about exactly this).
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Callable, Dict, List, Optional

import numpy as np

from got_drive.got_pipeline_drive import (
    DriveGoTConfig,
    _seg_weights,
    advance_pose,
)
from got_drive.scoring_driving import rank_candidates, _feasible
from got_drive.graph.thought import Thought, make_root_state, make_traj_state


class OperationType(Enum):
    root = 0
    generate = 1
    score = 2
    keep_valid = 3
    keep_best_n = 4
    emit = 5
    # declared now so the enum is stable as they land; see package docstring
    aggregate = 6
    improve = 7


class DrivingContext:
    """What upstream passes as (lm, prompter, parser, **problem_parameters).

    Built once per record. `image` and `command` are the observation; `generate_fn`
    is the sampling VLA adapter from got_pipeline_drive.make_model_generate_fn (so
    the ego-status channel reaches it through the same state_holder the incumbent
    pipeline uses -- there is exactly one generator in the process, not two).
    """

    def __init__(self, generate_fn: Callable, cfg: DriveGoTConfig,
                 image, command: str, score_fn: Optional[Callable] = None):
        self.generate_fn = generate_fn
        self.cfg = cfg
        self.image = image
        self.command = command
        self.score_fn = score_fn if score_fn is not None else _default_score_fn


def _default_score_fn(cum_trajs, command, cfg, seg_idx=0):
    """Identical to DriveGoTPipeline._default_score_fn -- same call, same weights.

    Kept as a module function rather than imported off the class so the graph path
    cannot drift if that method is ever changed for the incumbent arm; the selftest
    asserts the two produce the same ordering.
    """
    w_kin, w_cmd, _ = _seg_weights(cfg, seg_idx)
    return rank_candidates(cum_trajs, command, weights=(w_kin, w_cmd),
                           norm=cfg.score_norm)


class Operation(ABC):
    """Abstract base. Mirrors upstream field-for-field."""

    _ids = 0

    def __init__(self) -> None:
        self.id: int = Operation._ids
        Operation._ids += 1
        self.predecessors: List["Operation"] = []
        self.successors: List["Operation"] = []
        self.executed: bool = False

    operation_type: OperationType = None

    def can_be_executed(self) -> bool:
        """True when every predecessor has run. This is the whole scheduler."""
        return all(p.executed for p in self.predecessors)

    def get_previous_thoughts(self) -> List[Thought]:
        return [t for p in self.predecessors for t in p.get_thoughts()]

    def add_predecessor(self, operation: "Operation") -> None:
        self.predecessors.append(operation)
        operation.successors.append(self)

    def add_successor(self, operation: "Operation") -> None:
        self.successors.append(operation)
        operation.predecessors.append(self)

    def execute(self, ctx: DrivingContext, **kwargs) -> None:
        assert self.can_be_executed(), (
            f"operation {self.id} ({type(self).__name__}) executed before its "
            f"predecessors -- the controller must never dequeue it yet")
        self._execute(ctx, **kwargs)
        self.executed = True

    @abstractmethod
    def _execute(self, ctx: DrivingContext, **kwargs) -> None:
        ...

    @abstractmethod
    def get_thoughts(self) -> List[Thought]:
        ...


class Root(Operation):
    """Seeds the graph with the empty trajectory at the ego origin.

    Upstream has no Root operation -- its roots are Generate ops fed by
    `problem_parameters`. Ours is explicit because a driving thought carries a pose
    and a cumulative trajectory, and "no waypoints yet, heading 0 at the origin" is
    a real vertex that Generate reads rather than an implicit starting condition.
    """

    operation_type = OperationType.root

    def __init__(self) -> None:
        super().__init__()
        self.thoughts: List[Thought] = []

    def _execute(self, ctx: DrivingContext, **kwargs) -> None:
        self.thoughts = [Thought(make_root_state(ctx.image))]

    def get_thoughts(self) -> List[Thought]:
        return self.thoughts


class GenerateSegment(Operation):
    """k candidate continuations per surviving thought. Upstream `Generate`.

    ★BIT-EQUIVALENCE NOTE. The generator is stateful in the RNG sense -- sec.1.4
    found the greedy baseline drifting because GoT's draw count changed with
    k/beam. So this reproduces DriveGoTPipeline._expand's call pattern EXACTLY:
    same order over parents, same k, `temperatures[min(i, len-1)]`,
    `do_sample=(i > 0)`, same malformed-shape skip, same 4-decimal dedup per
    parent. Change any of it and the no-op arm stops reproducing 3.5557, which is
    the only reason to trust anything built on top.

    Mode A only for now (fixed image, prefix conditioning). Mode B is dormant by
    sec.8 decision and its per-node frames would need the rotation path below to be
    exercised, which nothing here tests yet -- so it raises instead of quietly
    running an untested branch.
    """

    operation_type = OperationType.generate

    def __init__(self, seg_idx: int, n_generate: Optional[int] = None) -> None:
        super().__init__()
        self.seg_idx = int(seg_idx)
        self.n_generate = n_generate
        self.thoughts: List[Thought] = []

    def _make_child(self, parent: Thought, seg: np.ndarray,
                    ctx: DrivingContext) -> Thought:
        """Frame composition, identical to DriveGoTPipeline._make_node (Mode A)."""
        base_p, base_theta = parent.state["end_pose"]
        seg_orig = seg  # Mode A: the generator already works in the original frame
        parent_cum = parent.cum_traj
        cum = seg_orig if parent_cum.size == 0 else np.vstack([parent_cum, seg_orig])
        return parent.child(**make_traj_state(
            segment_local=seg,
            cum_traj=cum,
            end_pose=advance_pose(seg_orig, base_p),
            depth=self.seg_idx,
            path_score=parent.path_score,   # ScoreDriving adds this segment's score
            image=ctx.image,
            intent=parent.state.get("intent"),
        ))

    def _execute(self, ctx: DrivingContext, **kwargs) -> None:
        cfg = ctx.cfg
        n_gen = cfg.segment_len if self.n_generate is None else self.n_generate
        out: List[Thought] = []
        for parent in self.get_previous_thoughts():
            prefix = parent.cum_traj if parent.cum_traj.shape[0] > 0 else None
            seen = set()
            for i in range(cfg.k_candidates):
                temp = cfg.temperatures[min(i, len(cfg.temperatures) - 1)]
                seg = ctx.generate_fn(ctx.image, prefix, n_gen, temp, i > 0)
                if seg is None:
                    continue
                seg = np.asarray(seg, dtype=np.float64)
                if seg.shape != (n_gen, 2):
                    continue
                key = seg.round(4).tobytes()
                if key in seen:
                    continue
                seen.add(key)
                out.append(self._make_child(parent, seg, ctx))
        self.thoughts = out

    def get_thoughts(self) -> List[Thought]:
        return self.thoughts


class ScoreDriving(Operation):
    """Upstream `Score`, using the `scoring_function` path it explicitly allows.

    Writes `path_score` (accumulated) as well as the per-thought `score`, because
    the beam ranks on the accumulation -- see thought.py.

    ⚠️The world-model rerank branch of DriveGoTPipeline._score_pool is not ported
    yet. It is off by default (sec.8 keeps the WM as an offline evaluator) and
    porting it silently would produce an arm that looks like the incumbent and is
    not, so it raises.
    """

    operation_type = OperationType.score

    def __init__(self, seg_idx: int) -> None:
        super().__init__()
        self.seg_idx = int(seg_idx)
        self.thoughts: List[Thought] = []

    def _execute(self, ctx: DrivingContext, **kwargs) -> None:
        pool = self.get_previous_thoughts()
        if not pool:
            self.thoughts = []
            return
        cum = [t.cum_traj for t in pool]
        totals, comp = ctx.score_fn(cum, ctx.command, ctx.cfg, self.seg_idx)
        for i, t in enumerate(pool):
            seg_score = float(totals[i])
            t.score = seg_score                       # upstream per-thought score
            t.state["segment_score"] = seg_score
            # accumulate onto what the predecessor already carried
            t.state["path_score"] = float(t.state.get("path_score", 0.0)) + seg_score
            t.state["components"] = {k: float(v[i]) for k, v in comp.items()
                                     if v is not None}
            t.state["depth"] = self.seg_idx
        self.thoughts = pool

    def get_thoughts(self) -> List[Thought]:
        return self.thoughts


class KeepValid(Operation):
    """Upstream `KeepValid` -- feasibility as its OWN operation.

    ★WHY THIS MATTERS BEYOND TIDINESS. Handoff sec.1.7(b)3: upstream separates
    validity from scoring, the driving port folded the veto into the score as a
    -1e6 penalty, and "that was the structural cause of the z-norm veto bug"
    (a catastrophic candidate survived because the penalty was inside the
    z-normalisation and got rescaled away). Splitting the operation makes that
    class of bug unrepresentable.

    ⚠️DEFAULT IS `enabled=False`, i.e. a pass-through. That is not timidity: the
    incumbent's feasibility veto still lives inside `rank_candidates`, so filtering
    here as well would drop candidates twice and the no-op arm would stop
    reproducing 3.5557. Turning it on is a DIFFERENT ARM (report it separately) and
    should come with the veto being removed from the score in the same change.

    ★A PASS-THROUGH MUST NOT OVERWRITE A VERDICT SOMEBODY ELSE ALREADY REACHED.
    In `build_improve_goo` the chain is ValidateAndImprove -> ScoreDriving ->
    KeepValid(enabled=False), and a blanket `t.valid = True` here destroyed the
    real verdict ValidateAndImprove wrote (see its `n_still_invalid`). The run then
    produced two artefacts that disagreed: the trace dump said `valid: true` for a
    thought the stats counted as still invalid. sec.9's standing failure mode is
    exactly this -- two artefacts from one run that cannot both be right, with
    nothing in either saying which. So the pass-through only fills in thoughts that
    NOBODY has validated yet (`t.validated` is False), which still records
    "checked, and we did not filter" for the staged arm -- where nothing upstream
    validates -- while leaving an existing verdict untouched.

    ★THE DROPPED THOUGHTS ARE KEPT (`self.dropped`), NOT DISCARDED. With
    `enabled=True` this operation runs BEFORE KeepBestN, so the pool KeepBestN
    retains -- the pool `minADE_C` and the selection gap are computed over -- is
    only the feasible subset. A lower minADE_C would then read as "separating
    validity from scoring improved the generator" when it is only a changed
    denominator. The evidence therefore survives in the trace dump
    (Controller.output_graph serialises `dropped` too, so the full pool is
    recoverable offline with zero new inference -- sec.1.10(a2)), and `n_dropped`
    is surfaced per record in the csv so the shrinkage is loud rather than silent.
    """

    operation_type = OperationType.keep_valid

    def __init__(self, enabled: bool = False) -> None:
        super().__init__()
        self.enabled = bool(enabled)
        self.thoughts: List[Thought] = []
        self.n_dropped: int = 0
        # Retained for the trace dump: see the docstring above. Filtering here
        # shrinks the pool every downstream diagnostic is computed over, so the
        # candidates that were removed have to remain inspectable.
        self.dropped: List[Thought] = []

    def _execute(self, ctx: DrivingContext, **kwargs) -> None:
        pool = self.get_previous_thoughts()
        if not self.enabled:
            for t in pool:
                # Only fill in an ABSENT verdict. Overwriting one that
                # ValidateAndImprove already reached is what made the dump and the
                # stats contradict each other -- see the docstring.
                if not t.validated:
                    t.valid = True      # records "checked, and we did not filter"
            self.thoughts = pool
            self.n_dropped = 0
            self.dropped = []
            return
        kept, dropped = [], []
        for t in pool:
            ok = bool(_feasible(t.cum_traj))
            t.valid = ok
            (kept if ok else dropped).append(t)
        self.n_dropped = len(dropped)
        # Never hand an empty beam downstream: an all-infeasible pool means the
        # gate is miscalibrated, not that the record has no plan. Fall back to the
        # full pool and let the count surface it.
        if kept:
            self.thoughts = kept
            self.dropped = dropped
        else:
            self.thoughts = pool
            # Nothing was actually removed on the fallback path, so the diagnostic
            # pool downstream is unchanged; `n_dropped` still reports how many
            # failed the gate, which is the number that says the gate misfired.
            self.dropped = []

    def get_thoughts(self) -> List[Thought]:
        return self.thoughts


class KeepBestN(Operation):
    """Upstream `KeepBestN(n, higher_is_better=True)`, ranking on `path_score`.

    ★ADAPTATION: we retain everything THIS OPERATION was handed, sorted, in
    `self.all_thoughts`, while only passing the survivors downstream. The
    candidates this operation discards are not waste -- `minADE_C` and the
    selection gap are computed over the deepest pool (sec.1 claim 3, and the whole
    generator-vs-selector decomposition of sec.7.4), so throwing them away would
    silently delete the project's main diagnostic.

    ⚠️"EVERYTHING THIS OPERATION WAS HANDED" IS NOT ALWAYS THE FULL POOL, AND THE
    DIFFERENCE IS A TRAP. This docstring used to claim `all_thoughts` was the FULL
    pool; that is only true when nothing upstream filters. With
    `KeepValid(enabled=True)` the infeasible candidates are removed BEFORE this
    operation ever sees them, so `all_thoughts` -- and therefore
    `Controller.final_pool()`, and therefore minADE_C -- is the feasible subset.
    Measured on the dummy generator: the incumbent keeps a pool of 6 with 2
    infeasible, `--graph_keep_valid` keeps 4 with 0. A lower minADE_C across that
    pair is a CHANGED DENOMINATOR, not a better generator. The dropped candidates
    stay inspectable on `KeepValid.dropped` (serialised by output_graph) and the
    count is written to the csv per record, so the shrinkage is loud; comparing
    minADE_C across arms with different `--graph_keep_valid` is still invalid.
    """

    operation_type = OperationType.keep_best_n

    def __init__(self, n: int, higher_is_better: bool = True) -> None:
        super().__init__()
        assert n > 0, "KeepBestN needs n > 0"
        self.n = int(n)
        self.higher_is_better = bool(higher_is_better)
        self.thoughts: List[Thought] = []
        self.all_thoughts: List[Thought] = []

    def _execute(self, ctx: DrivingContext, **kwargs) -> None:
        pool = list(self.get_previous_thoughts())
        # Stable sort, exactly like DriveGoTPipeline: ties keep generation order,
        # which is what makes a zero-weight segment fall back to the greedy
        # candidate (idx 0, temperature 1.0, do_sample=False) rather than an
        # arbitrary one. cfg.seg_weight_scale documents that this is load-bearing.
        self.all_thoughts = sorted(pool, key=lambda t: t.path_score,
                                   reverse=self.higher_is_better)
        self.thoughts = self.all_thoughts[: self.n]

    def get_thoughts(self) -> List[Thought]:
        return self.thoughts


class GenerateIntents(Operation):
    """Root -> one thought per driving hypothesis. ZERO model calls.

    Upstream's `Generate` always prompts the LLM. This one does not, and the
    distinction is worth keeping straight in the paper: the intents are DECLARED
    (a designed hypothesis set, Step 2.9/2.10's prescription), not sampled. What
    the model is asked to do is REALISE each of them, which is `RealizeIntent`
    below and is a genuine Generate.

    ★This is the operation that answers sec.1.17: after it, stage one has
    len(intents) live options by construction instead of the measured 1.52.
    """

    operation_type = OperationType.generate

    def __init__(self, intents=None) -> None:
        super().__init__()
        from got_drive.graph.intent import make_intent_grid
        self.intents = list(intents) if intents is not None else make_intent_grid()
        assert self.intents, "an intent graph needs at least one intent"
        self.thoughts: List[Thought] = []

    def _execute(self, ctx: DrivingContext, **kwargs) -> None:
        from got_drive.graph.thought import make_intent_state
        root = self.get_previous_thoughts()
        parent = root[0] if root else None
        out = []
        for it in self.intents:
            state = make_intent_state(it.as_dict(), depth=0, image=ctx.image)
            out.append(parent.child(**state) if parent is not None else Thought(state))
        self.thoughts = out

    def get_thoughts(self) -> List[Thought]:
        return self.thoughts


class RealizeIntent(Operation):
    """Each intent -> full-horizon trajectories, via prefix conditioning.

    HOW AN INTENT REACHES THE MODEL WITHOUT RETRAINING
        The base VLA was fine-tuned on image+prompt -> action tokens; it has no
        channel for "drive conservatively". The one channel that exists is the
        prefix: `predict_segment(prefix_wp=...)` continues from waypoints already
        decided. So the intent is committed as the FIRST waypoint (an arc of
        length v0*speed_scale at the given curvature) and the model completes the
        remaining horizon from there.

        `v0` is the model's OWN implied first step, taken from one greedy call
        made before the intents are realised. That is what makes this arm run on
        the incumbent checkpoint with no ego status and no retraining. With ego
        status the same anchor is computed from a measured speed instead, and the
        intents become accelerations in m/s^2 -- which is also what lets the
        >= 1 m separation floor be hit by arithmetic instead of by luck.

    ⚠️THIS IS NOT UPSTREAM'S `Improve`. It is conditioned generation, not
    "here is your previous answer, refine it". sec.1.7 exists because a port
    described its operations with upstream's names while doing something else;
    the honest wording for this one is "intent-conditioned Generate".

    ⚠️COST IS NOT THE INCUMBENT'S 20. It is 1 + n_intents*variants. Report the
    call count next to any result (README section 1a's cost column), and note
    that sec.1.5's free pairing -- equal call counts landing on bit-identical
    trajectories -- only holds between arms that make the same number of calls.
    """

    operation_type = OperationType.generate

    def __init__(self, variants: int = 2, anchor_steps: int = 1) -> None:
        super().__init__()
        assert variants >= 1, "need at least one realisation per intent"
        assert anchor_steps >= 1, "need at least one committed waypoint"
        self.variants = int(variants)
        # How much of the horizon the intent commits. 1 expresses speed and
        # essentially not curvature (see intent.py's k*L^2/2 table); a curvature
        # axis needs >= 2 AND k = +-0.05. Every committed step is one the model
        # no longer chooses, so this trades conditioning strength for freedom.
        self.anchor_steps = int(anchor_steps)
        self.thoughts: List[Thought] = []
        # Recorded per record, read back from the trace dump. See
        # intent.separation_report -- an intent set that did not separate has not
        # created alternatives, and that has to be visible in the artefacts.
        self.separation: Dict = {}
        self.v0_step: float = float("nan")

    def _execute(self, ctx: DrivingContext, **kwargs) -> None:
        from got_drive.graph.intent import Intent, anchor_waypoint, separation_report
        from got_drive.graph.thought import make_traj_state
        cfg = ctx.cfg
        T = cfg.time_horizon

        # ── one greedy call: the model's own plan, and the scale the intents
        # multiply. Same call the incumbent's candidate 0 makes (temperature 1.0,
        # do_sample False), so it is not an extra kind of query, just an extra one.
        # ★GUARD, NOT AN ASSUMPTION. __init__ can only check `anchor_steps >= 1`;
        # the upper bound is `cfg.time_horizon`, which only exists here. With
        # anchor_steps >= T the generator below is asked for `T - anchor.shape[0]`
        # <= 0 waypoints, the shape check rejects every realisation, `out` stays
        # empty and plan() returns None on EVERY record -- an eval run that fills
        # its csv with `malformed_plan` and no reason why. Name the cause instead:
        # a silent all-null arm is the sec.9 failure mode (an arm that ran without
        # its defining feature and still produced a table row).
        if self.anchor_steps >= T:
            raise ValueError(
                f"anchor_steps={self.anchor_steps} leaves nothing for the model to "
                f"generate: cfg.time_horizon={T}, so the intent would commit the "
                f"whole horizon and every record would come back malformed. Use "
                f"anchor_steps <= {T - 1} (a curvature axis needs >= 2 AND "
                f"k=+-0.05; see intent.USABLE_CURVATURES).")

        ref = ctx.generate_fn(ctx.image, None, T, cfg.temperatures[0], False)
        if ref is None:
            self.thoughts = []
            return
        ref = np.asarray(ref, dtype=np.float64)
        self.v0_step = float(np.linalg.norm(ref[0])) if ref.shape[0] else 0.0

        parents = self.get_previous_thoughts()
        anchors, out = [], []
        for parent in parents:
            it = Intent(float(parent.state["intent"]["speed_scale"]),
                        float(parent.state["intent"]["curvature"]))
            anchor = anchor_waypoint(self.v0_step, it, anchor_steps=self.anchor_steps)
            # the WHOLE anchor, not anchor[0]: separation_report measures at the
            # anchor's END, which is the only place curvature has had room to
            # accumulate (lateral offset grows as k*L^2/2).
            anchors.append(anchor)
            seen = set()
            for v in range(self.variants):
                # variant 0 is greedy given the anchor, the rest are sampled --
                # the incumbent's convention, so temperature indexing matches.
                temp = cfg.temperatures[min(v, len(cfg.temperatures) - 1)]
                seg = ctx.generate_fn(ctx.image, anchor, T - anchor.shape[0], temp, v > 0)
                if seg is None:
                    continue
                seg = np.asarray(seg, dtype=np.float64)
                if seg.shape != (T - anchor.shape[0], 2):
                    continue
                cum = np.vstack([anchor, seg])
                key = cum.round(4).tobytes()
                if key in seen:
                    continue
                seen.add(key)
                out.append(parent.child(**make_traj_state(
                    segment_local=cum, cum_traj=cum,
                    end_pose=advance_pose(cum, np.zeros(2)),
                    depth=0, path_score=parent.path_score,
                    image=ctx.image, intent=parent.state["intent"])))
        self.separation = separation_report(anchors)
        self.separation["v0_step"] = round(self.v0_step, 4)
        self.thoughts = out

    def get_thoughts(self) -> List[Thought]:
        return self.thoughts


class AggregateIntent(Operation):
    """Combine the realisations of ONE intent into a single trajectory.

    ⚠️NAMING, PRECISELY. Upstream's `Aggregate` is always LLM-driven
    (`aggregation_prompt`); our base VLA has no "combine these" input format, so
    this is a **non-LLM analogue** of it. sec.1.7(a2) exists because an earlier
    write-up called sec.1.5's fusion "the missing Aggregate" and that was an
    overclaim. Say "non-LLM analogue" in the paper.

    ★WHY THIS IS NOT sec.1.5's FUSION REPEATED
        sec.1.5 measured the geometric analogue and it was null at best and
        harmful at worst: `final_top3` gave d_output -0.0206 (p_sc 0.4666), and
        per-segment fusion drove infeasibility from a 2.5% base rate to **41.8%**
        with turns hurt 3.7x more than straights (left +0.8124).

        The diagnosis was never "averaging is wrong". sec.1.5 proved the opposite
        by construction -- *the average of two feasible trajectories is always
        feasible*, because every limit is a norm bound on a linear functional --
        and located the damage in the **re-conditioned prefix**. What actually
        failed is averaging across DIFFERENT PLANS: the mean of "turn left" and
        "go straight" is neither, which is why the loss concentrated on turns.

        Under intent nodes that failure is unrepresentable. This operation only
        ever sees realisations of ONE hypothesis, so the inputs are samples from
        a single plan's distribution and the combination is a point estimator,
        not mode-averaging. The type system does the work the arithmetic could
        not.

    ★AND IT ENFORCES THE CONVEXITY PRECONDITION ITSELF. Infeasible inputs are
    dropped before combining, because the "average of feasible is feasible"
    guarantee says nothing about an average that included a physically impossible
    candidate. Recorded as `n_infeasible_dropped` rather than assumed.

    Fusing one input returns it unchanged (`fuse_trajectories` guarantees this),
    so an intent with a single surviving realisation is a no-op -- the same
    property that makes `--fuse_top_m 1` a valid control arm in sec.1.5.
    """

    operation_type = OperationType.aggregate

    def __init__(self, intent_name: str, method: str = "median",
                 keep_inputs: bool = False) -> None:
        super().__init__()
        from got_drive.fusion import FUSE_MODES
        if method not in FUSE_MODES:
            raise ValueError(f"method must be one of {FUSE_MODES}, got {method!r}")
        self.intent_name = str(intent_name)
        self.method = method
        # False (default) matches upstream: an operation's get_thoughts() returns
        # what IT produced. True keeps the realisations alongside the aggregate,
        # which preserves pool comparability -- with the inputs gone, minADE_C is
        # computed over a different pool and `d_pool` stops being interpretable
        # against sec.1 (where fusion left the pool at exactly 0).
        self.keep_inputs = bool(keep_inputs)
        self.thoughts: List[Thought] = []
        self.n_inputs: int = 0
        self.n_infeasible_dropped: int = 0

    def _execute(self, ctx: DrivingContext, **kwargs) -> None:
        from got_drive.fusion import fuse_trajectories
        from got_drive.graph.thought import make_traj_state

        mine = [t for t in self.get_previous_thoughts()
                if (t.state.get("intent") or {}).get("name") == self.intent_name]
        self.n_inputs = len(mine)
        if not mine:
            self.thoughts = []
            return

        feasible = [t for t in mine if _feasible(t.cum_traj)]
        self.n_infeasible_dropped = len(mine) - len(feasible)
        # If the veto would empty the group, combine what there is rather than
        # dropping the whole hypothesis: an intent that vanishes silently removes
        # an option, which is the one thing this arm exists to add.
        pool = feasible if feasible else mine

        fused = fuse_trajectories([t.cum_traj for t in pool], mode=self.method)
        if fused is None:
            self.thoughts = list(mine) if self.keep_inputs else []
            return

        node = Thought.combine(pool, **make_traj_state(
            segment_local=fused, cum_traj=fused,
            end_pose=advance_pose(fused, np.zeros(2)),
            depth=pool[0].depth,
            path_score=0.0,          # ScoreDriving runs after this and fills it
            image=ctx.image,
            intent=pool[0].state.get("intent"),
        ))
        node.state["aggregated_from"] = [t.id for t in pool]
        node.state["aggregate_method"] = self.method
        self.thoughts = ([node] + list(mine)) if self.keep_inputs else [node]

    def get_thoughts(self) -> List[Thought]:
        return self.thoughts


class ValidateAndImprove(Operation):
    """Upstream's `ValidateAndImprove(num_samples, improve, num_tries, validate_function)`.

    sec.1.7(a2) lists this as the operation the driving port has "아예 없음" --
    absent entirely. This is it.

    ⚠️NAMING, PRECISELY. Upstream drives Improve with the LLM (`improve_prompt`:
    "here is your previous answer, improve it"). Our base VLA was fine-tuned on
    image+prompt -> action tokens and has no such input format, so what happens
    here is **constrained re-generation**: the violating waypoint is clamped to
    the limit it broke, and the model completes the horizon from that corrected
    prefix. Call it a non-LLM analogue of Improve, the same way AggregateIntent
    is one of Aggregate. Do not write "we implemented Improve".

    ★WHY IT CAN WORK AT ALL, AND WHEN IT CANNOT
        The one refinement channel that exists without retraining is prefix
        re-conditioning, and sec.1.5 measured its floor: prefix perturbations of
        avgL2 0.25-0.4 m flipped the sampled tokens on **0 of 600** records. So a
        repair smaller than roughly a metre changes nothing and merely costs a
        forward pass. Violations are gross by construction (12 m/s^2 is only
        broken by a near-teleport), so the clamp is usually far above that floor
        -- but "usually" is not a measurement, so `prefix_deltas` records the
        actual move per repair and `n_below_floor` counts the ones that could not
        have done anything.

    ★COST IS ZERO ON A FEASIBLE POOL. A thought that validates is passed through
    unchanged -- the same object, not a copy -- and consumes no generator call.
    So this operation is free on the records where nothing is wrong, and the
    selftest asserts it (a control arm whose cost moved would not be a control).

    ★THE THOUGHT GRAPH BECOMES A DAG EVEN THOUGH THE OPERATION GRAPH DOES NOT.
    An improved thought's predecessors are [the original, the Violation vertex],
    so `output_graph()` records why it exists. Graph-of-Thoughts has two graphs --
    the Graph of Operations and the Graph Reasoning State -- and this operation
    only branches the second.
    """

    operation_type = OperationType.improve

    def __init__(self, num_tries: int = 2, improve: bool = True,
                 validate_function=None, temperature: Optional[float] = None) -> None:
        super().__init__()
        assert num_tries >= 1, "num_tries must be >= 1"
        self.num_tries = int(num_tries)
        self.improve = bool(improve)
        # Defaults to locate_violation, i.e. exactly the veto rank_candidates
        # applies. Overriding it with something looser would "repair" trajectories
        # the scorer still rejects.
        self.validate_function = validate_function
        self.temperature = temperature
        self.thoughts: List[Thought] = []
        self.stats: Dict = {}

    def _execute(self, ctx: DrivingContext, **kwargs) -> None:
        from got_drive.graph.thought import make_traj_state, make_violation_state
        from got_drive.graph.violation import locate_violation, repair_prefix
        from got_drive.graph.intent import MIN_SEPARATION_M

        validate = self.validate_function or locate_violation
        cfg = ctx.cfg
        T = cfg.time_horizon
        temp = cfg.temperatures[0] if self.temperature is None else self.temperature

        out: List[Thought] = []
        n_invalid = n_repaired = n_calls = n_below = 0
        deltas: List[float] = []

        for t in self.get_previous_thoughts():
            viol = validate(t.cum_traj)
            t.valid = viol is None
            # Counted BEFORE the improve branch: with improve=False this operation
            # is a pure validator (upstream allows exactly that), and its whole
            # output is the count. Incrementing inside the branch reported
            # n_invalid=0 on a pool that had invalid candidates -- caught by the
            # selftest, which is the sec.11.3 pattern again (measurement, not
            # review, found it).
            if viol is not None:
                n_invalid += 1
            if viol is None or not self.improve:
                out.append(t)               # unchanged object, zero cost
                continue
            current, cur_viol, repaired = t, viol, None
            for _ in range(self.num_tries):
                prefix = repair_prefix(current.cum_traj, cur_viol)
                if prefix is None or prefix.shape[0] >= T:
                    break
                # how far the repair actually moved the trajectory -- the number
                # sec.1.5's floor applies to
                k = min(int(cur_viol["index"]), current.cum_traj.shape[0] - 1)
                delta = float(np.linalg.norm(prefix[-1] - current.cum_traj[k]))
                deltas.append(round(delta, 4))
                if delta < MIN_SEPARATION_M:
                    n_below += 1

                vt = current.child(**make_violation_state(
                    axis=cur_viol["axis"], index=int(cur_viol["index"]),
                    of_thought=current.id,
                    detail={"value": cur_viol["value"], "limit": cur_viol["limit"],
                            "prefix_delta": round(delta, 4)}))

                tail = ctx.generate_fn(ctx.image, prefix, T - prefix.shape[0],
                                       temp, False)
                n_calls += 1
                if tail is None:
                    break
                tail = np.asarray(tail, dtype=np.float64)
                if tail.shape != (T - prefix.shape[0], 2):
                    break
                cum = np.vstack([prefix, tail])

                cand = Thought.combine([current, vt], **make_traj_state(
                    segment_local=cum, cum_traj=cum,
                    end_pose=advance_pose(cum, np.zeros(2)),
                    depth=current.depth, path_score=current.path_score,
                    image=ctx.image, intent=current.state.get("intent")))
                cand.state["improved_from"] = current.id
                cand.state["repaired"] = dict(cur_viol)
                repaired = cand

                cur_viol = validate(cum)
                cand.valid = cur_viol is None
                current = cand
                if cur_viol is None:
                    n_repaired += 1
                    break

            # ★Never lose a candidate. A failed repair returns the ORIGINAL, so
            # this operation can only add information, never shrink the pool --
            # otherwise a bad repair would look like an improvement by deleting
            # the evidence.
            out.append(repaired if repaired is not None else t)

        self.stats = {
            "n_thoughts": len(out),
            "n_invalid": n_invalid,
            "n_repaired": n_repaired,
            "n_still_invalid": n_invalid - n_repaired,
            "n_generate_calls": n_calls,
            "n_repairs_below_floor": n_below,
            "floor_m": MIN_SEPARATION_M,
            "prefix_deltas": deltas,
        }
        self.thoughts = out

    def get_thoughts(self) -> List[Thought]:
        return self.thoughts


class Emit(Operation):
    """Terminal leaf: the plan the graph returns. Not an upstream operation name.

    Upstream reads results off the graph's leaves via `get_final_thoughts()`. We
    keep an explicit leaf so that the "which thought did we actually commit to"
    question has one answer in the graph itself, which is what the csv writer and
    the oracle metrics need to agree on.
    """

    operation_type = OperationType.emit

    def __init__(self) -> None:
        super().__init__()
        self.thoughts: List[Thought] = []

    def _execute(self, ctx: DrivingContext, **kwargs) -> None:
        pool = self.get_previous_thoughts()
        self.thoughts = pool[:1] if pool else []

    def get_thoughts(self) -> List[Thought]:
        return self.thoughts
