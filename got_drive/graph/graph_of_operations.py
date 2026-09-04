"""The Graph of Operations (GoO). Shaped after upstream's class of the same name.

Upstream fields: `operations`, `roots`, `leaves`.
Upstream methods:
    append_operation(op)  -- link to ALL current leaves; linear chaining
    add_operation(op)     -- respect predecessors/successors already set on `op`,
                             which is what allows arbitrary DAG topologies

★THAT SECOND METHOD IS WHERE "GRAPH" LIVES. Upstream's controller is a plain FIFO
queue -- it is not an adaptive planner, and an earlier sketch of this package wrongly
proposed one. The flexibility of Graph of Thoughts comes from the declared topology:
an Aggregate operation with several predecessors, a ValidateAndImprove loop, an
operation reachable from two different branches. `append_operation` alone can only
ever build a chain, i.e. a Tree of Thoughts pipeline -- which is precisely what
`build_staged_goo()` below reconstructs, on purpose, as the control arm.
"""

from typing import List

from got_drive.got_pipeline_drive import DriveGoTConfig
from got_drive.graph.operations import (
    AggregateIntent,
    Emit,
    GenerateIntents,
    GenerateSegment,
    KeepBestN,
    KeepValid,
    Operation,
    RealizeIntent,
    Root,
    ValidateAndImprove,
    ScoreDriving,
)


class GraphOfOperations:
    def __init__(self) -> None:
        self.operations: List[Operation] = []
        self.roots: List[Operation] = []
        self.leaves: List[Operation] = []

    def append_operation(self, operation: Operation) -> None:
        """Append to all current leaves. Linear chaining."""
        self.operations.append(operation)
        if not self.roots:
            self.roots = [operation]
        else:
            for leaf in self.leaves:
                leaf.add_successor(operation)
        self.leaves = [operation]

    def add_operation(self, operation: Operation) -> None:
        """Insert respecting whatever predecessors/successors `operation` carries.

        This is the entry point for non-chain topologies (Aggregate over several
        branches, an Improve loop feeding back into Score).
        """
        self.operations.append(operation)
        if not self.roots:
            self.roots = [operation]
            self.leaves = [operation]
            assert not operation.predecessors, (
                "first operation added to an empty graph cannot have predecessors")
            return
        if not operation.predecessors:
            self.roots.append(operation)
        for p in operation.predecessors:
            if p in self.leaves:
                self.leaves.remove(p)
        if not operation.successors:
            self.leaves.append(operation)

    # ── introspection, for the trace dump and the selftest ───────────────────
    def topology(self) -> List[dict]:
        return [{
            "id": op.id,
            "op": type(op).__name__,
            "type": op.operation_type.name if op.operation_type else None,
            "predecessors": [p.id for p in op.predecessors],
            "successors": [s.id for s in op.successors],
            "executed": op.executed,
            "n_thoughts": len(op.get_thoughts()) if op.executed else 0,
        } for op in self.operations]

    def is_chain(self) -> bool:
        """True when no operation has more than one predecessor.

        ★This is the ToT/GoT discriminator, and it is worth asserting in tests:
        `build_staged_goo()` MUST return True here (it is the control arm, a beam
        search), and any graph that adds Aggregate MUST return False. Handoff
        sec.1.7 exists because that distinction was asserted in prose and turned
        out to be false in code -- so assert it in code.
        """
        return all(len(op.predecessors) <= 1 for op in self.operations)


def build_staged_goo(cfg: DriveGoTConfig, keep_valid: bool = False) -> GraphOfOperations:
    """The CURRENT pipeline, declared as a graph. This is the no-op control arm.

        Root -> [Generate -> Score -> KeepValid -> KeepBestN] x n_segments -> Emit

    It must reproduce DriveGoTPipeline exactly: same trajectory, same number of
    generator calls. got_drive/graph/selftest.py asserts both. Until that passes,
    nothing built on this machinery can be compared with anything in sec.1.

    `keep_valid=False` keeps KeepValid a pass-through -- see its docstring; the
    incumbent's veto still lives inside the score, so filtering here too would be a
    different arm wearing the control arm's name.
    """
    goo = GraphOfOperations()
    goo.append_operation(Root())
    for seg_idx in range(cfg.n_segments):
        goo.append_operation(GenerateSegment(seg_idx))
        goo.append_operation(ScoreDriving(seg_idx))
        goo.append_operation(KeepValid(enabled=keep_valid))
        goo.append_operation(KeepBestN(cfg.beam_width))
    goo.append_operation(Emit())
    return goo


def build_intent_goo(cfg: DriveGoTConfig, intents=None, variants: int = 2,
                     keep_valid: bool = False, anchor_steps: int = 1) -> GraphOfOperations:
    """Branch on driving hypotheses instead of on time.

        Root -> GenerateIntents -> RealizeIntent -> Score -> KeepValid
             -> KeepBestN(1) -> Emit

    WHAT CHANGES VERSUS build_staged_goo, AND WHY EACH CHANGE IS PRESCRIBED

      * There are no time segments. Once a node is a full-horizon hypothesis
        there is no reason to walk the horizon in lockstep, and walking it is
        what produced sec.1.17's structural mismatch (branching 1.52 -> 2 -> 7.95
        while the decisive longitudinal error is fixed in the first segment).
      * Stage one has len(intents) options BY CONSTRUCTION, against the measured
        1.52 (48% of records with exactly one).
      * The diversity axis is the speed profile with a mandatory conservative
        end, not temperature. Step 2.11 measured our current pool as `GAP_dep`
        -0.0942 -- worse than random -- with "zero curvature diversity, speed
        ladder at v x 0.60-1.50 when the room is at v x 0.00-0.45" as the cause.
      * KeepBestN(1): with no segments there is one prune, at the end.

    ⚠️THIS IS NOT A CONTROL ARM. It cannot reproduce 3.5557 and is not supposed
    to; `build_staged_goo` is the control. Report it separately (results/graph/),
    with its call count, and never paired against sec.1's tables.

    ⚠️Read `RealizeIntent.separation` before reading the L2. If `separated` is
    False the intents collapsed to one plan (sec.1.5: sub-metre prefix
    perturbations flip zero tokens in 600 records) and the arm measured nothing.
    """
    goo = GraphOfOperations()
    goo.append_operation(Root())
    goo.append_operation(GenerateIntents(intents))
    goo.append_operation(RealizeIntent(variants=variants, anchor_steps=anchor_steps))
    goo.append_operation(ScoreDriving(0))
    goo.append_operation(KeepValid(enabled=keep_valid))
    goo.append_operation(KeepBestN(1))
    goo.append_operation(Emit())
    return goo


def build_aggregate_goo(cfg: DriveGoTConfig, intents=None, variants: int = 3,
                        method: str = "median", keep_inputs: bool = False,
                        keep_valid: bool = False,
                        anchor_steps: int = 1) -> GraphOfOperations:
    """★The first graph in this package that is not a chain.

        Root -> GenerateIntents -> RealizeIntent -+-> AggregateIntent(i_0) -+
                                                  +-> AggregateIntent(i_1) -+-> Score
                                                  +-> ...                   +
                                                                            |
                                          KeepValid -> KeepBestN(1) -> Emit <+

    `is_chain()` returns False here and True for the other two builders, and the
    selftest asserts both. That assertion is the whole point: sec.1.7 caught this
    project claiming a graph while running a beam search, because the claim lived
    in prose. Now it lives in a test.

    WHY THE FAN-OUT IS PER INTENT
        One AggregateIntent per hypothesis, each combining only that hypothesis's
        realisations. sec.1.5's fusion failed by averaging across different plans
        (turns hurt 3.7x more than straights, infeasibility 2.5% -> 41.8%); with
        one operation per intent that mixture cannot occur. See AggregateIntent's
        docstring for why this is a re-interpretation of that null rather than a
        retry of it.

    WHY SCORE COMES AFTER, NOT BEFORE
        The aggregate is a trajectory no candidate proposed, so it has never been
        scored or vetoed. Scoring the fan-in means the combination competes on the
        same criterion as everything else instead of being adopted by fiat.

    ⚠️`variants` defaults to 3 here, not 2: combining two samples is a midpoint,
    and the median -- the estimator sec.1.5 argued for, matched to an L2 metric --
    only starts to differ from the mean at three. Cost is 1 + n_intents*variants.

    ⚠️`keep_inputs=False` (upstream's semantics) means the pool handed downstream
    is the aggregates, so `minADE_C` is computed over a DIFFERENT pool than the
    other arms. Pass True to keep the realisations alongside and preserve
    comparability -- and say which one a reported number came from.
    """
    goo = GraphOfOperations()
    root = Root()
    goo.append_operation(root)
    gi = GenerateIntents(intents)
    goo.append_operation(gi)
    ri = RealizeIntent(variants=variants, anchor_steps=anchor_steps)
    goo.append_operation(ri)

    aggs = []
    for it in gi.intents:
        a = AggregateIntent(it.name, method=method, keep_inputs=keep_inputs)
        a.add_predecessor(ri)
        goo.add_operation(a)
        aggs.append(a)

    score = ScoreDriving(0)
    for a in aggs:
        score.add_predecessor(a)      # ← the fan-in: >1 predecessor, hence a DAG
    goo.add_operation(score)

    goo.append_operation(KeepValid(enabled=keep_valid))
    goo.append_operation(KeepBestN(1))
    goo.append_operation(Emit())
    return goo


def build_improve_goo(cfg: DriveGoTConfig, intents=None, variants: int = 3,
                      num_tries: int = 2, method: str = "median",
                      aggregate: bool = True, keep_inputs: bool = False,
                      keep_valid: bool = False,
                      anchor_steps: int = 1) -> GraphOfOperations:
    """The full operator set: Generate -> (Aggregate) -> ValidateAndImprove -> Score.

        Root -> GenerateIntents -> RealizeIntent -+-> AggregateIntent(i_0) -+
                                                  +-> ...                  +-> VandI
                                                                              |
                                             Score -> KeepValid -> KeepBestN(1) -> Emit

    With `aggregate=False` the fan-out is skipped and ValidateAndImprove runs
    directly on the realisations, which is the cheaper arm and isolates Improve's
    contribution from Aggregate's.

    ★WHY IMPROVE COMES AFTER AGGREGATE, NOT BEFORE
        The combination is a trajectory no candidate proposed, so it is the one
        thing in the graph that has never faced the feasibility veto. sec.1.5
        measured that per-segment fusion pushed infeasibility from a 2.5% base
        rate to 41.8% -- exactly the failure mode a validate-then-repair step
        exists to catch. Putting Improve first would repair the inputs and then
        emit an unchecked combination of them.

    ⚠️COST IS DATA-DEPENDENT. Improve only spends forward passes on candidates
    that fail validation, so calls per record vary with how infeasible the pool
    is. sec.1.5's free pairing (equal call counts -> bit-identical trajectories)
    does NOT hold for this arm; log `n_generate_calls` and report the
    distribution, not a single number.
    """
    goo = GraphOfOperations()
    goo.append_operation(Root())
    gi = GenerateIntents(intents)
    goo.append_operation(gi)
    ri = RealizeIntent(variants=variants, anchor_steps=anchor_steps)
    goo.append_operation(ri)

    if aggregate:
        aggs = []
        for it in gi.intents:
            a = AggregateIntent(it.name, method=method, keep_inputs=keep_inputs)
            a.add_predecessor(ri)
            goo.add_operation(a)
            aggs.append(a)
        vi = ValidateAndImprove(num_tries=num_tries)
        for a in aggs:
            vi.add_predecessor(a)
        goo.add_operation(vi)
    else:
        goo.append_operation(ValidateAndImprove(num_tries=num_tries))

    goo.append_operation(ScoreDriving(0))
    goo.append_operation(KeepValid(enabled=keep_valid))
    goo.append_operation(KeepBestN(1))
    goo.append_operation(Emit())
    return goo
