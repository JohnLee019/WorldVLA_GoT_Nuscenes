"""The controller. Shaped after `graph_of_thoughts/controller/controller.py`.

Upstream's loop, verbatim in structure:

    execution_queue = [op for op in self.graph.operations if op.can_be_executed()]
    while len(execution_queue) > 0:
        current_operation = execution_queue.pop(0)
        current_operation.execute(self.lm, self.prompter, self.parser, **params)
        for operation in current_operation.successors:
            if operation.can_be_executed():
                execution_queue.append(operation)

★IT IS A PLAIN FIFO OVER `can_be_executed()`. No policy, no scoring of operations,
no adaptive budget. Worth stating because it is counter-intuitive: the "graph" in
Graph of Thoughts is the DECLARED TOPOLOGY (see graph_of_operations.py), not a
clever scheduler. Any adaptive-controller idea is an extension beyond upstream and
must be reported as one -- and sec.1.2 already measured that the signals such a
policy would route on (`candidate_spread` -0.0123, score margin -0.0024,
`worst_candidate` -0.0231) are indistinguishable from random.

`output_graph()` is our trace dump. It exists for the same reason `got_cand_wps`
does: sec.1.10(a2) could not separate "the model did not generate it" from "the beam
pruned it" because the pre-prune pool was not retained, and sec.1.17 was only
possible because the surviving candidates' full trajectories were. Record the
execution graph from the start rather than discovering later that it is needed.
"""

import json
from typing import List, Optional

import numpy as np

from got_drive.graph.graph_of_operations import GraphOfOperations
from got_drive.graph.operations import DrivingContext, Emit, KeepBestN
from got_drive.graph.thought import Thought


class Controller:
    def __init__(self, graph: GraphOfOperations, ctx: DrivingContext) -> None:
        self.graph = graph
        self.ctx = ctx
        self.run_executed = False

    def run(self) -> "Controller":
        assert self.graph.roots, "the graph has no root operation"
        assert all(not op.executed for op in self.graph.operations), (
            "operations already executed -- build a fresh GraphOfOperations per "
            "record. Reusing one would leak the previous record's thoughts into "
            "this record's pool, which is exactly the stale-state failure "
            "DriveGoTPipeline.plan() resets against.")

        execution_queue = [op for op in self.graph.operations if op.can_be_executed()]
        while len(execution_queue) > 0:
            current_operation = execution_queue.pop(0)
            current_operation.execute(self.ctx)
            for operation in current_operation.successors:
                if operation.can_be_executed():
                    execution_queue.append(operation)

        unexecuted = [op.id for op in self.graph.operations if not op.executed]
        assert not unexecuted, (
            f"operations {unexecuted} never became executable -- the graph has a "
            f"cycle or an unsatisfied predecessor. Upstream's FIFO cannot make "
            f"progress on either.")
        self.run_executed = True
        return self

    # ── results ──────────────────────────────────────────────────────────────
    def get_final_thoughts(self) -> List[List[Thought]]:
        """Upstream's accessor: thoughts of every leaf operation."""
        assert self.run_executed, "call run() first"
        return [op.get_thoughts() for op in self.graph.leaves]

    def plan(self) -> Optional[np.ndarray]:
        """The committed trajectory, or None if the graph produced nothing.

        Same contract as DriveGoTPipeline.plan()'s first return value, so an eval
        script can swap one for the other.
        """
        assert self.run_executed, "call run() first"
        for op in self.graph.operations:
            if isinstance(op, Emit) and op.get_thoughts():
                return op.get_thoughts()[0].cum_traj
        return None

    def final_pool(self) -> List[Thought]:
        """The deepest scored pool, survivors AND pruned, best first.

        This is what `minADE_C` and the selection gap are computed over -- the
        diagnostic that separates "the generator had nothing good" from "the score
        failed to pick it" (sec.7.4). Reads the LAST KeepBestN's retained pool.

        ⚠️IT IS THE POOL THAT REACHED KeepBestN, WHICH IS NOT ALWAYS THE POOL THE
        GENERATOR PRODUCED. `KeepValid(enabled=True)` filters upstream of it, so
        under `--graph_keep_valid` this returns the FEASIBLE SUBSET and minADE_C is
        computed over a smaller denominator -- a lower value across that pair is
        not a better generator. `KeepValid.n_dropped` says by how much (the eval
        writes it per record) and `KeepValid.dropped` keeps the removed thoughts,
        which output_graph() serialises so the full pool is recoverable offline.
        """
        assert self.run_executed, "call run() first"
        last = None
        for op in self.graph.operations:
            if isinstance(op, KeepBestN) and op.executed and op.all_thoughts:
                last = op
        return list(last.all_thoughts) if last is not None else []

    def output_graph(self, path: Optional[str] = None) -> dict:
        """Serialise the executed graph: topology + every thought's trajectory.

        Written per record, this is what makes an offline post-hoc analysis
        possible with zero new inference -- the pattern that let sec.1.17 recover
        the branch structure from a finished run.

        ★IT WALKS THE PREDECESSOR CLOSURE, IT DOES NOT JUST LIST THE OPERATIONS'
        OUTPUTS. Iterating `op.get_thoughts()` alone missed the Violation vertices
        ValidateAndImprove creates: an improved thought lists one as a predecessor,
        but no operation returns it, so every improve-arm dump carried a DANGLING
        predecessor id (measured: 1 dangling id on the dummy generator) and
        ValidateAndImprove's claim that "output_graph() records why it exists" was
        false. A closure walk makes "every referenced vertex is serialised" a
        property of the algorithm rather than of which operations happen to expose
        their intermediates. `KeepBestN.all_thoughts` (the pruned pool) and
        `KeepValid.dropped` (the filtered-out pool) are seeded explicitly for the
        same reason -- they are the diagnostic sec.1.10(a2) could not run because
        the pre-prune pool had not been retained.

        ★AND IT EMITS PROVENANCE. `improved_from` / `repaired` /`aggregated_from` /
        `aggregate_method` are the only record of WHY a derived thought exists; a
        dump without them shows an aggregate as an unexplained extra trajectory.
        """
        assert self.run_executed, "call run() first"

        # Seed: everything any operation can show us, including the pools that are
        # deliberately not passed downstream.
        seeds: List[Thought] = []
        for op in self.graph.operations:
            seeds.extend(op.get_thoughts())
            seeds.extend(getattr(op, "all_thoughts", None) or [])   # KeepBestN
            seeds.extend(getattr(op, "dropped", None) or [])        # KeepValid

        thoughts = {}
        stack = list(seeds)
        while stack:
            t = stack.pop()
            if t.id in thoughts:
                continue
            rec = {
                "id": t.id,
                "kind": t.state.get("kind"),
                "depth": t.depth,
                "intent": t.state.get("intent"),
                "predecessors": [p.id for p in t.predecessors],
                "path_score": t.path_score,
                "segment_score": t.state.get("segment_score"),
                "components": t.state.get("components", {}),
                "valid": t.valid if t.validated else None,
                "cum_traj": t.cum_traj.round(4).tolist(),
                # provenance -- None on a plain Generate output, populated on the
                # thoughts the new operations create. Written unconditionally so
                # the csv/json schema does not change between arms (sec.9: a
                # missing column reads as a different eval).
                "improved_from": t.state.get("improved_from"),
                "repaired": t.state.get("repaired"),
                "aggregated_from": t.state.get("aggregated_from"),
                "aggregate_method": t.state.get("aggregate_method"),
            }
            if rec["kind"] == "violation":
                # A violation vertex carries no trajectory; without these it would
                # serialise as an empty shell and the edge would explain nothing.
                rec.update({"axis": t.state.get("axis"),
                            "index": t.state.get("index"),
                            "of_thought": t.state.get("of_thought"),
                            "detail": t.state.get("detail", {})})
            thoughts[t.id] = rec
            stack.extend(t.predecessors)
        out = {
            "topology": self.graph.topology(),
            "is_chain": self.graph.is_chain(),
            "command": self.ctx.command,
            "thoughts": [thoughts[k] for k in sorted(thoughts)],
        }
        if path:
            with open(path, "w") as f:
                json.dump(out, f, indent=2)
        return out
