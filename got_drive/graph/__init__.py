"""A real Graph of Thoughts for driving, shaped after the official implementation.

WHY THIS PACKAGE EXISTS
-----------------------
`got_drive/got_pipeline_drive.py` is a beam search: Generate -> Score -> KeepBestN,
repeated over three time segments. That is Tree of Thoughts. Handoff sec.1.7 caught
this once already -- the port had inherited the names `Aggregate` and `Merge` from
upstream while implementing scalar accumulation and parent-chain walking -- and
sec.1.7(a2) lists what is still missing against `spcl/graph-of-thoughts`:

    Generate    always LLM            ✅ base VLA samples it
    Score       LLM by default, but `scoring_function` is an explicit escape hatch
                                      ✅ kinematic+command is a legitimate one
    Aggregate   always LLM            ⚠️ only a non-LLM analogue was tested (sec.1.5)
    Improve /
    ValidateAndImprove
                always LLM            ⚠️ absent entirely

So this package rebuilds the deliberation layer on the official object model --
`Thought` with a state dict, `Operation` with predecessors/successors, a declared
`GraphOfOperations`, and a FIFO `Controller` gated on `can_be_executed()` -- so that
the operations the driving port never had can actually be expressed.

WHAT THIS IS NOT
----------------
★It is NOT an attempt to beat greedy on nuScenes. Handoff sec.1.12 measured the
information ceiling of a single frame as SATURATED (`resid` skill 0 on BOTH the
VQGAN encoder and a deliberately weakened reference encoder), and the absorption
law (sec.1 claims 2 and 11, three cases, both directions, ~103%) says pool changes
do not move the output. Expect a null. The value is elsewhere:

  1. the claim "we ported GoT" becomes true, and a null over the COMPLETE operator
     set is a much stronger negative result than a null over a partial one;
  2. it converts sec.1.17's diagnosis ("deliberation happens late, error accumulates
     early") from an ARGUMENT into a MEASUREMENT -- intent-shaped nodes move
     deliberation to the first decision, so if it still loses, that competing
     explanation is eliminated and sec.1.1 stands alone;
  3. NAVSIM needs it anyway: Step 2.11 measured that transplanting the current
     candidate generator to navtrain scores `GAP_dep` -0.0942, i.e. WORSE than a
     random pool, and Step 2.9/2.10 prescribe the fix as "diversity in the speed
     profile, curvature as symmetric pairs near 0, conservative end mandatory" --
     which is an intent axis by another name.

THE GATE THIS PACKAGE MUST PASS FIRST
-------------------------------------
Before any new operation is trusted, the machinery has to reproduce what is already
measured. `build_staged_goo()` declares the CURRENT pipeline as a graph, and
`got_drive/graph/selftest.py` asserts it returns the identical trajectory with the
identical number of generator calls as `DriveGoTPipeline`. Same idea as sec.1.5's
`--fuse_top_m 1` control and E3's identity rung: if the no-op arm does not
reproduce, nothing downstream of it means anything.
"""

from got_drive.graph.thought import (
    Thought,
    make_root_state,
    make_traj_state,
    make_intent_state,
    make_violation_state,
    validate_state,
)
from got_drive.graph.intent import (
    Intent,
    make_intent_grid,
    anchor_waypoint,
    separation_report,
    DEFAULT_SPEED_SCALES,
    DEFAULT_CURVATURES,
    USABLE_CURVATURES,
    MIN_SEPARATION_M,
)
from got_drive.graph.operations import (
    Operation,
    OperationType,
    DrivingContext,
    Root,
    GenerateSegment,
    GenerateIntents,
    RealizeIntent,
    AggregateIntent,
    ValidateAndImprove,
    ScoreDriving,
    KeepValid,
    KeepBestN,
    Emit,
)
from got_drive.graph.graph_of_operations import (
    GraphOfOperations, build_staged_goo, build_intent_goo, build_aggregate_goo,
    build_improve_goo)
from got_drive.graph.controller import Controller
from got_drive.graph.planner import GraphPlanner

__all__ = [
    "Thought",
    "make_root_state",
    "make_traj_state",
    "make_intent_state",
    "make_violation_state",
    "validate_state",
    "GraphPlanner",
    "Operation",
    "OperationType",
    "DrivingContext",
    "Root",
    "GenerateSegment",
    "ScoreDriving",
    "KeepValid",
    "KeepBestN",
    "Emit",
    "GraphOfOperations",
    "build_staged_goo",
    "build_intent_goo",
    "build_aggregate_goo",
    "build_improve_goo",
    "AggregateIntent",
    "ValidateAndImprove",
    "Intent",
    "make_intent_grid",
    "anchor_waypoint",
    "separation_report",
    "GenerateIntents",
    "RealizeIntent",
    "Controller",
]
