"""The vertex type. Deliberately a near-copy of the official `Thought`.

Upstream (`graph_of_thoughts/operations/thought.py`):

    class Thought:
        def __init__(self, state: Optional[Dict] = None) -> None:
            self.id: int = next(Thought._ids)
            self.state: Dict = state
            self._score: float = 0.0
            self._valid: bool = False
            self._solved: bool = False
            self.scored / self.validated / self.compared_to_ground_truth = False

★THE STATE IS A PLAIN DICT, AND THAT IS THE POINT. An earlier sketch of this
package proposed four dataclasses (Intent / Traj / Violation / Evidence). That was
over-engineering: upstream expresses heterogeneous thoughts as differently-shaped
`state` dicts under one class, which is what lets one graph hold a driving
hypothesis, a realised trajectory and a constraint violation without the operations
needing to know a type hierarchy. We follow that.

State keys this package uses (all optional -- read with .get):

    kind          "root" | "traj" | "intent" | "violation"
    segment_local (segment_len, 2) float64, in the generating frame
    cum_traj      (n, 2) float64, cumulative waypoints in the ORIGINAL frame
    end_pose      (position (2,), heading float) after this segment, original frame
    depth         which time segment produced it (-1 for the root)
    image         the observation frame used to generate this thought's children
    path_score    accumulated score along the path, what the beam sorts on
    components    {"kinematic": float, "command": float, "wm": float, ...}
    intent        driving hypothesis, once intent nodes land (see package docstring)

`_score` is kept as upstream's per-thought score. For driving, ranking happens on
`state["path_score"]` (accumulated) rather than on `_score` (this segment only),
because that is what the incumbent pipeline sorts on and the no-op arm has to
reproduce it exactly. Both are populated; nothing reads `_score` for ordering yet.
"""

import itertools
import logging
from typing import Dict, Optional, Tuple

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# state constructors
#
# The dict stays the storage (upstream fidelity, and output_graph() serialises it
# without a custom encoder), but nothing should hand-roll one: a typo'd key or a
# missing `end_pose` would surface far from its cause, as a shape error inside a
# later Generate. These builders are where a malformed state is impossible, which
# is the one thing a dataclass would have bought.
#
# ★A `kind` here is a REPRESENTATION tag, not an algorithm choice. Whether the
# graph branches on time segments or on driving intents is decided by the
# GraphOfOperations, not by which builders exist -- the two are orthogonal.
# ─────────────────────────────────────────────────────────────────────────────

REQUIRED_KEYS = {
    "root":      ("segment_local", "cum_traj", "end_pose", "depth", "path_score"),
    "traj":      ("segment_local", "cum_traj", "end_pose", "depth", "path_score"),
    "intent":    ("intent", "depth", "path_score"),
    "violation": ("axis", "index", "of_thought"),
}


def validate_state(state: Dict) -> Dict:
    """Assert a state dict is well formed for its `kind`. Returns it for chaining."""
    kind = state.get("kind")
    if kind not in REQUIRED_KEYS:
        raise ValueError(f"unknown thought kind {kind!r}; "
                         f"expected one of {sorted(REQUIRED_KEYS)}")
    missing = [k for k in REQUIRED_KEYS[kind] if k not in state]
    if missing:
        raise ValueError(f"{kind} thought is missing {missing}")
    for key in ("segment_local", "cum_traj"):
        arr = state.get(key)
        if arr is not None and (not isinstance(arr, np.ndarray) or arr.ndim != 2
                                or arr.shape[-1] != 2):
            raise ValueError(f"{key} must be a (n, 2) float array, got "
                             f"{type(arr).__name__} {getattr(arr, 'shape', None)}")
    return state


def make_root_state(image=None) -> Dict:
    """The empty trajectory at the ego origin, heading 0."""
    return validate_state({
        "kind": "root",
        "segment_local": np.empty((0, 2), dtype=np.float64),
        "cum_traj": np.empty((0, 2), dtype=np.float64),
        "end_pose": (np.zeros(2), 0.0),
        "image": image,
        "depth": -1,
        "path_score": 0.0,
        "components": {},
    })


def make_traj_state(segment_local: np.ndarray, cum_traj: np.ndarray,
                    end_pose: Tuple[np.ndarray, float], depth: int,
                    path_score: float = 0.0, image=None,
                    intent: Optional[Dict] = None) -> Dict:
    """One realised continuation. `path_score` is the parent's; Score adds to it."""
    return validate_state({
        "kind": "traj",
        "segment_local": np.asarray(segment_local, dtype=np.float64),
        "cum_traj": np.asarray(cum_traj, dtype=np.float64),
        "end_pose": end_pose,
        "image": image,
        "depth": int(depth),
        "path_score": float(path_score),
        "components": {},
        "intent": intent,
    })


def make_intent_state(intent: Dict, depth: int = 0, image=None) -> Dict:
    """A driving hypothesis, before any trajectory realises it.

    Not produced by the staged (control) graph -- it is what an intent-branching
    graph roots its search on. Handoff Step 2.9/2.10 fix the axes this should
    carry: diversity in the SPEED PROFILE, curvature as symmetric pairs near 0,
    and the conservative end (decelerate / stop) mandatory.
    """
    return validate_state({
        "kind": "intent",
        "intent": dict(intent),
        "image": image,
        "depth": int(depth),
        "path_score": 0.0,
        "components": {},
    })


def make_violation_state(axis: str, index: int, of_thought: int,
                         detail: Optional[Dict] = None) -> Dict:
    """A named constraint failure, as a vertex.

    ★This is the shape a tree cannot express, and the reason the port had no
    Improve: `_feasible` already computes WHICH limit was breached and at which
    waypoint (sec.1.5 measured acceleration as 71-74% of violations, concentrated
    at index 3-4, the segment 2->3 seam), and today that is collapsed into one
    scalar and discarded. As a vertex it becomes an input.
    """
    return validate_state({
        "kind": "violation",
        "axis": str(axis),
        "index": int(index),
        "of_thought": int(of_thought),
        "detail": dict(detail or {}),
    })


class Thought:
    """A vertex in the graph reasoning state."""

    _ids = itertools.count(0)

    def __init__(self, state: Optional[Dict] = None) -> None:
        self.logger: logging.Logger = logging.getLogger(self.__class__.__name__)
        self.id: int = next(Thought._ids)
        self.state: Dict = {} if state is None else state
        self._score: float = 0.0
        self._valid: bool = False
        self._solved: bool = False
        self.scored: bool = False
        self.validated: bool = False
        self.compared_to_ground_truth: bool = False
        # ★Graph, not tree: a thought can have several predecessors. Aggregate is
        # the operation that produces one, and it is the only reason this is a
        # list rather than the single `parent` DrivePathNode carries.
        self.predecessors: list = []

    # ── upstream property/setter pairs (the flag is set by the setter) ────────
    @property
    def valid(self) -> bool:
        return self._valid

    @valid.setter
    def valid(self, valid: bool) -> None:
        self._valid = valid
        self.validated = True

    @property
    def score(self) -> float:
        return self._score

    @score.setter
    def score(self, new_score: float) -> None:
        self._score = new_score
        self.scored = True

    @property
    def solved(self) -> bool:
        return self._solved

    @solved.setter
    def solved(self, solved: bool) -> None:
        self._solved = solved
        self.compared_to_ground_truth = True

    # ── driving conveniences (thin; the dict stays the source of truth) ───────
    @property
    def cum_traj(self) -> np.ndarray:
        return self.state.get("cum_traj", np.empty((0, 2), dtype=np.float64))

    @property
    def path_score(self) -> float:
        return float(self.state.get("path_score", 0.0))

    @property
    def depth(self) -> int:
        return int(self.state.get("depth", -1))

    def child(self, **state) -> "Thought":
        """A new thought that records this one as a predecessor.

        Use for Generate/Refine (one predecessor). Aggregate uses `combine`.
        """
        t = Thought(state)
        t.predecessors = [self]
        return t

    @staticmethod
    def combine(parents, **state) -> "Thought":
        """A new thought derived from SEVERAL predecessors.

        ★This is the only constructor that makes the reasoning state a graph
        rather than a tree, and it is the thing sec.1.7 found missing: the port
        called scalar accumulation "Aggregate" and parent-chain walking "Merge"
        while every thought still had exactly one parent. A thought built here
        has n, and `output_graph()` records all n, so the provenance of a fused
        trajectory is inspectable instead of asserted.
        """
        parents = list(parents)
        assert parents, "combine() needs at least one predecessor"
        t = Thought(state)
        t.predecessors = parents
        return t

    def __repr__(self) -> str:  # debugging only; never parsed
        k = self.state.get("kind", "?")
        n = self.cum_traj.shape[0]
        return (f"<Thought {self.id} {k} depth={self.depth} n_wp={n} "
                f"path={self.path_score:+.3f}>")
