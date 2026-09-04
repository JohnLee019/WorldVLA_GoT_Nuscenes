"""Self-test for got_drive/graph/. Pure numpy, no GPU, no model, no torch.

    python -m got_drive.graph.selftest

THE ONE THAT MATTERS IS [A]: the graph machinery, wired as `build_staged_goo()`,
must reproduce DriveGoTPipeline exactly -- same trajectory to the last decimal AND
the same number of generator calls. Handoff sec.1.5 (`--fuse_top_m 1`), E3 (the
identity rung) and sec.1.4 (the crop check that passed by comparing an object with
itself) all say the same thing: an equivalence claim is only worth what its no-op
control proves. If [A] fails, nothing built on this package can be put in a table
next to anything in sec.1.

The call-count half of [A] is not decoration. sec.1.4 traced the drifting greedy
baseline to GoT's draw count changing with k/beam, and sec.1.5 found that arms with
equal call counts land on bit-identical trajectories (600/600) because the RNG state
matches. Equal call count is therefore both a correctness check and the thing that
keeps future arms cheaply paired.
"""

import json
import sys

import numpy as np

from got_drive.scoring_driving import DEFAULT_DT, _feasible
from got_drive.got_pipeline_drive import (
    DriveGoTConfig,
    DriveGoTPipeline,
    _make_dummy_generate_fn,
)
from got_drive.graph import (
    Controller,
    DrivingContext,
    Emit,
    GenerateIntents,
    GenerateSegment,
    RealizeIntent,
    GraphOfOperations,
    GraphPlanner,
    KeepBestN,
    KeepValid,
    Root,
    ScoreDriving,
    Thought,
    build_staged_goo,
    build_intent_goo,
    build_aggregate_goo,
    build_improve_goo,
    AggregateIntent,
    ValidateAndImprove,
    make_traj_state,
    validate_state,
    Intent,
    make_intent_grid,
    anchor_waypoint,
    separation_report,
    MIN_SEPARATION_M,
    USABLE_CURVATURES,
)

RESULTS = []


class _Fixed(Root):
    """A source operation returning a fixed thought list, for driving one
    operation in isolation. Needed because §1.4's lesson is that a check which
    cannot fail proves nothing: the dummy generator only makes smooth, in-intent
    trajectories, so KeepValid and AggregateIntent must be fed the cases they
    exist to handle rather than whatever the pipeline happened to produce."""

    def __init__(self, thoughts):
        super().__init__()
        self._fixed = thoughts

    def _execute(self, ctx, **kw):
        self.thoughts = self._fixed




def _make_bumpy_generate_fn(step_x=4.0):
    """Like the dummy generator, but every third sampled call emits a physically
    impossible segment. KeepValid and AggregateIntent are no-ops on a pool where
    nothing is wrong, so a check driven by the smooth dummy cannot fail (sec.1.4).
    """
    base = _make_dummy_generate_fn(step_x)
    n = {"i": 0}

    def gen(image, prefix_wp, n_generate, temperature, do_sample):
        seg = base(image, prefix_wp, n_generate, temperature, do_sample)
        if do_sample:
            n["i"] += 1
            if n["i"] % 3 == 0 and seg is not None and len(seg):
                seg = np.asarray(seg, dtype=np.float64).copy()
                seg[:, 0] += np.arange(1, len(seg) + 1) * 40.0   # teleport
        return seg

    return gen


def _lift_eval_get_args():
    """Execute eval_got_nuscenes.py's REAL `get_args()` without importing it.

    ★WHY THE CONTORTION. The eval imports torch, which is not installed on every
    machine this self-test has to run on, so it cannot be imported. But a copy of
    its argparse defaults is worthless as a guard -- a renamed flag would leave
    the copy green and blow up on record 1 of a real run (sec.1.4: a check that
    cannot fail proves nothing). So the function is lifted out of the module's
    AST and executed on its own; `argparse` is stdlib, so it needs none of the
    module's heavy imports.

    Returns `make(planner) -> Namespace`, or None if the lift failed (the caller
    turns that into a visible FAIL rather than silently falling back).
    """
    import argparse as _argparse, ast as _ast, os as _os, sys as _sys
    path = _os.path.join(_os.path.dirname(_os.path.dirname(
        _os.path.dirname(_os.path.abspath(__file__)))), "eval_got_nuscenes.py")
    try:
        tree = _ast.parse(open(path, encoding="utf-8").read())
        fn = next(n for n in tree.body
                  if isinstance(n, _ast.FunctionDef) and n.name == "get_args")
        # DEFAULT_PROMPT is the one non-stdlib name get_args closes over.
        from data.dataset_nuscenes import DEFAULT_PROMPT as _P
    except Exception:
        try:
            _P = ""
        except Exception:
            return None
    try:
        ns = {"argparse": _argparse, "DEFAULT_PROMPT": _P}
        exec(compile(_ast.Module(body=[fn], type_ignores=[]), path, "exec"), ns)
        get_args = ns["get_args"]
    except Exception:
        return None

    def make(planner):
        argv = ["eval", "--resume_path", "x", "--tokenizer_path", "x",
                "--records_json", "x", "--planner", planner]
        old = _sys.argv
        try:
            _sys.argv = argv
            return get_args()
        finally:
            _sys.argv = old
    try:
        make("pipeline")
    except Exception:
        return None
    return make


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    return ok


class CountingFn:
    """Mirrors eval_got_nuscenes._CountingFn so both arms are counted the same way."""

    def __init__(self, fn):
        self.fn = fn
        self.n = 0

    def __call__(self, *a, **kw):
        self.n += 1
        return self.fn(*a, **kw)


def run_incumbent(cfg, command):
    gen = CountingFn(_make_dummy_generate_fn())
    pipe = DriveGoTPipeline(cfg, gen, initial_image=None)
    merged, best = pipe.plan(command)
    return merged, gen.n, pipe.last_final_pool


def run_graph(cfg, command, keep_valid=False):
    gen = CountingFn(_make_dummy_generate_fn())
    goo = build_staged_goo(cfg, keep_valid=keep_valid)
    ctrl = Controller(goo, DrivingContext(gen, cfg, image=None, command=command)).run()
    return ctrl.plan(), gen.n, ctrl.final_pool(), ctrl


def main():
    # ── [A] equivalence with the incumbent ───────────────────────────────────
    print("\n[A] build_staged_goo() reproduces DriveGoTPipeline (no-op control)")
    for command in ("straight", "left", "right"):
        cfg = DriveGoTConfig(verbose=False)
        ref_traj, ref_calls, ref_pool = run_incumbent(cfg, command)
        got_traj, got_calls, got_pool, ctrl = run_graph(cfg, command)

        same_traj = (ref_traj is not None and got_traj is not None
                     and ref_traj.shape == got_traj.shape
                     and np.array_equal(ref_traj, got_traj))
        check(f"[{command}] trajectory is bit-identical", same_traj,
              "" if same_traj else f"ref={None if ref_traj is None else ref_traj.tolist()} "
                                   f"got={None if got_traj is None else got_traj.tolist()}")
        check(f"[{command}] generator call count matches", ref_calls == got_calls,
              f"ref={ref_calls} graph={got_calls}")
        check(f"[{command}] final pool size matches", len(ref_pool) == len(got_pool),
              f"ref={len(ref_pool)} graph={len(got_pool)}")
        ref_scores = [round(n.path_score, 9) for n in ref_pool]
        got_scores = [round(t.path_score, 9) for t in got_pool]
        check(f"[{command}] final pool path_scores match in order",
              ref_scores == got_scores, f"ref={ref_scores[:4]} graph={got_scores[:4]}")
        ref_cum = [n.cum_traj.round(6).tolist() for n in ref_pool]
        got_cum = [t.cum_traj.round(6).tolist() for t in got_pool]
        check(f"[{command}] final pool trajectories match in order", ref_cum == got_cum)

    cfg = DriveGoTConfig(verbose=False)
    traj, calls, pool, ctrl = run_graph(cfg, "left")
    check("shape is (time_horizon, 2)", traj is not None and traj.shape == (cfg.time_horizon, 2),
          str(None if traj is None else traj.shape))
    # k + beam*k + beam*k = 4 + 8 + 8 with the defaults; the number sec.1.5's cost
    # column and the 15x figure in the README are both built on.
    expect = cfg.k_candidates * (1 + 2 * cfg.beam_width)
    check(f"call count is the documented {expect}", calls == expect, f"got {calls}")

    # ── [B] the ToT/GoT discriminator ────────────────────────────────────────
    print("\n[B] topology")
    goo = build_staged_goo(cfg)
    check("staged graph is a CHAIN (i.e. it is ToT -- the control arm)", goo.is_chain())
    check("one root, one leaf", len(goo.roots) == 1 and len(goo.leaves) == 1,
          f"roots={len(goo.roots)} leaves={len(goo.leaves)}")
    n_expect = 2 + 4 * cfg.n_segments
    check(f"operation count is {n_expect}", len(goo.operations) == n_expect,
          f"got {len(goo.operations)}")

    # add_operation with two predecessors -> no longer a chain. This is the shape
    # Aggregate will have, and the assertion that will flip when it lands.
    g2 = GraphOfOperations()
    r = Root()
    g2.append_operation(r)
    a, b = GenerateSegment(0), GenerateSegment(0)
    for op in (a, b):
        op.add_predecessor(r)
        g2.add_operation(op)
    join = ScoreDriving(0)
    join.add_predecessor(a)
    join.add_predecessor(b)
    g2.add_operation(join)
    check("add_operation builds a DAG (two predecessors -> not a chain)",
          not g2.is_chain())
    check("the joined operation is the only leaf", g2.leaves == [join],
          str([type(o).__name__ for o in g2.leaves]))

    # ── [C] scheduling ───────────────────────────────────────────────────────
    print("\n[C] controller scheduling")
    goo3 = build_staged_goo(cfg)
    order = []
    for op in goo3.operations:
        orig = op._execute
        op._execute = (lambda ctx, _o=op, _f=orig, **kw: (order.append(_o.id), _f(ctx, **kw))[1])
    gen = CountingFn(_make_dummy_generate_fn())
    Controller(goo3, DrivingContext(gen, cfg, None, "left")).run()
    check("every operation executed exactly once",
          sorted(order) == sorted(op.id for op in goo3.operations)
          and len(order) == len(set(order)), f"{len(order)} executions")
    pred_ok = all(order.index(p.id) < order.index(op.id)
                  for op in goo3.operations for p in op.predecessors)
    check("no operation ran before a predecessor", pred_ok)

    reused = False
    try:
        Controller(goo3, DrivingContext(gen, cfg, None, "left")).run()
    except AssertionError:
        reused = True
    check("re-running an executed graph is refused (stale-state guard)", reused)

    # ── [D] Thought semantics ────────────────────────────────────────────────
    print("\n[D] Thought")
    t = Thought({"kind": "traj"})
    check("score setter sets the scored flag", (not t.scored) and
          (setattr(t, "score", 1.5) or t.scored) and t.score == 1.5)
    t2 = Thought({"kind": "traj"})
    check("valid setter sets the validated flag", (not t2.validated) and
          (setattr(t2, "valid", True) or t2.validated) and t2.valid)
    kid = t.child(kind="traj", cum_traj=np.zeros((2, 2)))
    check("child() records exactly one predecessor", kid.predecessors == [t])
    check("ids are unique and increasing", kid.id > t.id)

    # state builders: the invariants a dataclass would have enforced. These have
    # to actually reject things -- a validator that never fails is the sec.1.4
    # mistake again (a check that cannot fail proves nothing).
    ok_state = make_traj_state(np.zeros((2, 2)), np.zeros((2, 2)), (np.zeros(2), 0.0), 0)
    check("make_traj_state produces a valid state", validate_state(ok_state) is ok_state)
    for name, bad in (
            ("unknown kind", {"kind": "nonsense"}),
            ("missing required key", {"kind": "traj", "cum_traj": np.zeros((2, 2))}),
            ("wrong array shape", {"kind": "traj", "segment_local": np.zeros((2, 3)),
                                   "cum_traj": np.zeros((2, 2)),
                                   "end_pose": (np.zeros(2), 0.0), "depth": 0,
                                   "path_score": 0.0}),
    ):
        rejected = False
        try:
            validate_state(bad)
        except ValueError:
            rejected = True
        check(f"validate_state rejects: {name}", rejected)

    # ── [E] KeepBestN / KeepValid semantics ──────────────────────────────────
    print("\n[E] KeepBestN / KeepValid")
    last_keep = [op for op in goo.operations if isinstance(op, KeepBestN)]
    check("one KeepBestN per segment", len(last_keep) == cfg.n_segments)
    _, _, pool_full, ctrl2 = run_graph(cfg, "left")
    survivors = None
    for op in ctrl2.graph.operations:
        if isinstance(op, KeepBestN) and op.executed:
            survivors = op
    check("KeepBestN retains the discarded pool for minADE_C",
          len(survivors.all_thoughts) >= len(survivors.thoughts)
          and len(survivors.thoughts) == cfg.beam_width,
          f"all={len(survivors.all_thoughts)} kept={len(survivors.thoughts)}")
    check("retained pool is sorted by path_score, best first",
          all(a.path_score >= b.path_score
              for a, b in zip(survivors.all_thoughts, survivors.all_thoughts[1:])))

    kv_off = [op for op in ctrl2.graph.operations if isinstance(op, KeepValid)]
    check("KeepValid is a pass-through by default (veto still lives in the score)",
          all(op.n_dropped == 0 for op in kv_off))
    t_on, calls_on, _, _ = run_graph(cfg, "left", keep_valid=True)
    check("KeepValid(enabled=True) still returns a full-horizon plan and does not "
          "change the generator call count",
          t_on is not None and t_on.shape == (cfg.time_horizon, 2)
          and calls_on == expect, f"calls={calls_on}")

    # ★The check above cannot FAIL: the dummy generator only emits smooth
    # trajectories, so an always-true KeepValid would pass it. sec.1.4's lesson --
    # the crop sanity check that handed the same object in twice and could not
    # detect non-determinism -- is that a check which cannot fail proves nothing.
    # So drive the operation directly with something that must be rejected.
    def _kv_thoughts(thoughts, enabled=True):
        src = _Fixed(list(thoughts))
        op = KeepValid(enabled=enabled)
        g = GraphOfOperations()
        # append_operation ALREADY links src -> op. Calling add_predecessor as well
        # registers it twice, and get_previous_thoughts() then returns the pool
        # doubled -- which is how this helper first read "dropped=2" on a 2-element
        # pool. Use exactly one of the two linking routes, never both.
        g.append_operation(src)
        g.append_operation(op)
        Controller(g, DrivingContext(lambda *a: None, cfg, None, "left")).run()
        return op

    def _kv(trajs, enabled=True):
        return _kv_thoughts([Thought({"kind": "traj", "cum_traj": np.asarray(t, float),
                                      "path_score": 0.0}) for t in trajs],
                            enabled=enabled)

    good = [[4.0, 0.0], [8.0, 0.0], [12.0, 0.0], [16.0, 0.0], [20.0, 0.0], [24.0, 0.0]]
    # 30 m in one 0.5 s step: ~60 m/s forward, far past FEAS_V_MAX
    bad = [[30.0, 0.0], [60.0, 0.0], [90.0, 0.0], [120.0, 0.0], [150.0, 0.0], [180.0, 0.0]]
    op = _kv([good, bad])
    check("KeepValid(enabled=True) actually drops an infeasible candidate",
          op.n_dropped == 1 and len(op.get_thoughts()) == 1,
          f"dropped={op.n_dropped} kept={len(op.get_thoughts())}")
    op_off = _kv([good, bad], enabled=False)
    check("KeepValid(enabled=False) keeps that same candidate (arms really differ)",
          op_off.n_dropped == 0 and len(op_off.get_thoughts()) == 2)
    op_all_bad = _kv([bad, bad])
    check("all-infeasible pool falls back to the full pool instead of an empty beam",
          len(op_all_bad.get_thoughts()) == 2 and op_all_bad.n_dropped == 2,
          f"kept={len(op_all_bad.get_thoughts())} dropped={op_all_bad.n_dropped}")

    # ★REGRESSION: the pass-through must not DESTROY a verdict. In
    # build_improve_goo the chain is ValidateAndImprove -> Score ->
    # KeepValid(enabled=False), and a blanket `t.valid = True` here overwrote the
    # real verdict, so output_graph() serialised valid=true for a thought the
    # improve stats counted as n_still_invalid=1: two artefacts from one run that
    # cannot both be right. Pre-mark one thought the way ValidateAndImprove does
    # and require the pass-through to leave it alone.
    already = Thought({"kind": "traj", "cum_traj": np.asarray(bad, float),
                       "path_score": 0.0})
    already.valid = False                      # the verdict a validator reached
    virgin = Thought({"kind": "traj", "cum_traj": np.asarray(good, float),
                      "path_score": 0.0})
    op_pt = _kv_thoughts([already, virgin], enabled=False)
    check("★KeepValid(enabled=False) does NOT overwrite a verdict already reached "
          "(the dump and the improve stats must not disagree)",
          already.validated and already.valid is False,
          f"valid={already.valid}")
    check("KeepValid(enabled=False) still records a verdict for unvalidated thoughts",
          virgin.validated and virgin.valid is True and len(op_pt.get_thoughts()) == 2)

    # ★REGRESSION: filtering shrinks the pool minADE_C is computed over, and that
    # must never be silent. KeepValid runs BEFORE KeepBestN, so with the gate on,
    # Controller.final_pool() is the FEASIBLE SUBSET -- a smaller denominator that
    # would read as a better generator. Require (a) the removed thoughts to survive
    # on `dropped` for the trace dump, and (b) `n_dropped` to account for the
    # shrinkage exactly.
    op_drop = _kv([good, bad], enabled=True)
    check("★KeepValid keeps the candidates it removed (the pool stays recoverable "
          "offline; a shrunken minADE_C denominator must stay auditable)",
          len(op_drop.dropped) == op_drop.n_dropped == 1
          and len(op_drop.dropped) + len(op_drop.get_thoughts()) == 2,
          f"dropped={len(op_drop.dropped)} kept={len(op_drop.get_thoughts())}")
    check("the pass-through drops nothing and reports nothing dropped",
          op_off.dropped == [] and op_off.n_dropped == 0)
    check("the all-infeasible fallback removed nobody, so `dropped` is empty while "
          "n_dropped still says the gate misfired",
          op_all_bad.dropped == [] and op_all_bad.n_dropped == 2)

    # ── [F] trace dump ───────────────────────────────────────────────────────
    print("\n[F] output_graph()")
    dump = ctrl2.output_graph()
    round_trip = json.loads(json.dumps(dump))
    check("serialises to JSON", round_trip["is_chain"] is True)
    check("every thought carries its trajectory and provenance",
          len(dump["thoughts"]) > 0
          and all("cum_traj" in d and "predecessors" in d for d in dump["thoughts"]),
          f"{len(dump['thoughts'])} thoughts")
    deepest = [d for d in dump["thoughts"] if d["depth"] == cfg.n_segments - 1]
    check("the deepest thoughts are the final pool",
          len(deepest) == len(ctrl2.final_pool()),
          f"dump={len(deepest)} pool={len(ctrl2.final_pool())}")

    # ★REGRESSION: A DUMP WITH A DANGLING PREDECESSOR ID IS A BROKEN ARTEFACT.
    # output_graph() used to iterate only op.get_thoughts() (plus KeepBestN's
    # retained pool), so the Violation vertices ValidateAndImprove creates -- which
    # no operation returns -- were referenced as predecessors and never serialised.
    # Measured: the improve arm's dump carried 1 dangling id, which made
    # ValidateAndImprove's docstring claim that "output_graph() records why it
    # exists" false. Assert the closure property on EVERY graph kind, because the
    # next operation that mints an intermediate vertex will hit the same hole.
    def _dump_for(kind, **kw):
        gp_d = GraphPlanner(cfg, _make_dummy_generate_fn(), graph_kind=kind,
                            initial_image=None, **kw)
        gp_d.plan("left")
        # through JSON, because that is the artefact analysts actually read
        return json.loads(json.dumps(gp_d.last_controller.output_graph())), gp_d

    PROV_KEYS = {"improved_from", "repaired", "aggregated_from", "aggregate_method"}
    for kind in ("staged", "intent", "aggregate", "improve"):
        kw = {} if kind == "staged" else {"variants": 3}
        d, gp_d = _dump_for(kind, **kw)
        ids = {x["id"] for x in d["thoughts"]}
        dangling = sorted({p for x in d["thoughts"]
                           for p in x["predecessors"] if p not in ids})
        check(f"[{kind}] ★the dump has NO dangling predecessor id",
              not dangling, f"dangling={dangling} of {len(ids)} thoughts")
        check(f"[{kind}] every serialised thought carries the provenance keys",
              all(PROV_KEYS <= set(x) for x in d["thoughts"]),
              str(sorted(PROV_KEYS - set(d["thoughts"][0]))))
        pool_ids = {t.id for t in gp_d.last_final_pool}
        check(f"[{kind}] every thought the metrics are computed over is serialised",
              pool_ids <= ids, f"missing={sorted(pool_ids - ids)}")
    # the aggregate arm's whole point is an edge with several predecessors -- if it
    # is not in the artefact, the graph claim is prose again (§1.7).
    d_agg, _ = _dump_for("aggregate", variants=3)
    check("the aggregate dump names what each combination was built from",
          any(x["aggregated_from"] for x in d_agg["thoughts"])
          and all(x["aggregate_method"] == "median"
                  for x in d_agg["thoughts"] if x["aggregated_from"]),
          f"{sum(1 for x in d_agg['thoughts'] if x['aggregated_from'])} aggregates")

    # ── [G] GraphPlanner: the adapter eval_got_nuscenes.py actually calls ─────
    #
    # [A] proved the machinery. This proves the thing wired into the eval, which
    # is a different object: the eval never touches Controller directly, it calls
    # plan() / final_candidates() / final_candidate_scores() on the planner. An
    # adapter that reproduced the trajectory but mis-ordered final_candidates()
    # would leave minADE_C and the selection gap wrong while the headline looked
    # right -- and those two are the generator-vs-selector decomposition (sec.7.4).
    print("\n[G] GraphPlanner (the interface eval_got_nuscenes.py uses)")
    for command in ("straight", "left"):
        cfg = DriveGoTConfig(verbose=False)
        ref_gen = CountingFn(_make_dummy_generate_fn())
        ref = DriveGoTPipeline(cfg, ref_gen, initial_image=None)
        ref_traj, _ = ref.plan(command)
        ref_c, ref_sel = ref.final_candidates()
        ref_s = ref.final_candidate_scores()

        gp_gen = CountingFn(_make_dummy_generate_fn())
        gp = GraphPlanner(cfg, gp_gen, graph_kind="staged", initial_image=None)
        gp_traj, _ = gp.plan(command)
        gp_c, gp_sel = gp.final_candidates()
        gp_s = gp.final_candidate_scores()

        # ★`np.array_equal(None, None)` is True, so the original form of this
        # check passed when BOTH planners returned nothing -- the §1.4 mistake
        # (a check that cannot fail) in the one place it matters most, since
        # "both arms are broken" is exactly the state it must not certify.
        # Assert the trajectories EXIST and have the horizon's shape first.
        both_real = (ref_traj is not None and gp_traj is not None
                     and ref_traj.shape == (cfg.time_horizon, 2)
                     and gp_traj.shape == ref_traj.shape)
        check(f"[{command}] planner trajectory is bit-identical",
              both_real and np.array_equal(ref_traj, gp_traj),
              f"ref={None if ref_traj is None else ref_traj.shape} "
              f"graph={None if gp_traj is None else gp_traj.shape}")
        check(f"[{command}] planner call count matches", ref_gen.n == gp_gen.n,
              f"ref={ref_gen.n} graph={gp_gen.n}")
        check(f"[{command}] final_candidates() same trajectories, same order",
              [c.round(6).tolist() for c in ref_c] == [c.round(6).tolist() for c in gp_c],
              f"{len(ref_c)} vs {len(gp_c)}")
        check(f"[{command}] final_candidates() selects the same index",
              ref_sel == gp_sel, f"ref={ref_sel} graph={gp_sel}")
        check(f"[{command}] final_candidate_scores() has the incumbent's schema",
              set(ref_s) == set(gp_s), str(set(ref_s) ^ set(gp_s)))
        for key in ("kinematic", "command", "segment_score", "path_score"):
            check(f"[{command}] scores[{key}] match",
                  [round(v, 9) for v in ref_s[key]] == [round(v, 9) for v in gp_s[key]])
        check(f"[{command}] selected candidate is the one plan() returned",
              gp_sel is not None and np.array_equal(gp_c[gp_sel], gp_traj))

    # unsupported arms must RAISE, never run a quietly different arm
    print("\n[H] unsupported arms refuse rather than degrade")
    cfg_fuse = DriveGoTConfig(verbose=False, fuse="median")
    for name, kw, c in (
            ("--fuse", {}, cfg_fuse),
            ("--wm_path scoring", {"wm_score_fn": lambda *a: 0.0}, DriveGoTConfig(verbose=False)),
            ("--w_likelihood", {"lik_score_fn": lambda *a: 0.0}, DriveGoTConfig(verbose=False)),
            ("--final_weights", {}, DriveGoTConfig(verbose=False, final_weights=(1.0, 0.0))),
            ("Mode B", {"context_update_fn": lambda img, seg: img}, DriveGoTConfig(verbose=False)),
    ):
        raised = False
        try:
            GraphPlanner(c, lambda *a: None, graph_kind="staged",
                         initial_image=None, **kw)
        except NotImplementedError:
            raised = True
        check(f"{name} raises NotImplementedError", raised)

    # ── [I] intent nodes ─────────────────────────────────────────────────────
    print("\n[I] intent nodes")
    grid = make_intent_grid()
    check("default grid is speed-only and collapses the stop degeneracy",
          len(grid) == 3 and sum(1 for i in grid if i.speed_scale == 0.0) == 1
          and {i.curvature for i in grid} == {0.0},
          f"{len(grid)} intents: {[i.name for i in grid]}")
    check("conservative end is present (Step 2.9: mandatory)",
          any(i.speed_scale == 0.0 for i in grid)
          and any(abs(i.speed_scale - 0.5) < 1e-9 for i in grid))
    ks = sorted({i.curvature for i in make_intent_grid(curvatures=USABLE_CURVATURES)
                 if i.speed_scale > 0})
    check("curvature grids stay symmetric about 0 (the k=+0.005 bias is rejected)",
          ks == sorted(-k for k in ks), str(ks))
    degenerate = False
    try:
        Intent(0.0, 0.01)
    except ValueError:
        degenerate = True
    check("a curving stopped intent is refused", degenerate)

    # anchors: a 2 m first step must put v x1.0 and v x0.0 two metres apart, and
    # v x0.5 one metre from each -- i.e. the set clears MIN_SEPARATION_M. This is
    # the arithmetic §1.5 says has to hold or the model ignores the prefix.
    anchors = [anchor_waypoint(2.0, i) for i in grid]
    rep = separation_report(anchors)
    check(f"a 2 m ego step separates the default intents (>= {MIN_SEPARATION_M} m)",
          rep["separated"], f"min_pairwise={rep['min_pairwise']} max={rep['max_pairwise']}")

    # ★The finding that set the default. k = +-0.01 over a one-step anchor is
    # 0.02 m, and even over the WHOLE 3 s horizon only ~0.72 m -- still under the
    # floor. NAVSIM's values were fitted to a 4 s horizon at nuPlan speeds. Assert
    # it so nobody re-imports them from Step 2.10 without re-deriving this.
    navsim_k = make_intent_grid(speed_scales=(1.0,), curvatures=(-0.01, 0.0, 0.01))
    rep_k01 = separation_report([anchor_waypoint(2.0, i) for i in navsim_k])
    check("k=+-0.01 does NOT separate on a one-step anchor (why it is not default)",
          not rep_k01["separated"] and rep_k01["max_lat_gap"] < 0.1,
          f"max_lat_gap={rep_k01['max_lat_gap']} m")
    # k = +-0.05 with a longer committed anchor is the configuration that could
    # carry a curvature axis; check it actually does before anyone builds on it.
    usable_k = make_intent_grid(speed_scales=(1.0,), curvatures=USABLE_CURVATURES)
    rep_k05 = separation_report([anchor_waypoint(2.0, i, anchor_steps=4)
                                 for i in usable_k])
    check("k=+-0.05 over a 4-step anchor does separate laterally",
          rep_k05["max_lat_gap"] >= MIN_SEPARATION_M,
          f"max_lat_gap={rep_k05['max_lat_gap']} m")
    check("separation_report splits the two axes",
          {"max_lon_gap", "max_lat_gap"} <= set(rep), str(sorted(rep)))
    check("anchor_steps returns that many waypoints",
          anchor_waypoint(2.0, Intent(1.0, 0.0), anchor_steps=3).shape == (3, 2))
    # and the guard must FIRE when the ego is nearly stopped -- otherwise it is
    # another check that cannot fail.
    rep_slow = separation_report([anchor_waypoint(0.3, i) for i in grid])
    check("the guard fires when the ego is nearly stopped (no separation)",
          not rep_slow["separated"], f"min_pairwise={rep_slow['min_pairwise']}")

    check("zero curvature gives a straight anchor",
          np.allclose(anchor_waypoint(2.0, Intent(1.0, 0.0)), [[2.0, 0.0]]))
    left = anchor_waypoint(2.0, Intent(1.0, +0.01))
    right = anchor_waypoint(2.0, Intent(1.0, -0.01))
    check("symmetric curvatures give mirrored anchors",
          np.allclose(left[0, 0], right[0, 0]) and np.allclose(left[0, 1], -right[0, 1]),
          f"left={left.round(4).tolist()} right={right.round(4).tolist()}")
    check("a stopped intent anchors at the origin",
          np.allclose(anchor_waypoint(2.0, Intent(0.0, 0.0)), [[0.0, 0.0]]))

    # ── [J] the intent graph end to end ──────────────────────────────────────
    print("\n[J] build_intent_goo")
    cfg = DriveGoTConfig(verbose=False)
    gen = CountingFn(_make_dummy_generate_fn())
    ip = GraphPlanner(cfg, gen, graph_kind="intent", initial_image=None, variants=2)
    traj, sel = ip.plan("left")
    check("intent planner returns a full-horizon plan",
          traj is not None and traj.shape == (cfg.time_horizon, 2),
          str(None if traj is None else traj.shape))
    expect_calls = 1 + len(grid) * 2
    check(f"call count is 1 + n_intents*variants = {expect_calls}",
          gen.n == expect_calls, f"got {gen.n}")
    check("separation is reported on every plan",
          set(ip.last_separation) >= {"min_pairwise", "separated", "v0_step"},
          str(ip.last_separation))
    cands, sel_idx = ip.final_candidates()
    check("final_candidates() are full-horizon and the pick is among them",
          all(c.shape == (cfg.time_horizon, 2) for c in cands)
          and sel_idx is not None and np.array_equal(cands[sel_idx], traj),
          f"{len(cands)} candidates, sel={sel_idx}")
    check("every candidate carries the intent that produced it",
          all(t.state.get("intent") for t in ip.last_final_pool))
    n_distinct = len({t.state["intent"]["name"] for t in ip.last_final_pool})
    check("the pool spans more than one intent (stage 1 has options by "
          "construction, against the measured 1.52)", n_distinct > 1,
          f"{n_distinct} distinct intents in the final pool")

    goo_i = build_intent_goo(cfg, variants=2)
    check("the intent graph is still a chain (no Aggregate yet -- honest label)",
          goo_i.is_chain())

    # ── [K] Aggregate: the operation that makes this a graph ─────────────────
    print("\n[K] AggregateIntent")
    cfg = DriveGoTConfig(verbose=False)
    goo_a = build_aggregate_goo(cfg, variants=3)
    check("★the aggregate graph is NOT a chain (§1.7's claim, now a test)",
          not goo_a.is_chain())
    check("staged and intent graphs still ARE chains (the discriminator works "
          "both ways)",
          build_staged_goo(cfg).is_chain() and build_intent_goo(cfg).is_chain())
    fan_in = [op for op in goo_a.operations if len(op.predecessors) > 1]
    check("exactly one fan-in operation, and it is Score",
          len(fan_in) == 1 and isinstance(fan_in[0], ScoreDriving),
          str([type(o).__name__ for o in fan_in]))
    n_agg = sum(1 for op in goo_a.operations if isinstance(op, AggregateIntent))
    check("one AggregateIntent per intent", n_agg == len(make_intent_grid()),
          f"{n_agg} aggregates")

    gen = CountingFn(_make_dummy_generate_fn())
    ap = GraphPlanner(cfg, gen, graph_kind="aggregate", initial_image=None, variants=3)
    a_traj, _ = ap.plan("left")
    check("aggregate planner returns a full-horizon plan",
          a_traj is not None and a_traj.shape == (cfg.time_horizon, 2))
    check("call count is 1 + n_intents*variants",
          gen.n == 1 + len(make_intent_grid()) * 3, f"got {gen.n}")
    # ★REGRESSION: THE AGGREGATE ARM'S ONLY PER-RECORD EVIDENCE. `last_aggregate`
    # was computed and never read -- eval_got_nuscenes.py logged got_intent_* and
    # got_improve_* and nothing at all for aggregate. `fuse_trajectories` returns a
    # single input UNCHANGED, so an arm where every intent yields one surviving
    # realisation is a pure identity, and with no column for it that arm's L2 would
    # be reported as an Aggregate result. These four keys are exactly what the eval
    # writes per record (got_aggregate_n / _inputs / _identity /
    # _infeasible_dropped); keep the names in step with it.
    AGG_KEYS = {"n_aggregates", "n_inputs", "n_identity", "n_infeasible_dropped"}
    check("★aggregate stats are reported per record (the identity-arm detector)",
          AGG_KEYS <= set(ap.last_aggregate), str(sorted(AGG_KEYS - set(ap.last_aggregate))))
    check("aggregate stats are self-consistent (one entry per intent, n_identity "
          "counts the intents that combined nothing)",
          ap.last_aggregate["n_aggregates"] == len(make_intent_grid())
          == len(ap.last_aggregate["n_inputs"])
          and ap.last_aggregate["n_identity"]
              == sum(1 for n in ap.last_aggregate["n_inputs"] if n <= 1),
          str(ap.last_aggregate))
    # the improve arm fans out through the same AggregateIntent operations, so it
    # must carry the same evidence -- the eval logs these columns for both.
    ip_gen = CountingFn(_make_dummy_generate_fn())
    ip_imp = GraphPlanner(cfg, ip_gen, graph_kind="improve", initial_image=None,
                          variants=3)
    ip_imp.plan("left")
    check("the improve arm reports the same aggregate evidence (it fans out "
          "through the same operation)",
          AGG_KEYS <= set(ip_imp.last_aggregate), str(ip_imp.last_aggregate))
    ip_plain = GraphPlanner(cfg, _make_dummy_generate_fn(), graph_kind="intent",
                            initial_image=None)
    ip_plain.plan("left")
    check("an arm with no Aggregate operation reports no aggregate stats "
          "(the column stays empty rather than being fabricated)",
          ip_plain.last_aggregate == {}, str(ip_plain.last_aggregate))

    # ★the fused thought must record ALL its predecessors -- that is the graph
    # edge, and §1.7 exists because it was asserted in prose instead of built.
    fused = [t for t in ap.last_final_pool if t.state.get("aggregated_from")]
    check("fused thoughts exist and carry >1 predecessor",
          fused and any(len(t.predecessors) > 1 for t in fused),
          f"{len(fused)} fused, max preds="
          f"{max((len(t.predecessors) for t in fused), default=0)}")
    check("aggregated_from matches the recorded predecessors",
          all(sorted(t.state["aggregated_from"]) == sorted(p.id for p in t.predecessors)
              for t in fused))
    check("each fused thought keeps the intent it came from",
          all((t.state.get("intent") or {}).get("name") for t in fused))

    # ★the convexity precondition: §1.5 proved the average of FEASIBLE
    # trajectories is feasible, which says nothing about an average that
    # included an impossible one. Drive it directly rather than hoping the dummy
    # produced an infeasible variant.
    def _agg(trajs, keep_inputs=False, method="median", names=None):
        # `names` labels each input with the intent it belongs to, so the filter
        # at AggregateIntent._execute can actually be exercised (see below).
        names = list(names) if names is not None else ["t"] * len(trajs)
        src = _Fixed([Thought(make_traj_state(np.asarray(t, float), np.asarray(t, float),
                                              (np.zeros(2), 0.0), 0,
                                              intent={"speed_scale": 1.0,
                                                      "curvature": 0.0, "name": nm}))
                      for t, nm in zip(trajs, names)])
        op = AggregateIntent("t", method=method, keep_inputs=keep_inputs)
        g = GraphOfOperations()
        g.append_operation(src)
        g.append_operation(op)
        Controller(g, DrivingContext(lambda *a: None, cfg, None, "left")).run()
        return op

    good_a = [[4.0, 0.0], [8.0, 0.0], [12.0, 0.0], [16.0, 0.0], [20.0, 0.0], [24.0, 0.0]]
    good_b = [[4.0, 1.0], [8.0, 2.0], [12.0, 3.0], [16.0, 4.0], [20.0, 5.0], [24.0, 6.0]]
    bad = [[30.0, 0.0], [60.0, 0.0], [90.0, 0.0], [120.0, 0.0], [150.0, 0.0], [180.0, 0.0]]
    op = _agg([good_a, good_b, bad])
    check("infeasible inputs are dropped before combining (convexity precondition)",
          op.n_infeasible_dropped == 1 and op.n_inputs == 3,
          f"dropped={op.n_infeasible_dropped} of {op.n_inputs}")
    fused_traj = op.get_thoughts()[0].cum_traj
    check("the combination is feasible", bool(_feasible(fused_traj)))
    check("the combination lies between its inputs (median of two)",
          np.allclose(fused_traj, (np.array(good_a) + np.array(good_b)) / 2.0),
          str(fused_traj.round(3).tolist()[:2]))

    op1 = _agg([good_a])
    check("aggregating ONE realisation is the identity (the no-op control)",
          np.array_equal(op1.get_thoughts()[0].cum_traj, np.array(good_a))
          and op1.n_inputs == 1)
    op_all_bad = _agg([bad, bad])
    check("an all-infeasible intent is combined rather than vanishing",
          len(op_all_bad.get_thoughts()) == 1 and op_all_bad.n_infeasible_dropped == 2)
    op_keep = _agg([good_a, good_b], keep_inputs=True)
    check("keep_inputs=True preserves the realisations for pool comparability",
          len(op_keep.get_thoughts()) == 3, f"{len(op_keep.get_thoughts())} thoughts")
    # ★THE FILTER THAT MAKES sec.1.5's FAILURE UNREPRESENTABLE, ACTUALLY EXERCISED.
    # This check used to construct AggregateIntent("SOMETHING_ELSE"), never execute
    # it, and then assert the string literal "SOMETHING_ELSE" != "t" -- it could
    # not fail, and deleting the intent filter in operations.py would not have
    # turned it red. That matters more here than almost anywhere else: the whole
    # argument for retrying fusion is that averaging now happens WITHIN one
    # hypothesis (mixing plans is what drove infeasibility 2.5% -> 41.8%, turns
    # 3.7x worse). So feed a FOREIGN-intent realisation in and require it to be
    # excluded -- both from the count and from the arithmetic.
    foreign = [[2.0, -3.0], [4.0, -6.0], [6.0, -9.0],
               [8.0, -12.0], [10.0, -15.0], [12.0, -18.0]]
    mixed = _agg([good_a, good_b, foreign], names=["t", "t", "OTHER_INTENT"])
    mixed_traj = mixed.get_thoughts()[0].cum_traj
    own_only = (np.array(good_a) + np.array(good_b)) / 2.0
    all_three = np.median(np.stack([np.array(good_a), np.array(good_b),
                                    np.array(foreign)]), axis=0)
    check("★an aggregate only ever sees its OWN intent (a foreign realisation is "
          "excluded from the count AND from the combination)",
          mixed.n_inputs == 2 and np.allclose(mixed_traj, own_only)
          and not np.allclose(mixed_traj, all_three),
          f"n_inputs={mixed.n_inputs} fused={mixed_traj.round(2).tolist()[:2]}")
    check("the foreign input would have changed the result had it leaked in "
          "(so the check above can fail)",
          not np.allclose(own_only, all_three),
          f"own={own_only.round(2).tolist()[:1]} all={all_three.round(2).tolist()[:1]}")
    check("a thought with no intent at all is not swept into an aggregate",
          _agg([good_a, good_b], names=["t", None]).n_inputs == 1)
    bad_method = False
    try:
        AggregateIntent("t", method="geometric")
    except ValueError:
        bad_method = True
    check("an unknown combination method is refused at construction", bad_method)

    # -- [L] ValidateAndImprove -------------------------------------------
    print("\n[L] ValidateAndImprove")
    from got_drive.graph.violation import locate_violation, repair_prefix

    # ★THE INVARIANT. If locate_violation and _feasible ever disagree the improve
    # loop either spins (repairing what the veto still rejects) or declares
    # success on a trajectory the scorer will veto anyway. Random trajectories
    # across three scales so both the feasible and the wild regime are covered --
    # §11.7: run a null on the world where there IS something and the world where
    # there is not, or it proves nothing.
    rng = np.random.default_rng(0)
    disagree = 0
    n_infeas = 0
    for _ in range(4000):
        tr = rng.normal(0, rng.choice([1.0, 6.0, 25.0]), size=(6, 2)).cumsum(axis=0)
        v = locate_violation(tr)
        n_infeas += v is not None
        disagree += (v is None) != bool(_feasible(tr))
    check("★locate_violation agrees with _feasible on 4000 random trajectories",
          disagree == 0, f"{disagree} disagreements")
    check("the invariant test actually exercises both regimes",
          0 < n_infeas < 4000, f"{n_infeas}/4000 infeasible")

    bad_t = np.array([[30.0, 0.0], [60.0, 0.0], [90.0, 0.0],
                      [120.0, 0.0], [150.0, 0.0], [180.0, 0.0]])
    v = locate_violation(bad_t)
    check("a teleporting trajectory is located as a speed violation",
          v is not None and v["axis"] == "speed", str(v))
    pref = repair_prefix(bad_t, v)
    # The clamp is `limit * dt`, and `dt` is scoring_driving.DEFAULT_DT -- imported,
    # not the literal 0.5 this line used to carry. A hard-coded step time turns a
    # config change into a check that passes for the wrong reason (the sec.1.4
    # pattern: the number agreed, the quantity did not).
    check("repair_prefix clamps the violating step to its limit (v_max * DEFAULT_DT)",
          pref is not None
          and np.linalg.norm(pref[-1] - np.zeros(2)) <= v["limit"] * DEFAULT_DT + 1e-6,
          f"|step|={0.0 if pref is None else round(float(np.linalg.norm(pref[-1])), 4)} "
          f"limit*dt={round(v['limit'] * DEFAULT_DT, 4)}")
    check("the repaired prefix is itself feasible", bool(_feasible(pref)))
    check("a feasible trajectory locates no violation",
          locate_violation(np.array(good_a)) is None)

    def _vandi(trajs, num_tries=2, improve=True, then_keep_valid=None):
        """Drive ValidateAndImprove in isolation.

        `then_keep_valid` appends a KeepValid with that `enabled` value, which is
        the downstream shape build_improve_goo has (VandI -> Score -> KeepValid).
        Needed because the pass-through used to overwrite VandI's verdict, and
        that only shows up when the two operations run in the same graph.
        """
        intent = {"speed_scale": 1.0, "curvature": 0.0, "name": "t"}
        src = _Fixed([Thought(make_traj_state(np.asarray(t, float), np.asarray(t, float),
                                              (np.zeros(2), 0.0), 0, intent=intent))
                      for t in trajs])
        op = ValidateAndImprove(num_tries=num_tries, improve=improve)
        g = GraphOfOperations()
        g.append_operation(src)
        g.append_operation(op)
        if then_keep_valid is not None:
            g.append_operation(KeepValid(enabled=then_keep_valid))
        gen = CountingFn(_make_dummy_generate_fn())
        ctl = Controller(g, DrivingContext(gen, cfg, None, "left")).run()
        return op, gen, src, ctl

    # ★COST CONTROL: a pool that validates must cost ZERO calls and return the
    # SAME objects. An operation whose control arm changed cost is not a control.
    op, gen, src, _ = _vandi([good_a, good_b])
    check("a fully feasible pool costs zero generator calls", gen.n == 0, f"{gen.n} calls")
    check("feasible thoughts pass through as the same objects (not copies)",
          all(a is b for a, b in zip(op.get_thoughts(), src.get_thoughts())))
    check("validation marks them valid", all(t.validated and t.valid
                                             for t in op.get_thoughts()))

    op, gen, src, _ = _vandi([good_a, bad])
    check("an infeasible candidate triggers exactly the repairs it needs",
          op.stats["n_invalid"] == 1 and gen.n == op.stats["n_generate_calls"],
          str({k: v for k, v in op.stats.items() if k != "prefix_deltas"}))
    check("the pool never shrinks (a failed repair returns the original)",
          len(op.get_thoughts()) == 2)
    improved = [t for t in op.get_thoughts() if t.state.get("improved_from") is not None]
    check("★an improved thought has TWO predecessors (original + violation) -- "
          "the thought graph is a DAG even though the operation graph is a chain",
          improved and all(len(t.predecessors) == 2 for t in improved),
          f"{len(improved)} improved")
    check("the violation vertex records axis, index and the prefix delta",
          improved and any(p.state.get("kind") == "violation"
                           and {"axis", "index", "detail"} <= set(p.state)
                           for p in improved[0].predecessors))
    check("prefix deltas are recorded against §1.5's floor",
          "prefix_deltas" in op.stats and "n_repairs_below_floor" in op.stats
          and len(op.stats["prefix_deltas"]) >= 1, str(op.stats["prefix_deltas"]))

    op_v, gen_v, _, _ = _vandi([good_a, bad], improve=False)
    check("improve=False validates without generating (pure validator)",
          gen_v.n == 0 and op_v.stats["n_invalid"] == 1
          and len(op_v.get_thoughts()) == 2)

    op1, gen1, _, _ = _vandi([bad], num_tries=1)
    op3, gen3, _, _ = _vandi([bad], num_tries=3)
    # ★`gen1.n <= 1 and gen3.n <= 3` was satisfied by an operation that made ZERO
    # calls -- i.e. by Improve being silently disabled, the one outcome this check
    # exists to catch. `bad` never repairs under the dummy generator, so the loop
    # must actually spend its whole budget: exactly one attempt at num_tries=1, and
    # strictly more at 3. Bounded AND non-zero AND scaling.
    check("num_tries bounds the repair attempts -- and they are actually spent",
          gen1.n == 1 and 1 < gen3.n <= 3 and gen3.n == op3.stats["n_generate_calls"]
          and gen1.n == op1.stats["n_generate_calls"],
          f"tries=1 -> {gen1.n} calls, tries=3 -> {gen3.n} calls")

    # ★REGRESSION (the two-artefact disagreement). build_improve_goo runs
    # ValidateAndImprove -> Score -> KeepValid(enabled=False), and the pass-through
    # used to blanket-write `valid = True`, so output_graph() serialised valid=true
    # for a thought the stats counted as still invalid. Reproduce that exact chain
    # and require the dump to agree with the stats.
    op_kv, _, _, ctl_kv = _vandi([good_a, bad], improve=False, then_keep_valid=False)
    dump_kv = json.loads(json.dumps(ctl_kv.output_graph()))
    n_invalid_dump = sum(1 for d in dump_kv["thoughts"] if d["valid"] is False)
    check("★the trace dump's validity agrees with the improve stats (a downstream "
          "pass-through must not rewrite the verdict)",
          n_invalid_dump == op_kv.stats["n_still_invalid"] == 1,
          f"dump says {n_invalid_dump} invalid, stats say "
          f"{op_kv.stats['n_still_invalid']}")

    # and the violation vertices must be IN the dump, not just referenced by it.
    # This is the case the end-to-end dangling check in [F] cannot guarantee to
    # reach (the dummy generator may produce no infeasible candidate); here `bad`
    # guarantees a repair, so this check goes red if the closure walk is removed.
    _, _, _, ctl_imp = _vandi([good_a, bad])
    dump_imp = json.loads(json.dumps(ctl_imp.output_graph()))
    ids_imp = {d["id"] for d in dump_imp["thoughts"]}
    dangling_imp = sorted({p for d in dump_imp["thoughts"]
                           for p in d["predecessors"] if p not in ids_imp})
    viol = [d for d in dump_imp["thoughts"] if d["kind"] == "violation"]
    check("★a repair's violation vertex is SERIALISED, not just referenced "
          "(no dangling predecessor id in an improve dump)",
          not dangling_imp and len(viol) >= 1,
          f"dangling={dangling_imp} violations={len(viol)}")
    check("the serialised violation vertex explains itself (axis/index/detail)",
          viol and all({"axis", "index", "of_thought", "detail"} <= set(v)
                       and v["detail"].get("prefix_delta") is not None for v in viol),
          str(viol[0] if viol else None))
    check("an improved thought's dump records what it was improved from and what "
          "was repaired", any(d["improved_from"] is not None and d["repaired"]
                              for d in dump_imp["thoughts"]))

    goo_v = build_improve_goo(cfg, variants=3)
    check("the full-operator graph is not a chain", not goo_v.is_chain())
    vandi_ops = [o for o in goo_v.operations if isinstance(o, ValidateAndImprove)]
    check("ValidateAndImprove sits AFTER the aggregates (the combination is the "
          "one trajectory nothing has vetoed)",
          len(vandi_ops) == 1
          and all(isinstance(p, AggregateIntent) for p in vandi_ops[0].predecessors),
          str([type(p).__name__ for p in vandi_ops[0].predecessors]))
    goo_na = build_improve_goo(cfg, variants=3, aggregate=False)
    check("aggregate=False isolates Improve (chain, no AggregateIntent)",
          goo_na.is_chain()
          and not any(isinstance(o, AggregateIntent) for o in goo_na.operations))

    # -- [M] the --planner -> arm wiring eval_got_nuscenes.py runs on ---------
    #
    # ★THE MOST EXPENSIVE BUG THIS PACKAGE HAS HAD, AND WHY IT IS TESTED HERE.
    # The eval built its planner kwargs inline, behind
    # `if args.planner in ("graph", "intent"):` -- which wrapped the aggregate and
    # improve branches too. Both arms therefore received an EMPTY kwarg dict,
    # GraphPlanner's `graph_kind` defaulted to "staged", and they silently ran the
    # INCUMBENT CONTROL ARM while summary.json recorded "planner": "aggregate".
    # Measured symptom: 20 forward calls/record (the incumbent's k*(1+2*beam))
    # where the intent-based arms make 1 + n_intents*variants (+ repairs).
    # ⚠️config-dependent, so the check below compares against the control arm's
    # formula rather than a literal. A wrong number would have been caught;
    # the control arm's number wearing another arm's name would not.
    #
    # The mapping now lives in got_drive/graph/planner.py precisely so it can be
    # asserted without importing the eval (which needs torch). Two properties are
    # checked for EVERY --planner value: the kwargs land on the intended
    # graph_kind, and every flag that arm owns actually reaches the planner.
    print("\n[M] --planner -> graph_kind wiring (the eval's arm selection)")
    from got_drive.graph.planner import (GRAPH_PLANNERS, PLANNER_GRAPH_KIND,
                                         planner_kwargs)

    # ★DRIVEN BY THE EVAL'S REAL PARSER, NOT A HAND-COPIED COPY OF IT.
    # An earlier version rebuilt argparse's defaults into a SimpleNamespace. That
    # is the sec.1.4 pattern one level up: renaming or retyping a flag in
    # eval_got_nuscenes.py would AttributeError on record 1 of a real run while
    # this section stayed green, because the copy still carried the old name.
    # eval_got_nuscenes.py cannot be IMPORTED here (it needs torch), so `get_args`
    # is lifted out of its AST and executed alone -- argparse is stdlib, so the
    # function runs without any of the module's heavy imports.
    _EVAL_ARGS = _lift_eval_get_args()
    check("[M] is driven by eval_got_nuscenes.py's REAL parser",
          _EVAL_ARGS is not None,
          "could not lift get_args() -- a flag rename would no longer be caught")

    def _args(planner, **over):
        """The eval's own defaults, with `planner` and any overrides applied.

        Overrides are deliberately non-default: a flag swallowed on the way to the
        planner is only visible when its value differs from the default.
        """
        ns = _EVAL_ARGS(planner)
        for k, v in over.items():
            if not hasattr(ns, k):
                raise AttributeError(
                    f"eval_got_nuscenes.py has no --{k} -- this check is testing a "
                    f"flag that no longer exists (renamed?)")
            setattr(ns, k, v)
        return ns

    check("the eval's planner choices and the graph_kind map agree",
          set(PLANNER_GRAPH_KIND) == set(GRAPH_PLANNERS)
          and "pipeline" not in GRAPH_PLANNERS,
          str(sorted(PLANNER_GRAPH_KIND)))
    check("--planner pipeline gets NO graph kwargs (they would reach "
          "DriveGoTPipeline, which has no such parameters)",
          planner_kwargs(_args("pipeline")) == {})

    # the operations each arm's name promises, so "it built the right graph" is
    # checked against the graph, not against the string we just passed in.
    EXPECT_OPS = {
        "graph":     (GenerateSegment, (GenerateIntents, AggregateIntent,
                                        ValidateAndImprove)),
        "intent":    (RealizeIntent, (AggregateIntent, ValidateAndImprove)),
        "aggregate": (AggregateIntent, (ValidateAndImprove,)),
        # ⚠️`()` was used here once, and `isinstance(o, ())` is ALWAYS False --
        # the "none of" half was vacuous, i.e. exactly the sec.1.4 failure this
        # file exists to prevent. GenerateSegment is a real exclusion: the
        # improve graph branches on intents, never on time segments.
        "improve":   (ValidateAndImprove, (GenerateSegment,)),
    }
    for planner in GRAPH_PLANNERS:
        kw = planner_kwargs(_args(planner), intents=make_intent_grid())
        check(f"[--planner {planner}] kwargs name graph_kind="
              f"{PLANNER_GRAPH_KIND[planner]!r}",
              kw.get("graph_kind") == PLANNER_GRAPH_KIND[planner],
              f"got {kw.get('graph_kind')!r} from {sorted(kw)}")
        gen_m = CountingFn(_make_dummy_generate_fn())
        gp_m = GraphPlanner(cfg, gen_m, initial_image=None, **kw)
        gp_m.plan("left")
        ops = gp_m.last_controller.graph.operations
        must, must_not = EXPECT_OPS[planner]
        check(f"[--planner {planner}] the graph it BUILT contains "
              f"{must.__name__} and none of "
              f"{[c.__name__ for c in must_not] or '[]'}",
              any(isinstance(o, must) for o in ops)
              and not any(isinstance(o, must_not) for o in ops),
              str(sorted({type(o).__name__ for o in ops})))
        # the observable symptom of the bug: a non-control arm costing exactly the
        # incumbent's 20 calls means it ran the incumbent.
        if planner == "graph":
            check("[--planner graph] costs the incumbent's call count (it IS the "
                  f"control arm)", gen_m.n == expect, f"{gen_m.n} calls")
        else:
            check(f"[--planner {planner}] does NOT cost the control arm's "
                  f"{expect} calls (that is how the bug showed up)",
                  gen_m.n != expect, f"{gen_m.n} calls")

    # ★every flag an arm owns must REACH the planner. These six were swallowed
    # together with graph_kind; a per-flag assertion is what makes the next added
    # flag's absence visible.
    kw_i = planner_kwargs(_args("intent", graph_keep_valid=True, intent_variants=4,
                                intent_anchor_steps=2), intents=make_intent_grid())
    check("--graph_keep_valid / --intent_variants / --intent_anchor_steps reach "
          "the intent arm",
          kw_i["keep_valid"] is True and kw_i["variants"] == 4
          and kw_i["anchor_steps"] == 2 and kw_i["intents"] is not None,
          str({k: v for k, v in kw_i.items() if k != "intents"}))
    kw_a = planner_kwargs(_args("aggregate", graph_keep_valid=True,
                                aggregate_method="mean", aggregate_keep_inputs=True,
                                intent_variants=3))
    check("--aggregate_method / --aggregate_keep_inputs / --graph_keep_valid reach "
          "the aggregate arm (all four were dropped on the floor)",
          kw_a["graph_kind"] == "aggregate" and kw_a["aggregate_method"] == "mean"
          and kw_a["aggregate_keep_inputs"] is True and kw_a["keep_valid"] is True
          and kw_a["variants"] == 3, str(kw_a))
    kw_v = planner_kwargs(_args("improve", improve_tries=5, improve_no_aggregate=True,
                                aggregate_method="mean"))
    check("--improve_tries / --improve_no_aggregate reach the improve arm",
          kw_v["graph_kind"] == "improve" and kw_v["num_tries"] == 5
          and kw_v["improve_aggregate"] is False
          and kw_v["aggregate_method"] == "mean", str(kw_v))
    # and they must survive into the planner's own state, not just the dict
    gp_v = GraphPlanner(cfg, _make_dummy_generate_fn(), initial_image=None, **kw_v)
    check("the flags survive into the planner (improve arm, no aggregate fan-out)",
          gp_v.graph_kind == "improve" and gp_v.num_tries == 5
          and gp_v.improve_aggregate is False
          and not any(isinstance(o, AggregateIntent) for o in gp_v._build().operations))

    # ★`graph_kind` MUST NOT DEFAULT. A default of "staged" is what turned the
    # wrong guard above into a plausible table row instead of a TypeError on
    # record 1 -- and it defaulted to the CONTROL arm, the one whose numbers
    # already exist to be mistaken for.
    no_kind = False
    try:
        GraphPlanner(cfg, _make_dummy_generate_fn(), initial_image=None)
    except TypeError:
        no_kind = True
    check("★GraphPlanner refuses to be built without an explicit graph_kind "
          "(no silent fallback to the control arm)", no_kind)

    # -- anchor_steps upper bound (§1.5's channel needs something left to generate)
    bad_anchor = False
    try:
        GraphPlanner(cfg, _make_dummy_generate_fn(), graph_kind="intent",
                     initial_image=None, anchor_steps=cfg.time_horizon)
    except ValueError as e:
        bad_anchor = "time_horizon" in str(e)
    check("★anchor_steps >= time_horizon is refused at construction, naming the "
          "cause (it used to make every record silently malformed)", bad_anchor)
    op_ri = RealizeIntent(anchor_steps=cfg.time_horizon + 3)
    g_ri = GraphOfOperations()
    g_ri.append_operation(Root())
    g_ri.append_operation(GenerateIntents())
    g_ri.append_operation(op_ri)
    raised_ri = False
    try:
        Controller(g_ri, DrivingContext(_make_dummy_generate_fn(), cfg, None,
                                        "left")).run()
    except ValueError as e:
        raised_ri = "anchor_steps" in str(e)
    check("RealizeIntent enforces the same bound at execute time (the only place "
          "cfg.time_horizon is known for certain)", raised_ri)
    ok_anchor = GraphPlanner(cfg, _make_dummy_generate_fn(), graph_kind="intent",
                             initial_image=None, anchor_steps=cfg.time_horizon - 1)
    check("the largest legal anchor still plans (the guard is not off by one)",
          ok_anchor.plan("left")[0] is not None)

    # -- [N] the reporting halves the verifier found unguarded ----------------
    #
    # ★WHY THIS SECTION EXISTS. An independent verify pass signed off seven fixes
    # and refused two: `KeepValid`'s shrinkage report and the aggregate evidence
    # were correct in behaviour but rested on code no check touched. Hard-coding
    # `n_dropped: 0` -- so a run reports no shrinkage while the pool really shrank
    # 8 -> 6 -- passed the whole suite; so did DELETING the eval's csv block.
    # That is the sec.1.4 pattern reappearing inside the fix written to prevent a
    # silently changed minADE_C denominator. Both are now executable.
    print("\n[N] shrinkage + evidence reporting (previously unguarded)")

    def _pool_and_report(keep_valid):
        gen_n = CountingFn(_make_bumpy_generate_fn())
        gp = GraphPlanner(cfg, gen_n, initial_image=None, graph_kind="intent",
                          intents=make_intent_grid(), variants=3,
                          keep_valid=keep_valid)
        gp.plan("left")
        return gp

    off, on = _pool_and_report(False), _pool_and_report(True)
    shrink = len(off.final_pool_size()) - len(on.final_pool_size())         if hasattr(off, "final_pool_size") else         len(off.last_final_pool) - len(on.last_final_pool)
    reported = (on.last_keep_valid or {}).get("n_dropped")
    check("KeepValid's reported n_dropped EQUALS the pool it actually removed "
          "(hard-coding it to 0 must go red)",
          reported == shrink and shrink > 0,
          f"pool {len(off.last_final_pool)} -> {len(on.last_final_pool)} "
          f"(shrank {shrink}), reported n_dropped={reported}")
    check("the shrinkage is recoverable from the trace, not just counted",
          all(any(t["id"] == d.id for t in on.last_controller.output_graph()["thoughts"])
              for op in on.last_controller.graph.operations
              if isinstance(op, KeepValid) for d in getattr(op, "dropped", [])))
    check("keep_valid OFF reports zero shrinkage (the check is not always-true)",
          (off.last_keep_valid or {}).get("n_dropped") == 0,
          str(off.last_keep_valid))

    # ---- the eval's csv columns, coupled to the planner by something executable
    #
    # The only link between GraphPlanner.last_* and the csv used to be a comment.
    # Deleting the eval's whole row block left the suite green, so findings 3 and 4
    # were "correct today, unprotected tomorrow". This reads the column names out
    # of eval_got_nuscenes.py's AST (it cannot be imported -- torch) and asserts
    # both directions: the eval still writes every column, and the planner still
    # produces the value behind each one.
    import ast as _ast, os as _os, re as _re
    _eval_src = open(_os.path.join(_os.path.dirname(_os.path.dirname(
        _os.path.dirname(_os.path.abspath(__file__)))), "eval_got_nuscenes.py"),
        encoding="utf-8").read()
    written = {n.value for n in _ast.walk(_ast.parse(_eval_src))
               if isinstance(n, _ast.Constant) and isinstance(n.value, str)
               and _re.match(r"^got_(intent|improve|aggregate|keep_valid)_", n.value)}
    REQUIRED = {
        "got_intent_sep_min", "got_intent_sep_max", "got_intent_separated",
        "got_improve_invalid", "got_improve_repaired", "got_improve_calls",
        "got_improve_below_floor",
        "got_aggregate_n", "got_aggregate_inputs", "got_aggregate_identity",
        "got_aggregate_infeasible_dropped",
        "got_keep_valid_dropped",
    }
    check("eval_got_nuscenes.py still writes every graph-arm csv column",
          REQUIRED <= written, f"missing {sorted(REQUIRED - written)}")
    gen_e = CountingFn(_make_bumpy_generate_fn())
    gp_e = GraphPlanner(cfg, gen_e, initial_image=None, graph_kind="improve",
                        intents=make_intent_grid(), variants=3, keep_valid=True)
    gp_e.plan("left")
    supplied = {
        "got_intent_sep_min": "min_pairwise", "got_intent_sep_max": "max_pairwise",
        "got_intent_separated": "separated",
    }
    check("the planner supplies the value behind every intent column",
          all(k in (gp_e.last_separation or {}) for k in supplied.values()),
          str(sorted(gp_e.last_separation or {})))
    check("...and behind every improve and aggregate column",
          {"n_invalid", "n_repaired", "n_generate_calls", "n_repairs_below_floor"}
          <= set(gp_e.last_improve or {})
          and {"n_aggregates", "n_inputs", "n_identity", "n_infeasible_dropped"}
          <= set(gp_e.last_aggregate or {})
          and "n_dropped" in (gp_e.last_keep_valid or {}),
          f"improve={sorted(gp_e.last_improve or {})} "
          f"aggregate={sorted(gp_e.last_aggregate or {})}")


    n_fail = sum(1 for _, ok in RESULTS if not ok)
    print(f"\n{'=' * 66}\n{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed")
    if n_fail:
        print("SELF-TEST: FAIL -- the no-op control does not reproduce the incumbent, "
              "so do not compare anything from this package with sec.1.")
    else:
        print("SELF-TEST: PASS")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
