"""
Preprocess NAVSIM into WORLD-MODEL-EVAL frame records.

This is the NAVSIM twin of preprocess_nuscenes_wm_eval.py, and it exists for
the same reason: the planning records (preprocess_navsim.py) carry the CURRENT
CAM_F0 frame plus the future WAYPOINTS, but the offline WM eval also needs the
REAL FUTURE FRAMES to compare the WM's predictions against.

WHY navtest AND NOT navtrain (handoff Step 2.15, measured -- do not re-derive)

    split      current frame   +1(0.5s)   +4(2.0s)   +8(4.0s)
    navtrain       23.4%         91.9%      61.1%      44.9%    PARTIAL
    navtest       100.0%        100.0%     100.0%     100.0%    PASS

navtrain ships sensor blobs in CONTIGUOUS BLOCKS: if a scenario's current frame
is on disk its neighbour almost certainly is too (91.9%), and coverage decays
as you walk off the end of the block. Requiring the full 4 s there would keep
45% -- plenty in absolute terms (~68k) but a subpopulation defined by disk
layout, which would then have to be declared and audited for bias.

navtest needs none of that -- measured on the FULL enumeration, not a sample:
69,405 / 69,405 records, zero missing frames across all five boundaries. And it
is where PDMS is computed, so the plan's PDMS and its WM delta land on the SAME
records: rho(delta, PDMS) becomes askable with no matching loss.

⚠️ TWO COUNTS, DO NOT CONFUSE THEM (this cost a build): navtest ENUMERATES
69,405 scenarios and PDMS SCORES 12,146 of them (an exact subset, Step 2.13).
This builder walks the enumeration; --scored_tokens checks that every scored
scenario got a record, which is the join the correlation actually needs. That correlation is the one question nuScenes
could not ask -- there the answer was rho(delta, L2) = -0.0091, and the sec.1.1
mechanism ("plausibility cannot predict which plausible future was realised")
applies to L2 but NOT to PDMS.

TEACHER FORCING, so the boundaries are what they are

The eval is teacher-forced per segment: each segment is fed the REAL frame at
its start (errors do not compound) and its prediction is scored against the
REAL frame at its end. With 8 waypoints at 0.5 s and segment_len=2, boundaries
fall at offsets 0, 2, 4, 6, 8 == 0.0, 1.0, 2.0, 3.0, 4.0 s.

    frames[s]    teacher-forcing anchor at the start of segment s
    frames[s+1]  the real target segment s is scored against

Emitted record (keyed so the eval can join it to the planning record):

    {
        "sample_token": str,
        "scene": str,                       # log name
        "frames": [p@+0, p@+2, p@+4, p@+6, p@+8],
        "frame_offsets": [0, 2, 4, 6, 8],
    }

No geometry or actions here, on purpose: plan and GT actions are both derived
in the eval from their (N,2) trajectories through one shared re-basing helper,
so the two cannot drift apart in convention.

Usage (navsim_wvg -- the devkit loader is required)
---------------------------------------------------
    python -m data.preprocess_navsim_wm_eval \
        --filter navtest --out_dir ./data/navsim_wm_records \
        --n_segments 4 --segment_len 2
"""

import argparse
import json
import os
import sys

from data.preprocess_navsim import (RAW_HOLDERS, describe_frame, locate_configs,
                                    parse_scene_filter_yaml, parse_split_yaml)

EXPECTED_DT = 0.5

# ENUMERATED vs SCORED -- two different numbers for the same split, and the
# first version of this guard confused them (session 16).
#
#   navtest ENUMERATES 69,405 scenarios
#   PDMS    SCORES     12,146 of them, an exact subset (handoff Step 2.13)
#
# This builder walks the enumeration, so the count guard belongs on 69,405.
# The number that actually matters for the WM x PDMS join is the other one, and
# it gets its own check (--scored_tokens): every scored token must have a
# record, or the correlation is computed on a silently reduced population --
# the Step 1-2 failure (a borrowed cache of 4,193 = 34% of the benchmark that
# produced numbers CLOSER to the paper than the full set).
EXPECTED_RECORDS = {"navtest": 69405}
SCORED_SUBSET = {"navtest": 12146}
WINDOW_FRAC = (0.90, 1.00)


def frame_image_at(loader, token, sensor_root, index):
    """CAM_F0 path for frames[index] of this scenario, or None.

    Same accessor as preprocess_navsim.raw_image_path, generalised off the
    current frame. The schema was discovered by shape, not guessed, and is
    recorded in handoff Step 2.15(e):

        loader.scene_frames_dicts[token][3 + h]['cams']['CAM_F0']['data_path']
    """
    for name in RAW_HOLDERS:
        holder = getattr(loader, name, None)
        if not isinstance(holder, dict):
            continue
        frames = holder.get(token)
        if not frames or index < 0 or index >= len(frames):
            continue
        try:
            value = frames[index]["cams"]["CAM_F0"]["data_path"]
        except (KeyError, TypeError, IndexError):
            return None
        if value:
            value = str(value)
            if not os.path.isabs(value) and sensor_root:
                value = os.path.join(sensor_root, value)
            return value
    return None


def boundary_offsets(n_segments, segment_len):
    return [s * segment_len for s in range(n_segments + 1)]


def check_spacing(loader, tokens, out=print):
    """Median frame spacing in seconds, measured rather than assumed.

    The 4 s horizon is 8 steps x 0.5 s. If dt were anything else the WM segment
    boundaries would not line up with the waypoints the model was trained on,
    and nothing downstream would notice.
    """
    gaps = []
    for token in tokens[:20]:
        for name in RAW_HOLDERS:
            holder = getattr(loader, name, None)
            if not isinstance(holder, dict) or token not in holder:
                continue
            stamps = [f.get("timestamp") for f in holder[token]
                      if isinstance(f, dict) and isinstance(f.get("timestamp"), (int, float))]
            gaps.extend(b - a for a, b in zip(stamps, stamps[1:]) if b > a)
            break
    if not gaps:
        out("    frame spacing: UNVERIFIED (no timestamps on the raw frames)")
        return None
    gaps.sort()
    dt = gaps[len(gaps) // 2] * 1e-6          # devkit stores microseconds
    out("    frame spacing: %.3f s (n=%d)" % (dt, len(gaps)))
    if abs(dt - EXPECTED_DT) > 0.05:
        raise SystemExit(
            "[fatal] frame spacing is %.3f s, not %.3f. The boundary offsets "
            "would not be the horizon they claim to be." % (dt, EXPECTED_DT))
    return dt


def build_records(loader, sensor_root, n_history, offsets, out=print):
    """One record per scenario that has EVERY boundary frame on disk.

    All-or-nothing on purpose: a record that is missing one boundary would make
    the eval open a file that is not there, halfway through a run. Skipping it
    here costs one scenario; discovering it there costs the run.
    """
    tokens = list(loader.tokens)
    base = max(int(n_history) - 1, 0)
    records = []
    dir_names = {}
    skipped = {"unresolved": 0, "missing_frame": 0}
    per_offset_missing = {h: 0 for h in offsets}

    def on_disk(path):
        directory = os.path.dirname(path)
        names = dir_names.get(directory)
        if names is None:
            try:
                names = set(os.listdir(directory))
            except OSError:
                names = set()
            dir_names[directory] = names
        return os.path.basename(path) in names

    for index, token in enumerate(tokens):
        paths, ok = [], True
        for h in offsets:
            path = frame_image_at(loader, token, sensor_root, base + h)
            if not path:
                skipped["unresolved"] += 1
                ok = False
                break
            if not on_disk(path):
                per_offset_missing[h] += 1
                skipped["missing_frame"] += 1
                ok = False
                break
            paths.append(path)
        if not ok:
            continue
        records.append({
            "sample_token": token,
            "scene": _log_name(loader, token),
            "frames": paths,
            "frame_offsets": list(offsets),
        })
        if (index + 1) % 5000 == 0:
            out("    %d / %d scenarios" % (index + 1, len(tokens)))

    out("    enumerated %d -> %d complete records "
        "(unresolved %d, missing frame %d)"
        % (len(tokens), len(records), skipped["unresolved"], skipped["missing_frame"]))
    if any(per_offset_missing.values()):
        out("    missing by offset: %s"
            % ", ".join("+%d:%d" % (h, n) for h, n in per_offset_missing.items() if n))
    return records, {"skipped": skipped, "per_offset_missing": per_offset_missing,
                     "enumerated": len(tokens)}


def _log_name(loader, token):
    for name in RAW_HOLDERS:
        holder = getattr(loader, name, None)
        if isinstance(holder, dict) and token in holder:
            frame = holder[token][0]
            if isinstance(frame, dict):
                for key in ("log_name", "log", "scene_name"):
                    if frame.get(key):
                        return str(frame[key])
    return "?"


def read_scored_tokens(path, out=print):
    """Token list for the scenarios PDMS actually scores.

    Accepts either a plain token-per-line file or a PDMS result CSV (the column
    named 'token'). The CSV is preferred because it is the authority: it IS the
    set the score was computed over, so it cannot drift from the experiment the
    correlation will be joined to.
    """
    values = []
    with open(path) as handle:
        first = handle.readline().strip()
        if "," in first:                       # csv
            header = [c.strip().strip('"') for c in first.split(",")]
            try:
                column = header.index("token")
            except ValueError:
                raise SystemExit("[fatal] %s looks like a CSV but has no 'token' "
                                 "column (header: %s)" % (path, header[:8]))
            for line in handle:
                parts = line.split(",")
                if len(parts) > column:
                    values.append(parts[column].strip().strip('"'))
        else:
            values = [line.strip() for line in [first] + handle.readlines()
                      if line.strip()]

    tokens, dropped = _keep_real_tokens(values)
    out("    scored tokens read: %d from %s" % (len(tokens), path))
    if dropped:
        out("    dropped %d non-token row(s): %s"
            % (len(dropped), ", ".join(repr(d) for d in dropped[:3])))
    return tokens


def _keep_real_tokens(values):
    """Drop the aggregate row PDMS CSVs end with.

    run_pdm_score writes a summary line whose `token` cell is the literal
    'average', and reading it as a scenario made the join look one token short
    (session 16). Rather than blacklisting that one word -- the next devkit
    version may call it something else -- keep the values whose length matches
    the modal one, which is what a fixed-width token id looks like, and REPORT
    what was dropped so a real mismatch cannot hide behind the same rule.
    """
    if not values:
        return set(), []
    lengths = {}
    for value in values:
        lengths[len(value)] = lengths.get(len(value), 0) + 1
    modal = max(lengths, key=lambda k: lengths[k])
    keep = {v for v in values if len(v) == modal}
    dropped = [v for v in values if len(v) != modal]
    return keep, dropped


def guard_scored_coverage(records, scored, split, forced, out=print):
    """Every scored token must have a record. This is the join, checked early.

    Reported as a MISS count rather than a ratio: one missing token is one
    scenario the correlation silently drops, and a ratio of 99.99% reads as
    success.
    """
    have = {r["sample_token"] for r in records}
    missing = scored - have
    expected = SCORED_SUBSET.get(split)
    if expected is not None and len(scored) != expected:
        out("    ⚠ scored list has %d tokens, preregistered %d for '%s' -- the "
            "list itself may be a partial run" % (len(scored), expected, split))
    out("    scored tokens with a record: %d / %d (missing %d)"
        % (len(scored) - len(missing), len(scored), len(missing)))
    if not missing:
        out("    join is lossless")
        return
    sample = list(missing)[:3]
    message = ("[fatal] %d scored tokens have no WM record (e.g. %s).\n"
               "  rho(delta, PDMS) would then be computed on a reduced "
               "population; --force_count overrides." % (len(missing), sample))
    if forced:
        out(message.replace("[fatal]", "[forced]"))
        return
    raise SystemExit(message)


def guard_count(records, split, enumerated, forced, out=print):
    expected = EXPECTED_RECORDS.get(split)
    if expected is None:
        out("    no preregistered count for '%s' -- reporting only" % split)
        return
    lo, hi = int(expected * WINDOW_FRAC[0]), int(expected * WINDOW_FRAC[1])
    out("    preregistered: %s = %d, window %d-%d" % (split, expected, lo, hi))
    if lo <= len(records) <= hi:
        out("    count ok")
        return
    message = ("[fatal] %d records is outside %d-%d for '%s' (enumerated %d).\n"
               "  A plausible-looking number from a PARTIAL split is this "
               "project's most expensive failure mode (handoff sec.9). Explain "
               "the gap before building on it; --force_count overrides."
               % (len(records), lo, hi, split, enumerated))
    if forced:
        out(message.replace("[fatal]", "[forced]"))
        return
    raise SystemExit(message)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--devkit_root", default=os.environ.get("NAVSIM_DEVKIT_ROOT"))
    ap.add_argument("--data_root", default=os.environ.get("OPENSCENE_DATA_ROOT"))
    ap.add_argument("--filter", default="navtest",
                    help="navtest by default: Step 2.15 measured 100%% frame "
                         "coverage there and it is where PDMS is computed")
    ap.add_argument("--out_dir", default="./data/navsim_wm_records")
    ap.add_argument("--n_segments", type=int, default=4)
    ap.add_argument("--segment_len", type=int, default=2,
                    help="waypoints per segment (2 = 1.0 s at dt 0.5 s)")
    ap.add_argument("--scored_tokens", default=None,
                    help="token-per-line file or a PDMS result CSV. Checks that "
                         "every SCORED scenario has a record -- the join the "
                         "rho(delta, PDMS) experiment depends on")
    ap.add_argument("--only_scored", action="store_true",
                    help="emit only the scored tokens (needs --scored_tokens). "
                         "Off by default: the extra records cost little and "
                         "rebuilding after a rescore costs an hour")
    ap.add_argument("--force_count", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(_selftest())
    if not args.devkit_root or not args.data_root:
        sys.exit("NAVSIM_DEVKIT_ROOT / OPENSCENE_DATA_ROOT unset -- activate navsim_wvg")

    offsets = boundary_offsets(args.n_segments, args.segment_len)
    print("segments=%d x %d waypoints -> boundary offsets %s (= %.1f s horizon)"
          % (args.n_segments, args.segment_len, offsets, offsets[-1] * EXPECTED_DT))

    split_yaml, filter_yaml = locate_configs(args.devkit_root, args.filter)
    if not split_yaml or not filter_yaml:
        sys.exit("could not find both yamls for '%s'" % args.filter)
    data_split = parse_split_yaml(split_yaml)
    filter_kwargs = parse_scene_filter_yaml(filter_yaml)
    n_history = int(filter_kwargs.get("num_history_frames") or 4)
    print("filter '%s' -> data_split=%s, num_history_frames=%d"
          % (args.filter, data_split, n_history))

    from pathlib import Path

    from navsim.common.dataclasses import SensorConfig
    from navsim.common.dataloader import SceneFilter, SceneLoader

    logs = Path(args.data_root) / "navsim_logs" / data_split
    blobs = Path(args.data_root) / "sensor_blobs" / data_split
    loader = SceneLoader(data_path=logs, sensor_blobs_path=blobs,
                         scene_filter=SceneFilter(**filter_kwargs),
                         sensor_config=SensorConfig.build_no_sensors())
    tokens = list(loader.tokens)
    print("scenarios enumerated: %d" % len(tokens))
    if not tokens:
        sys.exit("[fatal] loader enumerated nothing")

    print("\n[1] frame spacing")
    check_spacing(loader, tokens)

    print("\n[2] boundary frames on disk")
    if frame_image_at(loader, tokens[0], str(blobs), max(n_history - 1, 0)) is None:
        sys.exit("[fatal] no CAM_F0 path for the first scenario. Loader "
                 "attributes: %s -- run navsim_tools/probe_frame_images.py, "
                 "which discovers the accessor instead of assuming it."
                 % describe_frame(loader))
    records, stats = build_records(loader, str(blobs), n_history, offsets)

    print("\n[3] count guard (ENUMERATED population)")
    guard_count(records, args.filter, stats["enumerated"], args.force_count)

    print("\n[4] scored-token join (the population rho(delta, PDMS) lives on)")
    if args.scored_tokens:
        scored = read_scored_tokens(args.scored_tokens)
        guard_scored_coverage(records, scored, args.filter, args.force_count)
        if args.only_scored:
            records = [r for r in records if r["sample_token"] in scored]
            print("    --only_scored -> %d records kept" % len(records))
    else:
        expected = SCORED_SUBSET.get(args.filter)
        print("    skipped (no --scored_tokens). ⚠ PDMS scores only %s of the "
              "%d enumerated scenarios (Step 2.13), so the join is UNVERIFIED."
              % (expected if expected else "a subset of", stats["enumerated"]))
        print("    Pass a PDMS result CSV to check it, e.g.")
        print("      --scored_tokens $NAVSIM_EXP_ROOT/human_agent/<run>/*.csv")
        if args.only_scored:
            sys.exit("[fatal] --only_scored needs --scored_tokens")

    os.makedirs(args.out_dir, exist_ok=True)
    suffix = "_scored" if (args.scored_tokens and args.only_scored) else ""
    out_path = os.path.join(args.out_dir,
                            "navsim_wm_eval_%s%s.json" % (args.filter, suffix))
    with open(out_path, "w") as handle:
        json.dump(records, handle)
    print("\nwrote %s  (%d records, %d frames each)"
          % (out_path, len(records), len(offsets)))
    if records:
        print("first record: %s" % json.dumps(records[0])[:200])


# --------------------------------------------------------------------------- #
# self-test: synthetic loader, no devkit and no data
# --------------------------------------------------------------------------- #

class _FakeLoader:
    def __init__(self, tokens, n_frames=14):
        self.tokens = tokens
        holder = {}
        for token in tokens:
            holder[token] = [
                {"cams": {"CAM_F0": {"data_path": "%s_%02d.jpg" % (token, i)}},
                 "timestamp": 1_000_000 + i * 500_000,
                 "log_name": "log_%s" % token[:2]}
                for i in range(n_frames)
            ]
        self.scene_frames_dicts = holder


def _selftest():
    failures = []

    def check(label, ok, detail=""):
        print("  %-46s %s %s" % (label, "ok" if ok else "FAIL", detail))
        if not ok:
            failures.append(label)

    tokens = ["t%03d" % i for i in range(30)]
    loader = _FakeLoader(tokens)
    offsets = boundary_offsets(4, 2)
    check("boundary offsets", offsets == [0, 2, 4, 6, 8], str(offsets))
    check("horizon is 4.0 s", abs(offsets[-1] * EXPECTED_DT - 4.0) < 1e-9)

    check("dt measured from microsecond stamps",
          abs((check_spacing(loader, tokens, out=lambda *a: None) or 0) - 0.5) < 1e-6)

    # a record needs frames at indices 3,5,7,9,11 -- all inside 14
    path = frame_image_at(loader, "t007", "/blobs", 11)
    check("path at the last boundary index",
          path == os.path.join("/blobs", "t007_11.jpg"), str(path))
    check("index past the end -> None",
          frame_image_at(loader, "t007", "/blobs", 99) is None)

    # build with a substituted on-disk test: patch os.listdir for the fake dir
    import builtins  # noqa: F401  (kept explicit: we patch os, not builtins)
    real_listdir = os.listdir

    def fake_listdir_all(_directory):
        return ["%s_%02d.jpg" % (t, i) for t in tokens for i in range(14)]

    def fake_listdir_no_tail(_directory):
        return ["%s_%02d.jpg" % (t, i) for t in tokens for i in range(11)]

    try:
        os.listdir = fake_listdir_all
        records, stats = build_records(loader, "/blobs", 4, offsets, out=lambda *a: None)
        check("all frames present -> every scenario kept",
              len(records) == len(tokens), "%d/%d" % (len(records), len(tokens)))
        check("record carries one path per boundary",
              records and len(records[0]["frames"]) == len(offsets))
        check("log name recovered", records and records[0]["scene"].startswith("log_"))

        # the last boundary (index 11) is absent -> ALL records must drop
        os.listdir = fake_listdir_no_tail
        records2, stats2 = build_records(loader, "/blobs", 4, offsets, out=lambda *a: None)
        check("one missing boundary drops the whole record",
              len(records2) == 0, "%d kept" % len(records2))
        check("  and the missing offset is named",
              stats2["per_offset_missing"][8] == len(tokens),
              str(stats2["per_offset_missing"]))
    finally:
        os.listdir = real_listdir

    # count guard is on the ENUMERATED population (69,405), not the scored one.
    # The first version had 12,146 here and refused a correct build -- so the
    # test pins both numbers, which is what would have caught it.
    quiet = lambda *a, **k: None                                   # noqa: E731
    try:
        guard_count([{"sample_token": "x"}] * 69405, "navtest", 69405,
                    forced=False, out=quiet)
        check("a complete navtest build (69,405) passes", True)
    except SystemExit as exc:
        check("a complete navtest build (69,405) passes", False, str(exc)[:60])
    try:
        guard_count([{"sample_token": "x"}] * 12146, "navtest", 69405,
                    forced=False, out=quiet)
        check("the SCORED count (12,146) is not the build target", False,
              "did not raise")
    except SystemExit:
        check("the SCORED count (12,146) is not the build target", True)
    try:
        guard_count([{"sample_token": "x"}] * 4193, "navtest", 69405,
                    forced=True, out=quiet)
        check("--force_count overrides the guard", True)
    except SystemExit:
        check("--force_count overrides the guard", False)

    # scored-token join: one missing token must fail loudly, not round to 100%
    built = [{"sample_token": t} for t in tokens]
    try:
        guard_scored_coverage(built, set(tokens[:10]), "navtest", False, out=quiet)
        check("lossless scored join passes", True)
    except SystemExit:
        check("lossless scored join passes", False)
    try:
        guard_scored_coverage(built, set(tokens) | {"absent"}, "navtest", False, out=quiet)
        check("one missing scored token is refused", False, "did not raise")
    except SystemExit:
        check("one missing scored token is refused", True)

    # token list reader: csv with a 'token' column, and a plain list
    import tempfile
    hex16 = ["%016x" % i for i in range(20)]
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = os.path.join(tmp, "scores.csv")
        with open(csv_path, "w") as handle:
            handle.write("index,token,score\n")
            for i, tok in enumerate(hex16):
                handle.write("%d,%s,0.9\n" % (i, tok))
            handle.write("20,average,0.85\n")      # the summary row run_pdm_score adds
        got = read_scored_tokens(csv_path, out=quiet)
        check("reads tokens from a PDMS csv", got == set(hex16), "%d" % len(got))
        check("  drops the 'average' aggregate row", "average" not in got)

        txt_path = os.path.join(tmp, "tokens.txt")
        with open(txt_path, "w") as handle:
            handle.write("\n".join(hex16) + "\n")
        check("reads a plain token list",
              read_scored_tokens(txt_path, out=quiet) == set(hex16))

    kept, dropped = _keep_real_tokens(hex16 + ["average"])
    check("dropped rows are reported, not silently eaten",
          dropped == ["average"] and len(kept) == 20, str(dropped))

    print("\nSELF-TEST:", "PASS" if not failures else "FAIL %s" % failures)
    return 0 if not failures else 1


if __name__ == "__main__":
    main()
