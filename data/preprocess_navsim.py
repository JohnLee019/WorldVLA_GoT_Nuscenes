"""Build NAVSIM training records in the SAME schema as the nuScenes ones.

    {
        "sample_token": str,             # NAVSIM 16-hex scenario token
        "scene": str,                    # log name -- the clustering unit (sec.9)
        "images": [abs_path, ...],       # CAM_F0 first
        "waypoints": [[x, y], ...],      # length == n_future, ego frame, metres
        "headings": [h, ...],            # radians. EXTRA field, see below
        "command": "left" | "right" | "straight"
    }

Keeping the schema identical is the point: dataset_nuscenes.py and
item_processor.py then run unchanged, and so does the whole GoT candidate
path (handoff sec.8 keeps action_dim=2 for exactly this reason).

WHAT THE DEVKIT DOES FOR US (Step 2.13)

    scene.get_future_trajectory().poses  ->  ndarray (10, 3)   x, y, heading

so there is no ego2global differencing, no sample_next walking and no
logs-vs-blobs intersection to maintain. We take the first n_future rows.
`headings` is stored because it is exact and free; nothing reads it yet.

THREE THINGS THIS SCRIPT REFUSES TO ASSUME

  1. The command one-hot. NAVSIM ships driving_command as a 4-vector and the
     slot order is a guess until measured, so every record's slot is
     cross-tabulated against the geometric command derived from its own final
     lateral offset (the nuScenes rule). The mapping is REPORTED and checked;
     a disagreeing slot is a loud failure, not a silent relabel.

  2. The image path accessor. Several plausible ways to reach CAM_F0 are tried
     in order, the one that worked is printed, and if none work the script
     stops with a dump of what the frame object actually offers.

  3. The frame interval. Step 2.13 measured 0.5 s, but the script asserts it
     from TrajectorySampling rather than trusting the note.

VAL SPLIT IS LOG-DISJOINT

  navtest is the benchmark; validating on it during training would fit the
  thing we report. So a fraction of navtrain LOGS is held out -- logs, not
  scenarios, because scenarios from one log share a scene and are not
  independent (handoff sec.9, the project's largest trap).

Usage (navsim_wvg env, from the repo root):
    python -m data.preprocess_navsim --devkit_root $NAVSIM_DEVKIT_ROOT \
        --data_root $OPENSCENE_DATA_ROOT --filter navtrain --out_dir ./data/navsim_records
    python -m data.preprocess_navsim --selftest
"""

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict

import numpy as np

# preregistered: NAVSIM's driving_command one-hot, verified by cross-tab below
COMMAND_SLOTS = ["left", "straight", "right", "unknown"]
LATERAL_THRESH = 2.0      # same rule as nuScenes derive_command (handoff sec.3)
DEFAULT_N_FUTURE = 8      # 4 s at 0.5 s (handoff sec.8)
EXPECTED_DT = 0.5

# Preregistered, from navsim_tools/probe_frame_images.py (session 15, exact
# pass over all 651,526 enumerated scenarios, not an estimate):
#
#   enumerated 651,526  ->  151,778 have a CAM_F0 image on disk (23.3%)
#
# The paper's ~85k is NOT the target and disagreeing with it is not an error:
# NAVSIM ships sensor blobs per scenario while OpenScene's trainval LOGS carry
# every frame's metadata, and navtrain's 85k counts scenarios usable with 8
# cameras + lidar + history. We need one front frame, so our population is
# larger. The probe's per-log coverage settled that this is the distribution
# and not somebody's truncated download: 0 logs at 0%, 0 logs above 90%,
# 985/1192 between 10% and 50% -- unimodal, which a partial copy cannot be.
EXPECTED_RECORDS = 151778
RECORD_WINDOW = (120000, 180000)

# navtest is a different population and needs its own expectation: every one of
# its 69,405 enumerated scenarios has the frame on disk (session 16, exact pass
# -- handoff Step 2.15). Sharing navtrain's window would have made a correct
# navtest build look broken, which is how a guard turns into noise and then
# into --force_count.
EXPECTED_BY_FILTER = {
    "navtrain": (EXPECTED_RECORDS, RECORD_WINDOW),
    "navtest": (69405, (62000, 69405)),
}
# Which split the action-normalisation belongs to. navsim_norm.json defines what
# the action tokenizer can represent AT TRAINING TIME, so it is written from the
# training split and nothing else may overwrite it.
NORM_OWNER = "navtrain"
SCALARS = ("num_history_frames", "num_future_frames", "frame_interval",
           "has_route", "max_scenes")


# --------------------------------------------------------------------------- #
# config parsing (stdlib: PyYAML/hydra are avoided, sec.9 segfault environment)
# --------------------------------------------------------------------------- #

def _scalar(text):
    text = text.strip().strip("'\"")
    if text in ("null", "None", ""):
        return None
    if text in ("true", "True"):
        return True
    if text in ("false", "False"):
        return False
    try:
        return int(text)
    except ValueError:
        return text


def parse_scene_filter_yaml(path):
    kwargs, log_names, in_logs = {}, [], False
    with open(path) as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            if not line.strip() or line.strip().startswith("#"):
                continue
            if in_logs:
                stripped = line.strip()
                if stripped.startswith("- "):
                    log_names.append(stripped[2:].strip().strip("'\""))
                    continue
                in_logs = False
            if line.strip().startswith("log_names:"):
                tail = line.split(":", 1)[1].strip()
                in_logs = tail == ""
                continue
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            if key.strip() in SCALARS:
                kwargs[key.strip()] = _scalar(value)
    if log_names:
        kwargs["log_names"] = log_names
    return kwargs


def parse_split_yaml(path):
    data_split = None
    with open(path) as handle:
        for raw in handle:
            line = raw.strip()
            if line.startswith("data_split:"):
                data_split = line.split(":", 1)[1].strip()
    return data_split


def locate_configs(devkit_root, name):
    split_yaml = filter_yaml = None
    for root, dirs, names in os.walk(devkit_root):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        if "train_test_split" not in root:
            continue
        target = name + ".yaml"
        if target in names:
            path = os.path.join(root, target)
            if os.path.basename(root) == "scene_filter":
                filter_yaml = path
            else:
                split_yaml = path
    return split_yaml, filter_yaml


# --------------------------------------------------------------------------- #
# record assembly
# --------------------------------------------------------------------------- #

def derive_command(waypoints, lateral_thresh=LATERAL_THRESH):
    """Geometric command from the final lateral offset -- used here ONLY to
    audit the dataset's own one-hot, never to label the records."""
    final_y = float(waypoints[-1][1])
    if final_y > lateral_thresh:
        return "left"
    if final_y < -lateral_thresh:
        return "right"
    return "straight"


def resolve_image(frame, sensor_root, log_name, camera="cam_f0"):
    """Return an absolute CAM_F0 path, trying the plausible accessors in order.

    Returns (path, strategy_name) or (None, None). The strategy that worked is
    printed once, so the next reader knows which one this devkit uses instead
    of inferring it from a silent success.
    """
    cams = getattr(frame, "cams", None)
    if cams is not None:
        entry = None
        if isinstance(cams, dict):
            for key in (camera, camera.upper()):
                if key in cams:
                    entry = cams[key]
                    break
        else:
            entry = getattr(cams, camera, None) or getattr(cams, camera.upper(), None)
        if entry is not None:
            for attr in ("camera_path", "data_path", "filepath", "path"):
                value = entry.get(attr) if isinstance(entry, dict) else getattr(entry, attr, None)
                if value:
                    value = str(value)
                    if not os.path.isabs(value) and sensor_root:
                        value = os.path.join(sensor_root, value)
                    return value, "frame.cams[%s].%s" % (camera, attr)
    for attr in ("cam_f0", camera):
        entry = getattr(frame, attr, None)
        if entry is None:
            continue
        for sub in ("camera_path", "data_path", "path"):
            value = getattr(entry, sub, None)
            if value:
                value = str(value)
                if not os.path.isabs(value) and sensor_root:
                    value = os.path.join(sensor_root, value)
                return value, "frame.%s.%s" % (attr, sub)
    return None, None


def describe_frame(frame):
    fields = [n for n in dir(frame) if not n.startswith("_")]
    return ", ".join(fields[:40])


# --------------------------------------------------------------------------- #
# image prefilter -- measured, see EXPECTED_RECORDS above
# --------------------------------------------------------------------------- #

RAW_HOLDERS = ("scene_frames_dicts", "scene_frames")


def raw_image_path(loader, token, sensor_root, n_history):
    """CAM_F0 path straight from the raw log dict.

    The devkit's Frame exposes `cameras`, but with SensorConfig.build_no_sensors
    those Camera objects carry no path at all -- measured, session 15. The path
    only exists in the raw dict the loader keeps, at

        loader.scene_frames_dicts[token][n_history - 1]['cams']['CAM_F0']['data_path']

    and it is already absolute.
    """
    for name in RAW_HOLDERS:
        holder = getattr(loader, name, None)
        if not isinstance(holder, dict):
            continue
        frames = holder.get(token)
        if not frames:
            continue
        frame = frames[min(max(int(n_history) - 1, 0), len(frames) - 1)]
        try:
            value = frame["cams"]["CAM_F0"]["data_path"]
        except (KeyError, TypeError, IndexError):
            return None
        if value:
            value = str(value)
            if not os.path.isabs(value) and sensor_root:
                value = os.path.join(sensor_root, value)
            return value
    return None


def prefilter_by_image(loader, sensor_root, n_history, out=print):
    """Keep only scenarios whose CAM_F0 frame is actually on disk.

    Done BEFORE any Scene is built, which is the whole point: 76.7% of the
    enumerated scenarios have no image, and get_scene_from_token is the
    expensive call. One listdir per log makes the test nearly free.
    """
    tokens = list(loader.tokens)
    keep, paths = [], {}
    dir_names, missing, unresolved = {}, 0, 0
    for token in tokens:
        path = raw_image_path(loader, token, sensor_root, n_history)
        if not path:
            unresolved += 1
            continue
        directory = os.path.dirname(path)
        names = dir_names.get(directory)
        if names is None:
            try:
                names = set(os.listdir(directory))
            except OSError:
                names = set()
            dir_names[directory] = names
        if os.path.basename(path) in names:
            keep.append(token)
            paths[token] = path
        else:
            missing += 1
    out("image prefilter: %d enumerated -> %d with CAM_F0 on disk "
        "(absent %d, unresolved %d)"
        % (len(tokens), len(keep), missing, unresolved))
    if unresolved and unresolved == len(tokens):
        raise SystemExit(
            "[fatal] no scenario yielded a CAM_F0 path. The loader attributes "
            "are: %s -- re-run navsim_tools/probe_frame_images.py, which "
            "discovers the accessor instead of assuming it." % describe_frame(loader))
    return keep, paths


def build_records(loader, tokens, paths, n_future, out=print):
    """One record per scenario token that survived the image prefilter."""
    records = []
    crosstab = defaultdict(Counter)
    skipped = Counter()
    dt_checked = False

    for index, token in enumerate(tokens):
        scene = loader.get_scene_from_token(token)
        trajectory = scene.get_future_trajectory()
        poses = np.asarray(trajectory.poses, dtype=np.float64)
        if poses.shape[0] < n_future:
            skipped["short_future"] += 1
            continue

        if not dt_checked:
            sampling = getattr(trajectory, "trajectory_sampling", None)
            interval = getattr(sampling, "interval_length", None)
            out("    trajectory interval_length = %s" % interval)
            if interval is not None and abs(float(interval) - EXPECTED_DT) > 1e-6:
                raise SystemExit(
                    "[fatal] pose interval is %s s, not %s. n_future=%d would be "
                    "a %.1f s horizon, not the 4 s the pivot assumes (sec.8)."
                    % (interval, EXPECTED_DT, n_future, n_future * float(interval)))
            dt_checked = True

        # the scenario's "current" frame is the last history frame
        frames = getattr(scene, "frames", None)
        meta = getattr(scene, "scene_metadata", None)
        n_history = getattr(meta, "num_history_frames", None) or len(frames) - poses.shape[0]
        current = frames[max(int(n_history) - 1, 0)]

        image = paths[token]      # prefilter already proved this file exists
        if index == 0:
            out("    first image: %s" % image)

        command_vector = getattr(current, "driving_command", None)
        if command_vector is None:
            ego = getattr(current, "ego_status", None)
            command_vector = getattr(ego, "driving_command", None)
        if command_vector is None:
            skipped["no_driving_command"] += 1
            continue
        slot = int(np.argmax(np.asarray(command_vector)))
        if slot >= len(COMMAND_SLOTS):
            skipped["command_slot_out_of_range"] += 1
            continue

        waypoints = poses[:n_future, :2]
        crosstab[slot][derive_command(waypoints)] += 1

        records.append({
            "sample_token": token,
            "scene": str(getattr(meta, "log_name", "?")),
            "images": [image],
            "waypoints": waypoints.astype(np.float32).round(4).tolist(),
            "headings": poses[:n_future, 2].astype(np.float32).round(5).tolist(),
            "command": COMMAND_SLOTS[slot],
        })
        if (index + 1) % 10000 == 0:
            out("    %d / %d scenarios" % (index + 1, len(tokens)))

    return records, {"crosstab": crosstab, "skipped": dict(skipped)}


def audit_command_mapping(crosstab, out=print):
    """Is COMMAND_SLOTS the right labelling? Measured, not assumed.

    Each slot is cross-tabulated against the geometric command implied by that
    record's own waypoints. A slot named "left" whose records mostly turn right
    means the constant is wrong, and relabelling silently would corrupt every
    record in the run.
    """
    out("")
    out("[command audit] dataset one-hot vs geometry (|final lateral| > %.1f m)"
        % LATERAL_THRESH)
    out("    %-10s %8s %10s %8s %10s   %s"
        % ("slot", "left", "straight", "right", "n", "verdict"))
    ok = True
    for slot in sorted(crosstab):
        counts = crosstab[slot]
        total = sum(counts.values())
        name = COMMAND_SLOTS[slot]
        turns = {k: counts.get(k, 0) for k in ("left", "right")}
        modal_turn = max(turns, key=turns.get)
        # 'straight' dominating any slot is expected -- most driving is
        # straight even when a turn is commanded -- so the test is on which
        # TURN direction dominates, not on the overall mode.
        if name in ("left", "right"):
            good = modal_turn == name and turns[name] > turns[
                "right" if name == "left" else "left"]
        elif name == "straight":
            good = counts.get("straight", 0) >= max(turns.values())
        else:
            good = True
        ok = ok and good
        out("    %-10s %8d %10d %8d %10d   %s"
            % ("%d=%s" % (slot, name), counts.get("left", 0),
               counts.get("straight", 0), counts.get("right", 0), total,
               "ok" if good else "MISLABELLED"))
    if not ok:
        raise SystemExit(
            "[fatal] COMMAND_SLOTS does not match the data. Fix the constant "
            "at the top of this file; do NOT relabel records to match a guess.")
    out("    COMMAND_SLOTS verified against geometry")
    return ok


def compute_norm_stats(records, pad=1.25):
    """Identical rule to data/preprocess_nuscenes.py: x floored at 0 and padded,
    y forced symmetric. The range decides what the action tokenizer can
    represent at all, so a split that happens to lack one turn direction must
    not be able to make it unrepresentable.

    Recomputing this for NAVSIM is mandatory, not optional: 4 s at highway
    speed reaches ~53 m where nuScenes' 3 s reached ~45 m (Step 2.12a).
    """
    all_wp = np.array([wp for r in records for wp in r["waypoints"]],
                      dtype=np.float64)
    raw_min, raw_max = all_wp.min(axis=0), all_wp.max(axis=0)
    x_lo = min(0.0, float(raw_min[0])) - 1.0
    x_hi = float(raw_max[0]) * pad
    y_abs = max(abs(float(raw_min[1])), abs(float(raw_max[1]))) * pad
    return {
        "min": [round(x_lo, 4), round(-y_abs, 4)],
        "max": [round(x_hi, 4), round(y_abs, 4)],
        "raw_stats": {
            "min": raw_min.round(4).tolist(),
            "max": raw_max.round(4).tolist(),
            "mean": all_wp.mean(axis=0).round(4).tolist(),
            "std": all_wp.std(axis=0).round(4).tolist(),
            "p1": np.percentile(all_wp, 1, axis=0).round(4).tolist(),
            "p99": np.percentile(all_wp, 99, axis=0).round(4).tolist(),
        },
        "n_waypoint_vectors": int(all_wp.shape[0]),
    }


def split_by_log(records, val_frac, seed=0):
    """Hold out whole LOGS, never scenarios.

    Scenarios from one log share a scene, so a scenario-level split leaks:
    the val set would be near-duplicates of training frames and every val
    number would be optimistic. This is handoff sec.9's largest trap.
    """
    logs = sorted(set(r["scene"] for r in records))
    rng = random.Random(seed)
    n_val = max(1, int(round(len(logs) * val_frac)))
    val_logs = set(rng.sample(logs, min(n_val, len(logs))))
    train = [r for r in records if r["scene"] not in val_logs]
    val = [r for r in records if r["scene"] in val_logs]
    return train, val, sorted(val_logs)


def report(name, records, out=print):
    if not records:
        out("[%s] EMPTY" % name)
        return
    counts = Counter(r["command"] for r in records)
    wp = np.array([r["waypoints"] for r in records], dtype=np.float64)
    out("[%s] records=%d logs=%d  command=%s"
        % (name, len(records), len(set(r["scene"] for r in records)), dict(counts)))
    out("    x=[%.2f, %.2f]  y=[%.2f, %.2f]"
        % (wp[:, :, 0].min(), wp[:, :, 0].max(),
           wp[:, :, 1].min(), wp[:, :, 1].max()))
    for direction in ("left", "right"):
        if counts.get(direction, 0) == 0:
            out("    WARNING: no '%s' in this split -- do not fit the "
                "normalization range to it" % direction)


# --------------------------------------------------------------------------- #
# self-test (no devkit, no data)
# --------------------------------------------------------------------------- #

def _fake_records(n_logs=10, per_log=20, seed=0, mislabel=False):
    rng = random.Random(seed)
    out = []
    for log in range(n_logs):
        for i in range(per_log):
            turn = rng.choice(["left", "straight", "right"])
            lateral = {"left": 4.0, "straight": 0.0, "right": -4.0}[turn]
            wp = [[4.0 * (k + 1), lateral * (k + 1) / 8.0] for k in range(8)]
            name = turn
            if mislabel and turn in ("left", "right"):
                name = "right" if turn == "left" else "left"
            out.append({"sample_token": "%02d%02d" % (log, i),
                        "scene": "log_%02d" % log, "images": ["x.jpg"],
                        "waypoints": wp, "headings": [0.0] * 8,
                        "command": name})
    return out


def _selftest():
    failures = []

    def check(label, ok, detail=""):
        print("  %-34s %s %s" % (label, "ok" if ok else "FAIL", detail))
        if not ok:
            failures.append((label, detail))

    records = _fake_records()
    train, val, val_logs = split_by_log(records, 0.2, seed=1)
    train_logs = set(r["scene"] for r in train)
    check("val split is log-disjoint",
          not (train_logs & set(val_logs)) and len(val_logs) == 2,
          "%d val logs" % len(val_logs))
    check("split keeps every record", len(train) + len(val) == len(records))

    stats = compute_norm_stats(records)
    check("norm y is symmetric", stats["min"][1] == -stats["max"][1],
          str(stats["min"][1]))
    check("norm x floor <= -1", stats["min"][0] <= -1.0, str(stats["min"][0]))
    check("norm pads x above raw",
          stats["max"][0] > stats["raw_stats"]["max"][0])

    # the audit must PASS on consistent data and FAIL on flipped labels --
    # a check that only ever sees the good world proves nothing (sec.11.7)
    good = defaultdict(Counter)
    for r in _fake_records():
        good[COMMAND_SLOTS.index(r["command"])][derive_command(r["waypoints"])] += 1
    try:
        audit_command_mapping(good, out=lambda *a: None)
        passed_good = True
    except SystemExit:
        passed_good = False
    check("audit passes on correct labels", passed_good)

    bad = defaultdict(Counter)
    for r in _fake_records(mislabel=True):
        bad[COMMAND_SLOTS.index(r["command"])][derive_command(r["waypoints"])] += 1
    try:
        audit_command_mapping(bad, out=lambda *a: None)
        caught = False
    except SystemExit:
        caught = True
    check("audit catches flipped labels", caught)

    # the prefilter must drop scenarios whose image is absent and keep the
    # rest -- and it must do it from the raw dict, since that is the only
    # place the path exists (session 15)
    class _Loader(object):
        def __init__(self, tokens, present):
            self.tokens = tokens
            self.scene_frames_dicts = {}
            for token in tokens:
                path = "/blobs/log/CAM_F0/%s.jpg" % token
                self.scene_frames_dicts[token] = [
                    {"cams": {"CAM_F0": {"data_path": path}}} for _ in range(14)]
            self.present = present

    import tempfile
    tmp = tempfile.mkdtemp()
    cam_dir = os.path.join(tmp, "log", "CAM_F0")
    os.makedirs(cam_dir)
    toks = ["t%02d" % i for i in range(10)]
    for i, token in enumerate(toks):
        if i % 2 == 0:
            with open(os.path.join(cam_dir, token + ".jpg"), "w") as handle:
                handle.write("x")

    class _RelLoader(_Loader):
        def __init__(self, tokens):
            _Loader.__init__(self, tokens, None)
            for token in tokens:
                rel = os.path.join("log", "CAM_F0", token + ".jpg")
                self.scene_frames_dicts[token] = [
                    {"cams": {"CAM_F0": {"data_path": os.path.join(tmp, rel)}}}
                    for _ in range(14)]

    kept, paths = prefilter_by_image(_RelLoader(toks), tmp, 4, out=lambda *a: None)
    check("prefilter keeps only images on disk",
          sorted(kept) == ["t00", "t02", "t04", "t06", "t08"], str(sorted(kept)))
    check("prefilter returns a path per kept token",
          set(paths) == set(kept) and all(os.path.exists(p) for p in paths.values()))
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)

    check("derive_command thresholds",
          derive_command([[10.0, 3.0]]) == "left"
          and derive_command([[10.0, -3.0]]) == "right"
          and derive_command([[10.0, 1.0]]) == "straight")

    # --- per-filter expectations (session 16) ------------------------------ #
    # navtrain and navtest are different populations; sharing one window would
    # make a correct navtest build look broken.
    check("navtest has its own preregistered window",
          EXPECTED_BY_FILTER["navtest"][0] == 69405
          and not (EXPECTED_BY_FILTER["navtrain"][1][0]
                   <= 69405 <= EXPECTED_BY_FILTER["navtrain"][1][1]))

    # --- norm ownership: the silent-corruption guard ----------------------- #
    class _Args(object):
        def __init__(self, filter_name, write_norm="auto"):
            self.filter = filter_name
            self.write_norm = write_norm

    tmp = tempfile.mkdtemp()
    norm_path = os.path.join(tmp, "navsim_norm.json")
    quiet = lambda *a, **k: None                                    # noqa: E731

    check("norm: written for the training split when absent",
          _may_write_norm(norm_path, _Args("navtrain"), out=quiet) is True)
    check("norm: NOT written for an eval split",
          _may_write_norm(norm_path, _Args("navtest"), out=quiet) is False)

    with open(norm_path, "w") as handle:
        json.dump({"filter": "navtrain", "n_future": 8}, handle)
    check("norm: navtrain may rewrite its own file",
          _may_write_norm(norm_path, _Args("navtrain"), out=quiet) is True)
    check("norm: another split cannot overwrite it",
          _may_write_norm(norm_path, _Args("navtest"), out=quiet) is False)
    caught = False
    try:
        _may_write_norm(norm_path, _Args("navtest", "yes"), out=quiet)
    except SystemExit:
        caught = True
    check("norm: --write_norm yes does NOT override ownership", caught)

    # a file from before the ownership field must still be protected
    with open(norm_path, "w") as handle:
        json.dump({"n_future": 8}, handle)
    check("norm: legacy file (no owner field) is treated as navtrain's",
          _may_write_norm(norm_path, _Args("navtest"), out=quiet) is False
          and _may_write_norm(norm_path, _Args("navtrain"), out=quiet) is True)
    shutil.rmtree(tmp, ignore_errors=True)

    if failures:
        print("\nSELFTEST FAILED: %s" % failures)
        return 1
    print("\nSELFTEST PASS")
    return 0


# --------------------------------------------------------------------------- #

def _may_write_norm(norm_path, args, out=print):
    """Is this run allowed to write navsim_norm.json?

    This guard exists because the failure it prevents is SILENT and expensive.
    navsim_norm.json decides what the action tokenizer can represent; a run
    built for a different split (navtest reaches further than navtrain's 4 s
    highway range, sec.6.3) would rewrite that range under a model already
    trained against the old one, and nothing downstream would raise. run_navsim
    .sh checks n_future, not provenance.

    Ownership is recorded in the file from now on. Files written before this
    field existed are assumed to belong to NORM_OWNER, which is what they are.
    """
    if args.write_norm == "no":
        out("norm: skipped (--write_norm no)")
        return False
    if os.path.exists(norm_path):
        try:
            with open(norm_path) as handle:
                owner = json.load(handle).get("filter", NORM_OWNER)
        except (OSError, ValueError):
            owner = NORM_OWNER
        if owner != args.filter:
            message = ("norm: '%s' already belongs to split '%s' -- refusing to "
                       "overwrite it from '%s'." % (norm_path, owner, args.filter))
            if args.write_norm == "yes":
                raise SystemExit(
                    "[fatal] %s\n  --write_norm yes does not override ownership. "
                    "Move the existing file aside if you really mean to replace "
                    "the training normalisation." % message)
            out(message)
            return False
    if args.write_norm == "auto" and args.filter != NORM_OWNER:
        out("norm: skipped -- '%s' is not the training split (%s)"
            % (args.filter, NORM_OWNER))
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--devkit_root", default=os.environ.get("NAVSIM_DEVKIT_ROOT"))
    ap.add_argument("--data_root", default=os.environ.get("OPENSCENE_DATA_ROOT"))
    ap.add_argument("--filter", default="navtrain")
    ap.add_argument("--out_dir", default="./data/navsim_records")
    ap.add_argument("--n_future", type=int, default=DEFAULT_N_FUTURE)
    ap.add_argument("--val_frac", type=float, default=0.05,
                    help="fraction of LOGS held out for validation")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0,
                    help="debug only: stop after N scenarios. NEVER use for a "
                         "real build -- tokens are log-ordered, so a limit is "
                         "a handful of logs, not a sample (sec.9)")
    ap.add_argument("--force_count", action="store_true",
                    help="build even if the record count leaves the "
                         "preregistered window -- only once you know why")
    ap.add_argument("--no_split", action="store_true",
                    help="write one file instead of train/val. Use for an "
                         "EVALUATION split (navtest): holding logs out of it "
                         "would be splitting what we report on")
    ap.add_argument("--write_norm", choices=("auto", "yes", "no"), default="auto",
                    help="auto (default) writes navsim_norm.json only for the "
                         "training split and never over another split's file")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(_selftest())
    if not args.devkit_root or not args.data_root:
        sys.exit("NAVSIM_DEVKIT_ROOT / OPENSCENE_DATA_ROOT unset -- activate "
                 "navsim_wvg first")

    split_yaml, filter_yaml = locate_configs(args.devkit_root, args.filter)
    if not split_yaml or not filter_yaml:
        sys.exit("could not find both yamls for '%s'" % args.filter)
    data_split = parse_split_yaml(split_yaml)
    filter_kwargs = parse_scene_filter_yaml(filter_yaml)
    print("filter '%s' -> data_split=%s, log_names=%d"
          % (args.filter, data_split,
             len(filter_kwargs.get("log_names") or [])))

    from pathlib import Path

    from navsim.common.dataclasses import SensorConfig
    from navsim.common.dataloader import SceneFilter, SceneLoader

    logs = Path(args.data_root) / "navsim_logs" / data_split
    blobs = Path(args.data_root) / "sensor_blobs" / data_split
    loader = SceneLoader(data_path=logs, sensor_blobs_path=blobs,
                         scene_filter=SceneFilter(**filter_kwargs),
                         sensor_config=SensorConfig.build_no_sensors())
    print("scenarios enumerated: %d" % len(loader.tokens))
    if args.limit:
        loader.tokens = list(loader.tokens)[:args.limit]
        print("!! --limit %d is a handful of logs, not a sample. debug only."
              % args.limit)

    n_history = int(filter_kwargs.get("num_history_frames") or 4)
    kept, paths = prefilter_by_image(loader, str(blobs), n_history)
    records, stats = build_records(loader, kept, paths, args.n_future)
    print("records built: %d  skipped: %s" % (len(records), stats["skipped"]))
    if not records:
        sys.exit("[fatal] no records")

    # The enumeration count is NOT a check -- 651,526 sailed past this script
    # once already because nothing here compared it to anything. The usable
    # record count is the number with a preregistered expectation (see the
    # constants at the top), so that is what gets gated.
    expected, (low, high) = EXPECTED_BY_FILTER.get(
        args.filter, (EXPECTED_RECORDS, RECORD_WINDOW))
    in_window = low <= len(records) <= high
    if not in_window and not (args.limit or args.force_count):
        sys.exit(
            "[fatal] %d records is outside the preregistered window %d-%d "
            "for '%s' (expected ~%d, measured by the probes over the full "
            "enumeration).\n"
            "  Something upstream moved: the split yaml, the blob tree, or the "
            "prefilter. Find out what before building on it -- a plausible "
            "number from a changed population is this project's most repeated "
            "failure (sec.9). Re-run with --force_count once it is explained."
            % (len(records), low, high, args.filter, expected))
    if not in_window:
        print("[warn] record count outside %d-%d, continuing on --force_count/--limit"
              % (low, high))

    audit_command_mapping(stats["crosstab"])

    os.makedirs(args.out_dir, exist_ok=True)

    # navtest is an EVALUATION split: holding logs out of it would be splitting
    # the thing we report on. --no_split writes one file and skips the norm.
    if args.no_split:
        print("")
        report(args.filter, records)
        path = os.path.join(args.out_dir, "navsim_%s.json" % args.filter)
        with open(path, "w") as handle:
            json.dump(records, handle)
        print("wrote %d -> %s" % (len(records), path))
        train, val_logs = records, []
    else:
        train, val, val_logs = split_by_log(records, args.val_frac, args.seed)
        print("")
        report("train", train)
        report("val", val)
        print("    held-out logs: %s%s"
              % (", ".join(val_logs[:3]), " ..." if len(val_logs) > 3 else ""))
        for name, subset in (("train", train), ("val", val)):
            path = os.path.join(args.out_dir, "navsim_%s_%s.json" % (args.filter, name))
            with open(path, "w") as handle:
                json.dump(subset, handle)
            print("wrote %d -> %s" % (len(subset), path))

    norm_path = os.path.join(args.out_dir, "navsim_norm.json")
    if not _may_write_norm(norm_path, args, out=print):
        print("\n(norm not written -- the existing one belongs to another split)")
        return
    norm = compute_norm_stats(train)
    norm["n_future"] = args.n_future
    norm["action_dim"] = 2
    norm["cameras"] = ["CAM_F0"]
    norm["val_logs"] = val_logs
    norm["filter"] = args.filter
    with open(norm_path, "w") as handle:
        json.dump(norm, handle, indent=2)
    print("\n=== norm (recomputed for %d x 0.5 s; nuScenes' range is NOT reusable) ==="
          % args.n_future)
    print(json.dumps({k: norm[k] for k in ("min", "max", "n_future")}, indent=2))
    print("saved -> %s" % norm_path)


if __name__ == "__main__":
    main()
