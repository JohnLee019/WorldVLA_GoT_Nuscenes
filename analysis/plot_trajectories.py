"""E5: BEV trajectory figures from an existing per_sample.csv. No GPU, no model.

Draws, per record: the GT future, the greedy free-run, the GoT plan, and -- when
the csv has them (results/fusion/* only, sec.10) -- the candidate pool the
selector chose from. That last layer is the point: sec.1.3's claim is that every
computable rule is stuck at 62.8-63.8% because the CANDIDATES are the ceiling,
and a reader believes that faster from one picture of eight near-identical
trajectories than from a table.

SVG, written by hand, no matplotlib. Vector is what a paper wants, and this
env is one where adding a plotting dependency is a real risk (pandas segfaults
in navsim_wvg, scipy is optional everywhere else in this repo).

CHOOSING WHICH RECORDS TO DRAW

Only three selectors, all deliberate:
    --worst N     largest GoT-minus-greedy loss  (the tail sec.1.2 is about)
    --random N    seeded random sample
    --tokens ...  named records
There is no "first N". Records are scene-ordered, so a prefix is a handful of
scenes -- the trap this project has hit four times (sec.9, Step 2.12c).

Usage:
    python plot_trajectories.py results/fusion/final_top3/per_sample.csv \
        --records_json ./data/nuscenes_records/nuscenes_val_scenespread.json \
        --out_dir ./results/figures --worst 6
    python plot_trajectories.py --selftest
"""

import argparse
import csv
import json
import os
import random
import sys

import numpy as np

W, H = 420, 560                 # svg canvas
MARGIN = 46
STYLE = {                       # (stroke, width, dash, label)
    "cand": ("#b9b9c4", 1.4, "3,3", "candidates"),
    "gt": ("#111111", 3.0, "", "ground truth"),
    "base": ("#1f77b4", 2.4, "7,4", "greedy"),
    "got": ("#d62728", 2.4, "", "GoT"),
}


def _parse(cell):
    if not cell:
        return None
    try:
        return np.asarray(json.loads(cell), dtype=np.float64)
    except (ValueError, TypeError):
        return None


def load(csv_path, records_json):
    with open(records_json) as handle:
        gt_by_token = {r["sample_token"]: np.asarray(r["waypoints"], dtype=np.float64)
                       for r in json.load(handle)}
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            token = row.get("sample_token")
            got = _parse(row.get("got_pred"))
            if token not in gt_by_token or got is None:
                continue
            cands = _parse(row.get("got_cand_wps"))
            entry = {
                "token": token, "scene": row.get("scene", "?"),
                "command": row.get("command", "?"),
                "gt": gt_by_token[token][: got.shape[0]],
                "got": got,
                "base": _parse(row.get("base_pred")),
                "cands": cands if cands is not None and cands.ndim == 3 else None,
            }
            for key in ("got_avgL2@3s", "base_avgL2@3s"):
                try:
                    entry[key] = float(row[key])
                except (KeyError, TypeError, ValueError):
                    pass
            rows.append(entry)
    return rows


def select(rows, args):
    """worst / random / tokens -- never a positional prefix (see the docstring)."""
    if args.tokens:
        wanted = set(args.tokens)
        return [r for r in rows if r["token"] in wanted]
    if args.random:
        return random.Random(args.seed).sample(rows, min(args.random, len(rows)))
    usable = [r for r in rows if "got_avgL2@3s" in r and "base_avgL2@3s" in r]
    if not usable:
        raise SystemExit(
            "--worst needs got_avgL2@3s and base_avgL2@3s in the csv. Use "
            "--random N or --tokens instead.")
    usable.sort(key=lambda r: r["base_avgL2@3s"] - r["got_avgL2@3s"])
    return usable[: args.worst]


# --------------------------------------------------------------------------- #
# svg
# --------------------------------------------------------------------------- #

def _projector(trajs):
    """ego frame (x forward, y left) -> screen (x up, y left), equal aspect."""
    pts = np.concatenate([t.reshape(-1, 2) for t in trajs if t is not None])
    x_hi = max(float(pts[:, 0].max()), 1.0) * 1.08
    y_abs = max(float(np.abs(pts[:, 1]).max()), 1.0) * 1.15
    scale = min((H - 2 * MARGIN) / x_hi, (W - 2 * MARGIN) / (2 * y_abs))
    cx, cy = W / 2.0, H - MARGIN

    def project(point):
        return (cx - point[1] * scale, cy - point[0] * scale)
    return project, scale, x_hi, y_abs


def _polyline(traj, project, stroke, width, dash, opacity=1.0):
    pts = " ".join("%.1f,%.1f" % project(p) for p in traj)
    dash_attr = ' stroke-dasharray="%s"' % dash if dash else ""
    return ('<polyline points="%s" fill="none" stroke="%s" stroke-width="%.1f"'
            ' stroke-linejoin="round" stroke-linecap="round" opacity="%.2f"%s/>'
            % (pts, stroke, width, opacity, dash_attr))


def render(entry, project=None):
    layers = [entry["gt"], entry["got"], entry["base"]]
    if entry["cands"] is not None:
        layers.append(entry["cands"].reshape(-1, 2))
    project, scale, x_hi, y_abs = _projector(layers)

    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" '
             'viewBox="0 0 %d %d" font-family="Helvetica,Arial,sans-serif">' % (W, H, W, H),
             '<rect width="%d" height="%d" fill="white"/>' % (W, H)]

    # grid: 10 m rings along x, centre line at y=0
    step = 10.0
    tick = step
    while tick <= x_hi:
        y = project((tick, 0.0))[1]
        parts.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="#e8e8ee" '
                     'stroke-width="1"/>' % (MARGIN // 2, y, W - MARGIN // 2, y))
        parts.append('<text x="%d" y="%.1f" font-size="10" fill="#9a9aa6">%dm</text>'
                     % (MARGIN // 2 + 2, y - 3, int(tick)))
        tick += step
    x0 = project((0.0, 0.0))[0]
    parts.append('<line x1="%.1f" y1="%d" x2="%.1f" y2="%.1f" stroke="#e8e8ee" '
                 'stroke-width="1"/>' % (x0, MARGIN // 2, x0, H - MARGIN))

    if entry["cands"] is not None:
        stroke, width, dash, _ = STYLE["cand"]
        for cand in entry["cands"]:
            parts.append(_polyline(cand, project, stroke, width, dash, 0.85))
    for key in ("base", "gt", "got"):          # GoT on top, GT under it
        traj = entry.get(key)
        if traj is None:
            continue
        stroke, width, dash, _ = STYLE[key]
        parts.append(_polyline(traj, project, stroke, width, dash))
        end = project(traj[-1])
        parts.append('<circle cx="%.1f" cy="%.1f" r="3.2" fill="%s"/>' % (end[0], end[1], stroke))

    ego = project((0.0, 0.0))
    parts.append('<circle cx="%.1f" cy="%.1f" r="4" fill="#111111"/>' % ego)

    # legend + caption
    y = 18
    for key in ("gt", "base", "got", "cand"):
        if key == "cand" and entry["cands"] is None:
            continue
        if key == "base" and entry.get("base") is None:
            continue
        stroke, width, dash, label = STYLE[key]
        dash_attr = ' stroke-dasharray="%s"' % dash if dash else ""
        parts.append('<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="%s" '
                     'stroke-width="%.1f"%s/>' % (MARGIN // 2, y, MARGIN // 2 + 26, y,
                                                  stroke, width, dash_attr))
        parts.append('<text x="%d" y="%d" font-size="11" fill="#333">%s</text>'
                     % (MARGIN // 2 + 32, y + 4, label))
        y += 16
    caption = "%s  %s" % (entry["token"][:12], entry["command"])
    if "got_avgL2@3s" in entry and "base_avgL2@3s" in entry:
        caption += "   GoT %.2f  greedy %.2f  (%+.2f)" % (
            entry["got_avgL2@3s"], entry["base_avgL2@3s"],
            entry["got_avgL2@3s"] - entry["base_avgL2@3s"])
    parts.append('<text x="%d" y="%d" font-size="11" fill="#555">%s</text>'
                 % (MARGIN // 2, H - 12, caption))
    parts.append("</svg>")
    return "\n".join(parts)


# --------------------------------------------------------------------------- #

def _selftest():
    failures = []

    def check(label, ok, detail=""):
        print("  %-40s %s %s" % (label, "ok" if ok else "FAIL", detail))
        if not ok:
            failures.append(label)

    t = np.arange(1, 7) * 0.5
    entry = {"token": "abcdef012345", "scene": "s0", "command": "straight",
             "gt": np.stack([8 * t, 0.2 * t], 1),
             "got": np.stack([8 * t, 0.5 * t], 1),
             "base": np.stack([8 * t, 0.1 * t], 1),
             "cands": np.stack([np.stack([8 * t, c * t], 1) for c in (-1, 0, 1)]),
             "got_avgL2@3s": 3.60, "base_avgL2@3s": 3.55}
    svg = render(entry)
    check("svg is well formed",
          svg.startswith("<svg") and svg.rstrip().endswith("</svg>")
          and svg.count("<svg") == 1)
    check("every layer is drawn", svg.count("<polyline") == 3 + 3)
    check("caption carries the paired numbers", "+0.05" in svg, "")

    project, _, _, _ = _projector([entry["gt"]])
    ego = project((0.0, 0.0))
    ahead = project((10.0, 0.0))
    left = project((0.0, 5.0))
    check("forward is up", ahead[1] < ego[1])
    check("left is left", left[0] < ego[0])
    check("origin is on the centre line", abs(ahead[0] - ego[0]) < 1e-6)

    rows = [dict(entry, token="t%02d" % i,
                 **{"got_avgL2@3s": 3.0 + i * 0.1, "base_avgL2@3s": 3.0})
            for i in range(20)]
    args = argparse.Namespace(tokens=None, random=None, worst=3, seed=0)
    picked = [r["token"] for r in select(rows, args)]
    check("--worst picks the biggest GoT loss", picked == ["t19", "t18", "t17"], str(picked))
    args = argparse.Namespace(tokens=None, random=5, worst=0, seed=0)
    a = [r["token"] for r in select(rows, args)]
    b = [r["token"] for r in select(rows, args)]
    check("--random is seeded and not a prefix",
          a == b and a != [r["token"] for r in rows[:5]], str(a))

    # a csv without the candidate column must still draw, minus that layer
    bare = dict(entry, cands=None, base=None)
    svg = render(bare)
    # ">greedy<" is the LEGEND entry; the caption legitimately prints the word
    # next to greedy's number, so matching on the bare word would be wrong
    check("degrades without candidates or greedy",
          svg.count("<polyline") == 2 and ">candidates<" not in svg
          and ">greedy<" not in svg, "gt + got remain")

    if failures:
        print("\nSELFTEST FAILED: %s" % failures)
        return 1
    print("\nSELFTEST PASS")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?")
    ap.add_argument("--records_json")
    ap.add_argument("--out_dir", default="./results/figures")
    ap.add_argument("--worst", type=int, default=6)
    ap.add_argument("--random", type=int, default=0)
    ap.add_argument("--tokens", nargs="*")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(_selftest())
    if not args.csv or not args.records_json:
        sys.exit("need a per_sample.csv and --records_json")

    rows = load(args.csv, args.records_json)
    print("usable records: %d  (candidate pool in csv: %s)"
          % (len(rows), "yes" if rows and rows[0]["cands"] is not None else
             "NO -- only results/fusion/* has got_cand_wps, sec.10"))
    if not rows:
        sys.exit("[fatal] nothing to draw")
    picked = select(rows, args)
    os.makedirs(args.out_dir, exist_ok=True)
    for entry in picked:
        path = os.path.join(args.out_dir, "%s_%s.svg" % (entry["token"][:12], entry["command"]))
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(render(entry))
        note = ""
        if "got_avgL2@3s" in entry and "base_avgL2@3s" in entry:
            note = "  GoT-greedy %+.3f" % (entry["got_avgL2@3s"] - entry["base_avgL2@3s"])
        print("  %s%s" % (path, note))
    print("%d figures -> %s" % (len(picked), args.out_dir))


if __name__ == "__main__":
    main()
