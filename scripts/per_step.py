import json
import sys

s = json.load(open(sys.argv[1] if len(sys.argv) > 1
                   else "results/headline/ref/summary.json"))


def dig(*path):
    cur = s
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


seed = sorted(dig("got_per_seed") or {})
rows = [
    ("GoT s" + seed[0] if seed else "GoT",
     dig("got_per_seed", seed[0], "per_step_L2") if seed else None),
    ("greedy free-run", dig("baseline_free_run", "per_step_L2")),
    ("mean-traj prior", dig("baseline_mean_traj", "per_step_L2")),
]

t = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
print(f"{'arm':<18}" + "".join(f"{x:>8.1f}s" for x in t))
print("-" * (18 + 9 * len(t)))
for name, v in rows:
    if not v:
        print(f"{name:<18}  (absent)")
        continue
    print(f"{name:<18}" + "".join(f"{x:>9.3f}" for x in v[:6]))
    # error per second: flat => a constant speed error integrated over time
    print(f"{'  -> m/s':<18}" + "".join(f"{x / ti:>9.3f}"
                                        for x, ti in zip(v[:6], t)))
    # ratio to the first step: 1,2,3,4,5,6 would be exactly linear in t
    print(f"{'  -> x step1':<18}" + "".join(f"{x / v[0]:>9.2f}"
                                            for x in v[:6]))
print("\nflat 'm/s' row  = error is a CONSTANT INITIAL SPEED ERROR integrated over t")
print("rising 'm/s'    = compounding wrong decisions (superlinear)")
print("'x step1' close to 1,2,3,4,5,6 = perfectly linear in t")