"""Driving-domain Graph-of-Thought building blocks (open-loop nuScenes).

Kept separate from `got_vla_v2/`, which is LIBERO closed-loop only (it drives
`env.step` and its three scores are simulator-bound). Driving GoT is open-loop
trajectory planning, so it needs its own segment-generation primitive and,
later, its own driving-appropriate scoring.
"""
