"""
score_heuristic.py
------------------
Heuristic Score: 현재 이미지의 픽셀 분산 기반.
후보 간 구분 불가 (모두 동일 score) → 사실상 greedy 선택과 동일.
GoT 구조 효과만 측정하는 기준선 조건(got_no_score)에 사용.
"""
from __future__ import annotations
from typing import Optional
import numpy as np
from PIL import Image


def heuristic_score(candidates: list, current_img: Image.Image) -> list:
    """모든 후보에 동일한 픽셀 분산 점수를 부여한다."""
    if current_img is None:
        for c in candidates:
            c.score = 0.5
        return candidates

    arr = np.array(current_img.convert("RGB")).astype(np.float32) / 255.0
    H, W = arr.shape[:2]
    h0, h1 = int(H * 0.3), int(H * 0.7)
    w0, w1 = int(W * 0.3), int(W * 0.7)
    variance = float(np.var(arr[h0:h1, w0:w1]))
    score = min(1.0, variance / 0.05)

    for c in candidates:
        c.score = score
        c.score_collision = score
        c.score_consistency = 0.5

    return candidates
