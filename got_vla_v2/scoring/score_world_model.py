"""
score_world_model.py
--------------------
World Model Score: WorldVLA의 World Model로 미래 이미지를 생성하고
현재 이미지와의 픽셀 차이로 행동 후보를 평가한다.

FD Score와의 차이:
  FD Score:     시뮬레이터 직접 실행 → 시뮬레이터 전용
  WM Score:     모델 추론으로 미래 예측 → 실제 로봇 배포 가능

RynnVLA-002(WorldVLA)는 Action Model과 World Model을 단일 모델로 통합.
--unmask_image_logits 플래그로 이미지 생성 모드를 활성화한다.
"""
from __future__ import annotations

import torch
import numpy as np
from PIL import Image
from transformers import GenerationConfig

IMG_START_TOKEN = 8197
IMG_END_TOKEN = 8196


def world_model_score(
    candidates: list,
    current_img: Image.Image,
    wrist_img,
    world_model,
    item_processor,
    device,
    verbose: bool = False,
    **kwargs,
) -> list:
    """
    World Model로 각 후보의 미래 이미지를 생성하고
    현재 이미지와의 픽셀 차이(MAE)를 score로 사용한다.
    변화가 클수록(MAE 높을수록) 더 적극적인 행동 → 높은 score.
    """
    if not candidates:
        return candidates

    # generate_img 호출 시 init_input_ids 충돌 방지
    saved_init = getattr(world_model, "init_input_ids", None)
    world_model.init_input_ids = None

    if verbose:
        print(f"  [WM Score] {len(candidates)}개 후보 평가 시작")

    for i, cand in enumerate(candidates):
        if not cand.actions:
            cand.score = -float("inf")
            continue

        conv = {
            "conversations": [{
                "from": "human",
                "value": "Generate the image based on the current image and the action."
                         "<|image|><|image|><|action|>",
            }],
            "image": [current_img, wrist_img if wrist_img else current_img],
            "action": [cand.actions[0]],
        }

        try:
            tokens = item_processor.process_item(conv, training_mode=False)
            input_ids = torch.tensor(
                tokens, dtype=torch.int64, device=device
            ).unsqueeze(0)

            generation_config = GenerationConfig(
                max_new_tokens=700,
                max_length=world_model.config.max_position_embeddings,
                temperature=1.0, top_k=None, do_sample=False,
                eos_token_id=[8710],
            )

            with torch.no_grad():
                g_image_tokens = world_model.generate_img(
                    input_ids, generation_config)

            tokens_sequence = g_image_tokens[0]
            start_indices = torch.where(tokens_sequence == IMG_START_TOKEN)[0]
            end_indices = torch.where(tokens_sequence == IMG_END_TOKEN)[0]

            pred_img = None
            if len(start_indices) >= 2 and len(end_indices) >= 2:
                front_tokens = tokens_sequence[
                    start_indices[0]:end_indices[0] + 1
                ].cpu().tolist()
                try:
                    pred_img = item_processor.decode_image(front_tokens)
                except Exception as de:
                    if verbose:
                        print(f"  [WM Score] 후보{i+1}: decode 실패(2장): {de}")

            elif len(start_indices) >= 1 and len(end_indices) >= 1:
                front_tokens = tokens_sequence[
                    start_indices[0]:end_indices[0] + 1
                ].cpu().tolist()
                try:
                    pred_img = item_processor.decode_image(front_tokens)
                    if verbose:
                        print(f"  [WM Score] 후보{i+1}: 1장 생성")
                except Exception as de:
                    if verbose:
                        print(f"  [WM Score] 후보{i+1}: decode 실패(1장): {de}")

            if pred_img is not None:
                arr_cur = np.array(current_img.convert("RGB")).astype(np.float32)
                arr_pred = np.array(pred_img.convert("RGB")).astype(np.float32)
                # 픽셀 변화량(MAE): 클수록 더 많이 움직임 → 높은 score
                cand.score = float(np.mean(np.abs(arr_cur - arr_pred)) / 255.0)
            else:
                cand.score = -1.0

            if verbose:
                print(f"  [WM Score] 후보{i+1}: score={cand.score:.4f}")

        except Exception as e:
            if verbose:
                print(f"  [WM Score] 후보{i+1} 예외: {e}")
            cand.score = -float("inf")

    if verbose:
        scores = [f"{c.score:.4f}" for c in candidates]
        print(f"  [WM Score] 점수: {scores}")

    world_model.init_input_ids = saved_init
    return candidates
