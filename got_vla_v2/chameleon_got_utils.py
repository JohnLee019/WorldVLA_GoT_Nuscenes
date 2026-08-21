"""
chameleon_got_utils.py
----------------------
GoT 추론을 위한 행동 생성 유틸리티.

기존 Chameleon_utils.py 대비 변경사항:
  - do_sample / temperature 인자 추가 (후보별 다양성 제어)
  - top_k 제거 (temperature와 역할 중복, LLaMA 가이드라인 참조)
  - top_p=0.95 추가 (nucleus sampling, do_sample=True 시 적용)

temperature 설정:
  i=0: 1.0 (greedy, exploitation — 가장 확률 높은 행동)
  i=1: 1.2 (약한 탐색, ±20% 다양성)
  i=2: 1.4 (강한 탐색, ±40% 다양성, 1.5 초과 시 품질 저하)
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from transformers import GenerationConfig


def parse_his_type(his_type: str) -> dict:
    try:
        parts = his_type.split("_")
        if len(parts) < 5 or parts[0] != "his" or parts[-1] != "state":
            return {"his": 1, "views": ["front", "wrist"], "with_state": True}
        his = int(parts[1])
        state_prefix = parts[-2]
        with_state = state_prefix == "w"
        views = parts[2:-2]
        return {"his": his, "views": views, "with_state": with_state}
    except Exception:
        return {"his": 1, "views": ["front", "wrist"], "with_state": True}


def _build_image_list(
    his_type_dict: dict,
    his_img: List[Image.Image],
    his_wrist_img: List[Image.Image],
    cur_img: Image.Image,
    cur_wrist_img: Optional[Image.Image],
) -> List[Image.Image]:
    his_n = his_type_dict["his"]
    n_views = len(his_type_dict["views"])

    if his_n == 2 and n_views == 3:
        img_c = his_img[-1:] + his_wrist_img[-1:] + [cur_img]
        if cur_wrist_img is not None:
            img_c.append(cur_wrist_img)
    elif his_n == 1 and n_views == 3:
        img_c = [cur_img]
        if cur_wrist_img is not None:
            img_c.append(cur_wrist_img)
    elif his_n == 2 and n_views == 2:
        img_c = his_img[-1:] + [cur_img]
    else:
        img_c = [cur_img]

    return img_c


def get_action_for_got(
    model,
    cur_img: Image.Image,
    cur_wrist_img: Optional[Image.Image],
    task_description: str,
    item_processor,
    his_img: List[Image.Image],
    his_wrist_img: List[Image.Image],
    cur_state: np.ndarray,
    his_action: List[np.ndarray],
    his_type: str,
    action_steps: int,
    do_sample: bool = False,
    temperature: float = 1.0,
    device: Optional[torch.device] = None,
) -> list:
    """
    GoT용 행동 생성 함수.

    Returns:
        list of torch.Tensor, 각 shape (7,). 실패 시 [] 반환.
    """
    if device is None:
        device = next(model.parameters()).device

    try:
        his_type_dict = parse_his_type(his_type)
        img_c = _build_image_list(
            his_type_dict, his_img, his_wrist_img, cur_img, cur_wrist_img
        )

        if his_type_dict["with_state"]:
            human_val = (
                f"What action should the robot take to {task_description}?"
                + "<|state|>" * 1
                + "<|image|>" * len(img_c)
            )
        else:
            human_val = (
                f"What action should the robot take to {task_description}?"
                + "<|image|>" * len(img_c)
            )

        conv = {
            "conversations": [{"from": "human", "value": human_val}],
            "image": img_c,
            "action": [],
        }
        if his_type_dict["with_state"]:
            conv["state"] = cur_state

        tokens = item_processor.process_item(conv, training_mode=False)

        generation_config = GenerationConfig(
            max_new_tokens=action_steps * 12,
            max_length=model.config.max_position_embeddings,
            temperature=temperature if do_sample else 1.0,
            top_k=None,                          # top_k 제거
            top_p=0.95 if do_sample else None,   # nucleus sampling
            do_sample=do_sample,
            eos_token_id=[8710],
        )

        input_ids = torch.tensor(
            tokens, dtype=torch.int64, device=device
        ).unsqueeze(0)

        with torch.no_grad():
            dis_action = model.generate_dis_ma(input_ids, generation_config)

        return dis_action

    except Exception as e:
        print(f"[chameleon_got_utils] get_action_for_got 실패: {e}")
        return []
