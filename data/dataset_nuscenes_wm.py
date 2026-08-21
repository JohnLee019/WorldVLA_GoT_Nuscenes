"""
NuScenes WORLD-MODEL dataset: (current frame + action) -> future frame.

Yields the same 4-tuple as the action dataset
    (conversations, images, action, state)
so it is a drop-in for the on-the-fly tokenization path in
train_one_epoch_awm (the batch handler builds
{"conversations", "image", "action"} and calls item_processor.process_item).

The conversation puts the future frame in the ASSISTANT turn, so its VQGAN image
tokens become prediction targets. This only actually contributes to the loss
when the model is loaded with mask_image_logits=False (image logits unmasked) --
train_nuscenes_wm.py sets that. With masking on (the action-model default) the
image-token loss is suppressed and the model learns nothing, so the flag is not
optional.

    conversations : [{human: prompt + <|image|>(current) + <|action|>*k},
                     {gpt:   <|image|>(future)}]
    images        : [current_frame, future_frame]   (order == <|image|> order)
    action        : k waypoints [x, y], ego frame, metres
    state         : []

Records come from data/preprocess_nuscenes_wm.py.
"""

import json
import logging

from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# kept explicit so the WM prompt is stable across preprocessing/eval
WM_PROMPT = ("Predict the future front-camera image after the ego vehicle "
             "follows the planned waypoints.")


class NuScenesWorldModelConversation(Dataset):
    def __init__(self, config_path, resolution=256, prompt=None, **kwargs):
        # kwargs swallows with_state/with_wrist/... passed positionally by the solver
        self.resolution = resolution
        self.prompt = prompt or WM_PROMPT

        records_json = self._resolve_json(config_path)
        logger.info(f"[NuScenes-WM] loading records from {records_json}")
        with open(records_json, "r") as f:
            self.data_list = json.load(f)
        logger.info(f"[NuScenes-WM] {len(self.data_list)} records")

    @staticmethod
    def _resolve_json(config_path):
        if config_path.endswith(".json"):
            return config_path
        import yaml
        with open(config_path, "r") as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
        return cfg["META"]["records_json"]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        rec = self.data_list[idx]

        cur_img = Image.open(rec["cur_image"]).convert("RGB")
        future_img = Image.open(rec["future_image"]).convert("RGB")

        action = [list(map(float, wp)) for wp in rec["action"]]  # k x [x, y]

        conversations = [
            {
                "from": "human",
                # current image first, then the action tokens that condition it
                "value": self.prompt + "<|image|>" + "<|action|>" * len(action),
            },
            {
                "from": "gpt",
                # the future frame is the generation target
                "value": "<|image|>",
            },
        ]

        # media are matched to <|image|> tokens in order of appearance:
        # human's <|image|> -> images[0] (current), gpt's -> images[1] (future)
        images = [cur_img, future_img]
        state = []
        return conversations, images, action, state


if __name__ == "__main__":
    import sys
    ds = NuScenesWorldModelConversation(sys.argv[1])
    print("len:", len(ds))
    conv, imgs, act, st = ds[0]
    print("conversations:", conv)
    print("num images:", len(imgs), "cur:", imgs[0].size, "future:", imgs[1].size)
    print("action:", act)
