"""
Train the nuScenes WORLD MODEL: (current frame + action) -> future frame.

This reuses the action model's FSDP training machinery unchanged and swaps only
what the world-model objective requires:

  * Dataset -> NuScenesWorldModelConversation (future frame is the target).
  * mask_image_logits = False -> image-token logits are UNMASKED, so the future
    frame's VQGAN tokens actually contribute to the loss. (With the action
    model's default masking they don't, and the WM would learn nothing.)
  * Training loss = c_loss only. The action model's train loop adds the L1
    action-head loss and SKIPS any batch whose loss_ct == 0; a WM batch has no
    action target, so loss_ct is always 0 and every batch would be skipped.
    Here we call the model without output_hidden_states (no action-head branch)
    and optimise the plain cross-entropy over the predicted image tokens.
  * trainable defaults to "full": generating a new domain's frames needs the
    backbone, not just the embeddings/head.

Everything else -- FSDP FULL_SHARD, activation checkpointing, bf16, accum,
checkpoint saving -- is inherited from NuScenesTrainSolver / the base solver.

Usage (3x RTX 4090, FSDP shards the 7B across the cards)
-------------------------------------------------------
    torchrun --nproc_per_node=3 train_nuscenes_wm.py \
        --init_from ../ckpts/<lumina-mgpt-7b> \
        --tokenizer_path ../ckpts/<lumina-mgpt-7b> \
        --norm_path ./data/nuscenes_records/nuscenes_norm_v1.0-trainval.json \
        --data_config_train    ./data/nuscenes_wm_records/nuscenes_wm_v1.0-mini_train.json \
        --data_config_val_ind  ./data/nuscenes_wm_records/nuscenes_wm_v1.0-mini_val.json \
        --data_config_val_ood  ./data/nuscenes_wm_records/nuscenes_wm_v1.0-mini_val.json \
        --output_dir ./output/nuscenes_wm \
        --batch_size 1 --accum_iter 8 --epochs 20 --lr 1e-4 \
        --precision bf16 --checkpointing --ft true --trainable full

Smoke-test FIRST on ONE card (`--nproc_per_node=1`, a few iters) to confirm the
image-token loss goes down and a checkpoint saves, exactly as train_nuscenes.py
recommends -- multi-GPU image-gen memory is heavier than action training
(two 256px frames per sample) so watch for OOM and lower --max_seq_len /
--resolution if needed.
"""
import contextlib
import math
import sys

from fairscale.nn.model_parallel import initialize as fs_init
import torch

import xllmx.util as util
import xllmx.util.lr_sched as lr_sched
import xllmx.util.misc as misc

from train_nuscenes import NuScenesTrainSolver
from data.dataset_nuscenes_wm import NuScenesWorldModelConversation


class NuScenesWorldModelSolver(NuScenesTrainSolver):

    @classmethod
    def get_args_parser(cls):
        parser = super().get_args_parser()
        # world-model defaults: unmask image logits so the future frame is
        # actually learned, and train the backbone (not just head).
        parser.set_defaults(mask_image_logits=False, trainable="full")
        return parser

    # -- data: future-frame world-model records ------------------------------
    def _dataset_func_wo_processed(self):
        train = NuScenesWorldModelConversation(self.args.data_config_train, resolution=self.args.resolution)
        val_ind = NuScenesWorldModelConversation(self.args.data_config_val_ind, resolution=self.args.resolution)
        val_ood = NuScenesWorldModelConversation(self.args.data_config_val_ood, resolution=self.args.resolution)
        return train, val_ind, val_ood

    # -- shared tokenization (mirrors the base train loop's batch handling) --
    def _tokenize_batch(self, batch_data):
        if len(batch_data) == 2:
            examples, labels = batch_data
            return list(examples), list(labels)
        conversations, images, actions, _states = batch_data
        examples, labels = [], []
        for conv, img, act in zip(conversations, images, actions):
            conversation = {"conversations": conv, "image": img, "action": act}
            tokens, labels_ = self.item_processor_ar.process_item(conversation, training_mode=True)
            examples.append(tokens)
            labels.append(labels_)
        return examples, labels

    # -- training: plain image-token cross-entropy, no action head -----------
    def train_one_epoch_awm(self, epoch, start_iter, log_writer=None, metric_logger=None):
        self.model.train(True)
        if metric_logger is None:
            metric_logger = misc.MetricLogger(delimiter="  ")
            metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))

        header = "Epoch: [{}] (WM)".format(epoch)
        print_freq = 10
        accum_iter = self.args.accum_iter
        accum_counter = 0

        self.optimizer.zero_grad()
        for data_iter_step, batch_data in enumerate(
            metric_logger.log_every(
                self.dataloader_train, print_freq, header, start_iter,
                self.args.batch_size * fs_init.get_data_parallel_world_size(),
            ),
            start=start_iter,
        ):
            accum_counter = (accum_counter + 1) % accum_iter
            is_gradient_accumulation_boundary = accum_counter == 0

            examples, labels = self._tokenize_batch(batch_data)

            if is_gradient_accumulation_boundary or data_iter_step == start_iter:
                lr_sched.adjust_learning_rate_epoch(
                    self.optimizer, data_iter_step / len(self.dataloader_train) + epoch, self.args
                )

            with {
                "bf16": torch.cuda.amp.autocast(dtype=torch.bfloat16),
                "fp16": torch.cuda.amp.autocast(dtype=torch.float16),
                "fp32": contextlib.nullcontext(),
                "tf32": contextlib.nullcontext(),
            }[self.args.precision]:
                # no output_hidden_states -> forward returns (c_loss, add_loss_dict);
                # the action-head branch (and its loss_ct skip) is never entered.
                c_loss, additional_loss_dict = self.model(
                    input_ids=examples, labels=labels, training=True, att_mask=True
                )

            loss = c_loss
            for add_loss, weight in additional_loss_dict.values():
                loss = loss + add_loss * weight
            loss_value = loss.item()
            c_loss_value = c_loss.item()

            if not math.isfinite(loss_value):
                self.logger.error("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

            effective_loss = loss / accum_iter
            with (
                self.model.no_sync()
                if self.args.data_parallel in ["sdp", "hsdp"] and not is_gradient_accumulation_boundary
                else contextlib.nullcontext()
            ):
                effective_loss.backward()

            if is_gradient_accumulation_boundary:
                grad_norm = self.model.clip_grad_norm_(max_norm=self.args.clip_grad)
                metric_logger.update(grad_norm=grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

            torch.cuda.synchronize()

            metric_logger.update(closs=c_loss_value)
            metric_logger.update(**{key: val[0].item() for key, val in additional_loss_dict.items()})
            metric_logger.update(lr=self.optimizer.param_groups[0]["lr"])

            n_update_per_save = self.args.save_iteration_interval // accum_iter
            if (
                is_gradient_accumulation_boundary and ((data_iter_step + 1) // accum_iter) % n_update_per_save == 0
            ) or (data_iter_step + 1 == accum_iter and epoch == 0):
                util.ckpt.save(
                    self.args.output_dir, self.global_rank == 0, self.model, self.optimizer,
                    self.tokenizer, self.args, epoch=epoch, iteration=data_iter_step,
                    additional_rank_specific={"metric_logger": metric_logger},
                    max_keep=self.args.ckpt_max_keep,
                )

        metric_logger.synchronize_between_processes()
        self.logger.info(f"Averaged stats:\n{metric_logger}")
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    # -- validation: mean image-token loss on the future frame ---------------
    @torch.no_grad()
    def _val_wm(self, dataloader, epoch, tag):
        self.model.eval()
        metric_logger = misc.MetricLogger(delimiter="  ")
        header = "Val[{}] {} (WM)".format(tag, epoch)
        for batch_data in metric_logger.log_every(dataloader, 10, header, 0, self.args.batch_size):
            examples, labels = self._tokenize_batch(batch_data)
            with {
                "bf16": torch.cuda.amp.autocast(dtype=torch.bfloat16),
                "fp16": torch.cuda.amp.autocast(dtype=torch.float16),
                "fp32": contextlib.nullcontext(),
                "tf32": contextlib.nullcontext(),
            }[self.args.precision]:
                c_loss, _ = self.model(input_ids=examples, labels=labels, training=True, att_mask=True)
            metric_logger.update(wm_closs=c_loss.item())
        self.model.train(True)
        metric_logger.synchronize_between_processes()
        self.logger.info(f"[{tag}] {metric_logger}")
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    def val_one_epoch_awm_ind(self, epoch, start_iter, log_writer=None, metric_logger=None):
        return self._val_wm(self.dataloader_val_ind, epoch, "val_ind")

    def val_one_epoch_awm_ood(self, epoch, start_iter, log_writer=None, metric_logger=None):
        return self._val_wm(self.dataloader_val_ood, epoch, "val_ood")


def main():
    parser = NuScenesWorldModelSolver.get_args_parser()
    args = parser.parse_args()
    solver = NuScenesWorldModelSolver(args)
    solver.run_with_eval_awm()


if __name__ == "__main__":
    main()
