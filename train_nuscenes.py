"""
Fine-tune the RynnVLA-002 / WorldVLA action model on nuScenes trajectory
planning: CAM_FRONT image -> future ego waypoints, trained with the existing
discrete action-token CE loss (`c_loss`) + continuous action-head L1 loss
(`loss_ct`).

Pipeline reuse
--------------
This subclasses the repo's FSDP training solver
(`PretrainSolverBase_ck_action_head`) and only swaps the domain-specific parts:
    * _model_func            : load the ckpt with action_dim=2, time_horizon=6
    * __init__ (item proc)   : use FlexARItemProcessor_Action_NuScenes (2-D norm)
    * _dataset_func_wo_processed : use NuScenesFinetuneConversation (json records)

Prerequisite: run data/preprocess_nuscenes.py first to produce the json records
and nuscenes_norm.json.

Launch (single node, N GPUs)
----------------------------
    torchrun --nproc_per_node=<N> train_nuscenes.py \
        --resume_path ~/ckpts/VLA_model_256/libero_spatial \
        --tokenizer_path ~/ckpts/models--Alpha-VLLM--Lumina-mGPT-7B-768 \
        --data_config_train  ./data/nuscenes_records/nuscenes_v1.0-mini_train.json \
        --data_config_val_ind ./data/nuscenes_records/nuscenes_v1.0-mini_val.json \
        --data_config_val_ood ./data/nuscenes_records/nuscenes_v1.0-mini_val.json \
        --norm_path ./data/nuscenes_records/nuscenes_norm_v1.0-trainval.json \
        --output_dir ./output/nuscenes_mini \
        --trainable head \
        --batch_size 1 --accum_iter 8 --epochs 20 --lr 1e-4 \
        --precision bf16 --checkpointing --ft true

Notes
-----
* v1.0-mini is ~340 records -> use it only as a *smoke test* of the loop
  (does loss_ct / c_loss go down, does a checkpoint save). Switch to
  v1.0-trainval + `--trainable full` for a model that actually learns.
* `--trainable head` freezes the 7B backbone and trains only the action head,
  the token embeddings and lm_head (~571M trainable). lm_head and embed_tokens
  have to stay trainable: c_loss trains the discrete action tokens through them,
  and freezing them leaves the model unable to emit a well-formed action group.
  Budget roughly: 12.9GB frozen backbone (bf16) + 2.3GB trainable (fp32) +
  4.6GB AdamW + 2.3GB grads ~= 22GB, so a 40GB A100 is comfortable and a 24GB
  card is borderline. Full FT (`--trainable full`) keeps every param in fp32
  (28GB of weights alone) and wants far more.
* `ignore_mismatched_sizes=True` re-initializes the action head because we
  change its shape from (7 dim x 5 steps) to (2 dim x 6 steps).
* --batch_size 1 for the smoke test (raise --accum_iter for a larger effective
  batch). Both sides of the action-head L1 loss are (B * time_horizon,
  action_dim) -- ActionHead.forward ends in `actions.reshape(-1, action_dim)`
  and `get_action_hs_label` returns one row per action group -- and both are
  ordered batch-major, so B>1 is shape-correct in principle. The hazard is
  ActionHead.forward silently `continue`-ing over any batch row with <2
  occurrences of token 10004 while get_action_hs_label still counts that row's
  groups from the labels: one skipped row misaligns the two tensors. Every
  nuScenes sample emits exactly `time_horizon` action groups, so no row is
  skipped and B>1 should be safe -- but that is untested, so keep B=1 until the
  mini run is green.
"""

import contextlib
import gc
from pathlib import Path
import types

from fairscale.nn.model_parallel import initialize as fs_init
import torch

from model import ChameleonXLLMXForConditionalGeneration_ck_action_head, ChameleonXLLMXConfig
from xllmx.solvers.pretrain.pretrain_ck_action_head import PretrainSolverBase_ck_action_head
import xllmx.util.misc as misc
from data.dataset_nuscenes import NuScenesFinetuneConversation
from data.item_processor import FlexARItemProcessor_Action_NuScenes


class NuScenesTrainSolver(PretrainSolverBase_ck_action_head):

    @classmethod
    def get_args_parser(cls):
        parser = super().get_args_parser()
        # loading / model
        parser.add_argument("--resolution", type=int, default=256)
        parser.add_argument("--tokenizer_path", type=str, required=True)
        parser.add_argument("--norm_path", type=str, default=None,
                            help="nuscenes_norm.json (waypoint min/max) from preprocess")
        parser.add_argument("--max_seq_len", type=int, default=4096)
        parser.add_argument("--mask_image_logits", default=True)
        parser.add_argument("--dropout", type=float, default=0.0)
        parser.add_argument("--z_loss_weight", type=float, default=0.0)
        parser.add_argument("--action_dim", type=int, default=2)
        parser.add_argument("--time_horizon", type=int, default=6)
        parser.add_argument("--trainable", choices=["full", "head"], default="head",
                            help="head = freeze backbone, train action_head + embeddings + lm_head")
        # flags the base solver reads but does not declare in its own parser
        parser.add_argument("--preprocess", type=str, default="false", choices=["true", "false"])
        parser.add_argument("--with_state", type=lambda x: x == "true", default=False)
        parser.add_argument("--with_wrist", type=lambda x: x == "true", default=False)
        parser.add_argument("--with_action", type=lambda x: x == "true", default=True)
        parser.add_argument("--with_world_model", type=lambda x: x == "true", default=False)
        return parser

    def __init__(self, args):
        super().__init__(args)
        # The base __init__ builds a LIBERO 7-DoF action processor; replace it
        # with the 2-D waypoint processor for the on-the-fly tokenization path.
        if args.preprocess == "false":
            self.item_processor_ar = FlexARItemProcessor_Action_NuScenes(
                tokenizer=args.tokenizer_path,
                target_size=args.resolution,
                norm_path=args.norm_path,
            )

        # The base __init__ also eagerly builds `item_processor` and
        # `item_processor_action`, each of which loads its own copy of the
        # Chameleon VQGAN onto the GPU (~280MB of weights apiece). Only the
        # image-generation and world-model paths use them; run_with_eval_awm
        # tokenizes exclusively through item_processor_ar. Drop them so the
        # dead copies do not eat into the action-head training budget.
        for attr in ("item_processor", "item_processor_action"):
            if hasattr(self, attr):
                delattr(self, attr)
        gc.collect()
        torch.cuda.empty_cache()

    def _model_func(self, init_from: str):
        from_kwargs = dict(
            action_dim=self.args.action_dim,
            time_horizon=self.args.time_horizon,
            max_position_embeddings=self.args.max_seq_len,
            mask_image_logits=self.args.mask_image_logits,
            dropout=self.args.dropout,
            z_loss_weight=self.args.z_loss_weight,
        )
        # Only rank 0 materialises real weights on host RAM. Under --trainable
        # full every param is later promoted to an fp32 master copy (~28GB for
        # 7B); doing that on all N ranks at once (N*28GB) OOM-kills a 62GB host
        # (SIGKILL/-9) even with low_cpu_mem_usage. setup_fsdp_sync wraps with
        # sync_module_states=True and a to_empty param_init_fn, so the other
        # ranks only need an empty (meta) model and receive rank 0's weights over
        # the broadcast -> host peak drops from N*28GB to ~28GB.
        if self.dp_rank == 0:
            # NOTE: do NOT pass low_cpu_mem_usage=True here. It loads the model on
            # meta and only fills params present in the checkpoint; the reshaped
            # action head (ignore_mismatched_sizes) is absent from the ckpt and is
            # left on meta. rank 0 has param_init_fn=None, so FSDP then calls
            # module.reset_parameters() to materialise it -- which nn.MultiheadAttention
            # inside ActionHead does not implement (only _reset_parameters), crashing.
            # Without the flag from_pretrained runs _init_weights, so the action head
            # is real. Only rank 0 loads now, so the host-RAM saving is unneeded.
            model = ChameleonXLLMXForConditionalGeneration_ck_action_head.from_pretrained(
                init_from,
                **from_kwargs,
                torch_dtype=torch.bfloat16,
                ignore_mismatched_sizes=True,  # action head reshaped -> re-init
            )
        else:
            # include_buffers=False keeps buffers real: `inv_freq` is a
            # persistent=False rotary buffer, so sync_module_states does NOT
            # broadcast it. Recomputing it locally from config on every rank
            # (params stay on meta, filled by the broadcast) avoids NaN rotary.
            from accelerate import init_empty_weights
            config = ChameleonXLLMXConfig.from_pretrained(init_from, **from_kwargs)
            with init_empty_weights(include_buffers=False):
                model = ChameleonXLLMXForConditionalGeneration_ck_action_head(config)

        # The model's own get_fsdp_wrap_module_list leaves model.norm (the final
        # RMSNorm) unwrapped, so it lands in FSDP's root unit. Under --trainable
        # head it is also frozen, and FSDP's FULL_STATE_DICT save then asserts
        # "FSDP assumes model.norm.weight is in the state_dict" when ckpt.save
        # calls model.state_dict(). Wrapping it as its own unit leaves the root
        # unit with no parameters of its own, which sidesteps that save path.
        # Training itself (forward/backward/optimizer.step) is unaffected --
        # only the checkpoint hook was failing.
        def get_fsdp_wrap_module_list(self):
            modules = [*list(self.model.layers), self.lm_head, self.model.embed_tokens,
                       self.action_head, self.model.norm]
            if hasattr(self.model, "vqmodel"):  # may be deleted
                modules.append(self.model.vqmodel)
            return modules
        model.get_fsdp_wrap_module_list = types.MethodType(get_fsdp_wrap_module_list, model)

        if self.args.trainable == "head":
            def get_trainable_params(self):
                keys = ("action_head", "embed_tokens", "lm_head")
                return {n: p for n, p in self.named_parameters()
                        if any(k in n for k in keys)}
            model.get_trainable_params = types.MethodType(get_trainable_params, model)
        # trainable == "full": no get_trainable_params -> base makes all params trainable

        return model, None

    def _make_and_save_starting_point(self, save_path: str):
        pass

    def _item_processor_func(self):
        # only used on the preprocess=='true' (pre-tokenized pkl) path
        from data.pre_tokenize_action import ItemProcessor
        return ItemProcessor(target_size=self.args.resolution, tokenizer=self.args.tokenizer_path)

    def _dataset_func_wo_processed(self):
        # `--with_state` has to reach the dataset, not just the training loop:
        # the loop only forwards a state the dataset actually emitted, so
        # dropping it here trains a silently stateless model that still looks
        # like the ego-status arm in the logs.
        kw = dict(resolution=self.args.resolution, with_state=self.args.with_state)
        train = NuScenesFinetuneConversation(self.args.data_config_train, **kw)
        val_ind = NuScenesFinetuneConversation(self.args.data_config_val_ind, **kw)
        val_ood = NuScenesFinetuneConversation(self.args.data_config_val_ood, **kw)
        return train, val_ind, val_ood

    # ------------------------------------------------------------------
    # Validation
    #
    # The base class' val_one_epoch_awm_{ind,ood} unpack 5 values from a
    # forward that returns 7 whenever output_hidden_states is passed (see
    # ChameleonXLLMXForConditionalGeneration_ck_action_head.forward), so they
    # raise before the first val batch finishes. They also run the forward
    # without no_grad and then call clip_grad_norm_ on an eval model. Both are
    # replaced here rather than patched in the base class, which the LIBERO
    # entrypoints still share.
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _val_one_epoch(self, dataloader, epoch, log_writer=None, tag="val"):
        self.model.eval()

        metric_logger = misc.MetricLogger(delimiter="  ")
        header = "Val[{}] {}".format(tag, epoch)

        for data_iter_step, batch_data in enumerate(
            metric_logger.log_every(
                dataloader,
                10,
                header,
                0,
                self.args.batch_size * fs_init.get_data_parallel_world_size(),
            )
        ):
            conversations, images, actions, states = batch_data
            examples, labels = [], []
            for conv, img, act, sta in zip(conversations, images, actions, states):
                item = {"conversations": conv, "image": img, "action": act}
                if self.args.with_state:
                    item["state"] = sta
                tokens, labels_ = self.item_processor_ar.process_item(item, training_mode=True)
                examples.append(tokens)
                labels.append(labels_)

            with {
                "bf16": torch.cuda.amp.autocast(dtype=torch.bfloat16),
                "fp16": torch.cuda.amp.autocast(dtype=torch.float16),
                "fp32": contextlib.nullcontext(),
                "tf32": contextlib.nullcontext(),
            }[self.args.precision]:
                (
                    c_loss,
                    additional_loss_dict,
                    logits,
                    hidden_states,
                    labels_c,
                    predicted_actions,
                    loss_ct,
                ) = self.model(
                    input_ids=examples,
                    labels=labels,
                    output_hidden_states=True,
                    training=True,
                    att_mask=True,
                )

            loss = c_loss + self.args.loss_ct_weights * loss_ct
            for add_loss, weight in additional_loss_dict.values():
                loss = loss + add_loss * weight

            accuracies_action, accuracies_image, l1_loss = self.calculate_accuracies(labels_c, logits)
            for i in range(len(accuracies_action)):
                metric_logger.update(**{f"acc_action_{i}": accuracies_action[i]})
                metric_logger.update(**{f"l1_loss_action_{i}": l1_loss[i]})
            for i in range(len(accuracies_image)):
                metric_logger.update(**{f"acc_image_{i}": accuracies_image[i]})

            metric_logger.update(closs=c_loss.item())
            metric_logger.update(loss_ct=loss_ct.item())
            metric_logger.update(loss=loss.item())
            metric_logger.update(**{key: val[0].item() for key, val in additional_loss_dict.items()})

            if self.global_rank == 0 and log_writer is not None:
                for metric_name, metric in metric_logger.meters.items():
                    log_writer.add_scalar(
                        f"{tag}_{metric_name}", metric.value, data_iter_step + len(dataloader) * epoch
                    )

        metric_logger.synchronize_between_processes()
        self.logger.info(f"Averaged {tag} stats:\n{metric_logger}")
        self.model.train(True)
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    def val_one_epoch_awm_ind(self, epoch, start_iter, log_writer=None, metric_logger=None):
        return self._val_one_epoch(self.dataloader_val_ind, epoch, log_writer, tag="val_ind")

    def val_one_epoch_awm_ood(self, epoch, start_iter, log_writer=None, metric_logger=None):
        return self._val_one_epoch(self.dataloader_val_ood, epoch, log_writer, tag="val_ood")


def main():
    parser = NuScenesTrainSolver.get_args_parser()
    args = parser.parse_args()
    # PretrainSolverBase_ck_action_head.__init__ attaches a logging.FileHandler at
    # output_dir/{common,rank-N}.log before it mkdir's output_dir, so a fresh
    # --output_dir raises FileNotFoundError before the run starts. exist_ok makes
    # the race between ranks harmless.
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    solver = NuScenesTrainSolver(args)
    solver.run_with_eval_awm()


if __name__ == "__main__":
    main()
