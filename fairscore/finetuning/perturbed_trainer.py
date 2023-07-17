#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from transformers import Trainer
from transformers.deepspeed import deepspeed_init
from transformers.file_utils import is_torch_tpu_available
from transformers.trainer_pt_utils import (
    IterableDatasetShard,
    find_batch_size,
    nested_concat,
    nested_numpify,
    nested_truncate,
)
from transformers.trainer_utils import (
    EvalLoopOutput,
    denumpify_detensorize,
    has_length,
)
from transformers.utils import logging
from typing import NamedTuple, Tuple, Union

import numpy as np
import torch

logger = logging.get_logger(__name__)

if is_torch_tpu_available():
    import torch_xla.distributed.parallel_loader as pl
    import torch_xla.core.xla_model as xm

class FairScoreEvalPrediction(NamedTuple):
    """
        Evaluation output (always contains labels), to be used to compute metrics.
        Parameters:
            predictions (`np.ndarray`): Predictions of the model.
            label_ids (`np.ndarray`): Targets to be matched.
            ex_ids (`np.ndarray`): Indices for each example to load perturbation data.
    """
    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    label_ids: Union[np.ndarray, Tuple[np.ndarray]]
    ex_ids: Union[np.ndarray, Tuple[np.ndarray]]
    indices: Union[np.ndarray, Tuple[np.ndarray]]
    is_perturbed: Union[np.ndarray, Tuple[np.ndarray]]

class FairscoreTrainerEvalOnly(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert not self.args.use_legacy_prediction_loop, "Trainer only written for default, non-legacy prediction loop"

    def evaluation_loop(
        self, dataloader, description, prediction_loss_only=None, ignore_keys=None, metric_key_prefix: str = "eval",
    ):
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        # if eval is called w/o train init deepspeed here
        if args.deepspeed and not self.deepspeed:

            # XXX: eval doesn't have `resume_from_checkpoint` arg but we should be able to do eval
            # from the checkpoint eventually
            deepspeed_engine, _, _ = deepspeed_init(
                self, num_training_steps=0, resume_from_checkpoint=None, inference=True
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine

        model = self._wrap_model(self.model, training=False)

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = dataloader.batch_size

        logger.info(f"***** Running {description} *****")
        if has_length(dataloader.dataset):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = dataloader.dataset

        if is_torch_tpu_available():
            dataloader = pl.ParallelLoader(dataloader, [args.device]).per_device_loader(args.device)

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        losses_host = None
        preds_host = None
        labels_host = None
        ex_ids_host = None
        indices_host = None
        is_perturbed_host = None
        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_preds = None
        all_labels = None
        all_ex_ids = None
        all_indices = None
        all_is_perturbed = None
        # Will be useful when we have an iterable dataset so don't know its length.

        observed_num_examples = 0
        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            ex_ids = inputs.pop("ex_id", None)
            indices = inputs.pop("idx", None)
            is_perturbed = inputs.pop("is_perturbed", None)
           
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)

            if is_torch_tpu_available():
                xm.mark_step()

            # Update containers on host
            if loss is not None:
                losses = self._nested_gather(loss.repeat(batch_size))
                losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)
            if labels is not None:
                labels = self._pad_across_processes(labels)
                labels = self._nested_gather(labels)
                labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)
            if logits is not None:
                logits = self._pad_across_processes(logits)
                logits = self._nested_gather(logits)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)
            if ex_ids is not None:
                if ex_ids.device == torch.device("cpu"):
                    ex_ids = ex_ids.to(logits.device) # FIXME not very elegant, esp if we dont have logits
                ex_ids = self._nested_gather(ex_ids)
                ex_ids_host = ex_ids if ex_ids_host is None else nested_concat(ex_ids_host, ex_ids, padding_index=-100)
            if indices is not None:
                indices = self._pad_across_processes(indices)
                indices = self._nested_gather(indices)
                indices_host = indices if indices_host is None else nested_concat(indices_host, indices, padding_index=-100)
            if is_perturbed is not None:
                if is_perturbed.device == torch.device("cpu"):
                    is_perturbed = is_perturbed.to(logits.device) # FIXME not very elegant, esp if we dont have logits
                is_perturbed = self._pad_across_processes(is_perturbed)
                is_perturbed = self._nested_gather(is_perturbed)
                is_perturbed_host = is_perturbed if is_perturbed_host is None else nested_concat(is_perturbed_host, is_perturbed, padding_index=-100)

            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
                if preds_host is not None:
                    logits = nested_numpify(preds_host)
                    all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
                if labels_host is not None:
                    labels = nested_numpify(labels_host)
                    all_labels = (
                        labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
                    )
                if ex_ids_host is not None:
                    ex_ids = nested_numpify(ex_ids)
                    all_ex_ids = (
                        ex_ids if all_ex_ids is None else nested_concat(all_ex_ids, ex_ids, padding_index=-100)
                    )
                if indices_host is not None:
                    indices = nested_numpify(indices)
                    all_indices = (
                        indices if all_indices is None else nested_concat(all_indices, indices, padding_index=-100)
                    )
                if is_perturbed_host is not None:
                    is_perturbed = nested_numpify(is_perturbed)
                    all_is_perturbed = (
                        is_perturbed if all_is_perturbed is None else nested_concat(all_is_perturbed, is_perturbed, padding_index=-100)
                    )

                # Set back to None to begin a new accumulation
                losses_host, preds_host, labels_host, ex_ids_host, is_perturbed_host = None, None, None, None, None

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
        if preds_host is not None:
            logits = nested_numpify(preds_host)
            all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
        if labels_host is not None:
            labels = nested_numpify(labels_host)
            all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
        if ex_ids_host is not None:
            ex_ids = nested_numpify(ex_ids_host)
            all_ex_ids = ex_ids if all_ex_ids is None else nested_concat(all_ex_ids, ex_ids, padding_index=-100)
        if indices_host is not None:
            indices = nested_numpify(indices_host)
            all_indices = indices if all_indices is None else nested_concat(all_indices, indices, padding_index=-100)
        if is_perturbed_host is not None:
            is_perturbed = nested_numpify(is_perturbed_host)
            all_is_perturbed = is_perturbed if all_is_perturbed is None else nested_concat(all_is_perturbed, is_perturbed, padding_index=-100)

        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and hasattr(eval_dataset, "num_examples"):
            num_samples = eval_dataset.num_examples
        else:
            num_samples = observed_num_examples

        # Number of losses has been rounded to a multiple of batch_size and in a distributed training, the number of
        # samplers has been rounded to a multiple of batch_size, so we truncate.
        if all_losses is not None:
            all_losses = all_losses[:num_samples]
        if all_preds is not None:
            all_preds = nested_truncate(all_preds, num_samples)
        if all_labels is not None:
            all_labels = nested_truncate(all_labels, num_samples)
        if all_ex_ids is not None:
            all_ex_ids = nested_truncate(all_ex_ids, num_samples)
        if all_indices is not None:
            all_indices = nested_truncate(all_indices, num_samples)
        if all_is_perturbed is not None:
            all_is_perturbed = nested_truncate(all_is_perturbed, num_samples)

        # Metrics!
        if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
            metrics = self.compute_metrics(
                FairScoreEvalPrediction(
                    predictions=all_preds,
                    label_ids=all_labels,
                    ex_ids=all_ex_ids,
                    indices=all_indices,
                    is_perturbed=all_is_perturbed
                    )
                    )
        else:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)
