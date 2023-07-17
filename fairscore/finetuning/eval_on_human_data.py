#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import configargparse
import os
from transformers import (
    AutoModelForMultipleChoice,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments
)
from transformers.utils import logging
from utils import (
    MULTIPLE_CHOICE_TASKS,
    REGRESSION_TASKS,
    TASK2KEYS,
    PerturbedDataCollatorWithPadding,
    PerturbedDataCollatorWithPaddingForMultipleChoice,
    compute_metrics,
    load_and_format_dataset,
    load_metric_from_dataset_name
)
from perturbed_trainer import FairscoreTrainerEvalOnly

logging.set_verbosity_debug()
logger = logging.get_logger(__name__)
logger.setLevel("INFO")

def main(args):
    assert args.task_name in TASK2KEYS, f"Unsupported task: {args.task_name}"
    # -- load model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    f_auto_model = AutoModelForMultipleChoice if args.task_name in MULTIPLE_CHOICE_TASKS else AutoModelForSequenceClassification
    model = f_auto_model.from_pretrained(
        args.model_name_or_path, num_labels=args.num_labels
        )
    resume_from_checkpoint = os.path.exists(
        os.path.join(args.model_name_or_path, "trainer_state.json")
    )
    
    # -- load dataset & create collator
    if args.task_name in MULTIPLE_CHOICE_TASKS:
        collator = PerturbedDataCollatorWithPaddingForMultipleChoice(
            tokenizer, pad_to_multiple_of=8 if args.fp16 else None
            )
    else:
        collator = PerturbedDataCollatorWithPadding(
            tokenizer, pad_to_multiple_of=8 if args.fp16 else None
            )

    # load unperturbed eval dataset
    eval_dataset = load_and_format_dataset(
        task_name=args.task_name,
        dataset_name_or_path=args.human_eval_dataset_name_or_path,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        description="human_perturbed_eval"
        )

    indices = eval_dataset["idx"]
    eval_dataset = eval_dataset.remove_columns(["idx", "input_ids_str"])
    logger.info(f"Num eval examples for unperturbed: {len(eval_dataset)}")

    # -- set up trainer for eval
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_eval_batch_size=args.eval_batch_size,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        report_to="tensorboard",
        logging_steps=args.num_logging_steps,
        dataloader_num_workers=args.num_workers,
        fp16=args.fp16,
        sharded_ddp=args.sharded_ddp,
    )

    metric = load_metric_from_dataset_name(args.task_name)
    is_regression_task = args.task_name in REGRESSION_TASKS
    
    trainer = FairscoreTrainerEvalOnly(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        eval_dataset=eval_dataset,
        data_collator=collator,
        compute_metrics=lambda preds: compute_metrics(
            preds, metric, indices=indices, is_regression=is_regression_task, dataset_type=eval_dataset.description
        )
    )
    os.makedirs("results", exist_ok=True)

    # run evaluation
    logger.info(f"Running eval for {args.output_dir} on unperturbed")
    print(f"Running eval for {args.output_dir} on unperturbed")
    trainer.evaluate()


if __name__ == "__main__":
    parser = configargparse.ArgumentParser()
    parser.add_argument("-c", "--config", is_config_file=True)
    parser.add_argument("--output_dir", required=True, help="path to save model checkpoints")
    parser.add_argument("--seed", type=int, default=10)

    # dataset
    parser.add_argument("--task_name", required=True, type=str)
    parser.add_argument("--human_eval_dataset_name_or_path", type=str, required=True)
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--num_labels", type=int, required=True, help="number of labels for classifier head")

    # model
    parser.add_argument("--model_name_or_path", type=str, default="roberta-large", help="either name of pretrained model or path to saved checkpoint")
    parser.add_argument("--max_seq_length", type=int)
    parser.add_argument("--num_logging_steps", type=int, default=250)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--sharded_ddp", default="", choices=["simple", "zero_dp_2", "zero_dp_3", "offload"])
    parser.add_argument("--disable_verbose", action="store_true")
    args = parser.parse_args()

    main(args)