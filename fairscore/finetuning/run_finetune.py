#!/usr/bin/env python3.8

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from transformers import BertForMaskedLM

model = BertForMaskedLM.from_pretrained("bert-base-uncased")

import configargparse
import torch
from transformers import (
    AutoModelForMultipleChoice,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments
)
from transformers.utils import logging
from compute_fairscore import (
    compute_metrics_fairscore, compute_fairscore
)
from utils import (
    MULTIPLE_CHOICE_TASKS,
    REGRESSION_TASKS,
    TASK2KEYS,
    PerturbedDataCollatorWithPadding,
    PerturbedDataCollatorWithPaddingForMultipleChoice,
    load_and_format_dataset,
    load_metric_from_dataset_name
)
from perturbed_trainer import FairscoreTrainerEvalOnly

logging.set_verbosity_debug()
logger = logging.get_logger(__name__)
logger.setLevel("INFO")

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_datasets_and_collator(
    task_name,
    eval_unperturbed_dataset_name_or_path,
    eval_perturbed_dataset_name_or_path,
    eval_unperturbed_split_names,
    eval_perturbed_split_names,
    tokenizer,
    train_dataset_name_or_path=None,
    train_split_names=None,
    fp16=False,
    max_seq_length=None
    ):
    if task_name in MULTIPLE_CHOICE_TASKS:
        collator = PerturbedDataCollatorWithPaddingForMultipleChoice(
            tokenizer, pad_to_multiple_of=8 if fp16 else None
            )
    else:
        collator = PerturbedDataCollatorWithPadding(
            tokenizer, pad_to_multiple_of=8 if fp16 else None
            )
    
    # load train dataset (if name/path provided)
    if train_dataset_name_or_path:
        train_dataset = load_and_format_dataset(
            task_name=task_name,
            dataset_name_or_path=train_dataset_name_or_path,
            split_names=train_split_names,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length
            )
        train_dataset = train_dataset.remove_columns(["idx", "input_ids_str"])
        logger.info(f"Num train examples: {len(train_dataset)}")
    else:
        train_dataset = None
        
    # load unperturbed eval dataset
    eval_unperturbed_dataset = load_and_format_dataset(
        task_name=task_name,
        dataset_name_or_path=eval_unperturbed_dataset_name_or_path,
        split_names=eval_unperturbed_split_names,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        description="unperturbed_eval"
        )
    if "idx" in eval_unperturbed_dataset.column_names:
        indices = eval_unperturbed_dataset["idx"]
        eval_unperturbed_dataset = eval_unperturbed_dataset.remove_columns(["idx", "input_ids_str"])
    logger.info(f"Num eval examples for unperturbed: {len(eval_unperturbed_dataset)}")

    # load perturbed eval dataset
    eval_perturbed_dataset = load_and_format_dataset(
        task_name=task_name,
        dataset_name_or_path=eval_perturbed_dataset_name_or_path,
        split_names=eval_perturbed_split_names, # for now, this should always be none
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        description="perturbed_eval"
        )
    if "idx" in eval_perturbed_dataset.column_names:
        indices = eval_perturbed_dataset["idx"]
        eval_perturbed_dataset = eval_perturbed_dataset.remove_columns(["idx", "input_ids_str"])
    logger.info(f"Num eval examples for perturbed: {len(eval_perturbed_dataset)}")
    
    return train_dataset, eval_unperturbed_dataset, eval_perturbed_dataset, collator, indices

def compute_metrics(output, metric, indices, is_regression, dataset_type):
    if is_regression:
        predictions = torch.tensor(output.predictions, dtype=torch.float)
    else:
        logits = torch.tensor(output.predictions, dtype=torch.float)
        probs = torch.softmax(logits, dim=1)
        predictions = probs.argmax(dim=1)
    
    if metric is not None:
        if isinstance(indices[0], dict):
            predictions_with_indices = [{"idx" : idx, "prediction" : pred.item()} for pred,idx in zip(predictions, indices)]
            result = metric.compute(
                predictions=predictions_with_indices, references=output.label_ids
            )
        else:
            result = metric.compute(
                predictions=predictions, references=output.label_ids
            )

        if len(result) > 1:
            combined_results = torch.tensor(list(result.values()))
            result["combined_score"] = torch.mean(combined_results).item()

    if "acc" not in result:
        labels = torch.tensor(output.label_ids, dtype=torch.long)
        result["acc"] = (predictions == labels).sum() / len(labels)

    result["dataset_type"] = dataset_type
    print(result)
    return result

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


    train_dataset, eval_unperturbed_dataset, eval_perturbed_dataset, collator, indices = load_datasets_and_collator(
        task_name=args.task_name,
        train_dataset_name_or_path=args.train_dataset_name_or_path if args.mode == "train" else None,
        train_split_names=args.train_split_names,
        eval_unperturbed_dataset_name_or_path=args.eval_unperturbed_dataset_name_or_path,
        eval_perturbed_dataset_name_or_path=args.eval_perturbed_dataset_name_or_path,
        eval_unperturbed_split_names=args.eval_unperturbed_split_names,
        eval_perturbed_split_names=args.eval_perturbed_split_names,
        tokenizer=tokenizer,
        fp16=args.fp16,
        max_seq_length=args.max_seq_length,
    )
    
    # -- set up training
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        learning_rate=args.learning_rate,
        adam_epsilon=args.adam_eps,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        report_to="tensorboard",
        save_strategy="epoch",
        evaluation_strategy="epoch",
        metric_for_best_model=args.metric_for_best_model,
        logging_steps=args.num_logging_steps,
        dataloader_num_workers=args.num_workers,
        fp16=args.fp16,
        sharded_ddp=args.sharded_ddp,
        save_total_limit=2,
        gradient_accumulation_steps=args.gradient_accum_steps,
        load_best_model_at_end=True,
    )

    metric = load_metric_from_dataset_name(args.task_name)
    is_regression_task = args.task_name in REGRESSION_TASKS
    
    trainer = FairscoreTrainerEvalOnly(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_unperturbed_dataset,
        data_collator=collator,
        compute_metrics=lambda preds: compute_metrics(
            preds, metric, indices=indices, is_regression=is_regression_task, dataset_type=eval_unperturbed_dataset.description
        )
    )
    os.makedirs("results", exist_ok=True)
    
    if args.mode == "train":
        logger.info("Starting training..")
        print(resume_from_checkpoint)
        trainer.train(
            resume_from_checkpoint=resume_from_checkpoint
            )

    # run evaluation
    logger.info(f"Running eval for {args.model_name_or_path} on unperturbed")
    print(f"Running eval for {args.model_name_or_path} on unperturbed")
    trainer.evaluate()

    # now switch and run over perturbed data
    logger.info(f"Running eval for {args.model_name_or_path} on perturbed data")
    print(f"Running eval for {args.model_name_or_path} on perturbed data")
    trainer.eval_dataset = eval_perturbed_dataset
    trainer.compute_metrics = lambda preds: compute_metrics(
        preds, metric, indices=indices, is_regression=is_regression_task, dataset_type=eval_perturbed_dataset.description
    )
    trainer.evaluate()
    
    # -- compute fairscore --
    # load perturbed examples only for computing fairscore
    cols_to_skip = ["ex_id", "is_perturbed"]
    eval_perturbed_only_dataset = load_and_format_dataset(
        task_name=args.task_name,
        dataset_name_or_path=args.eval_perturbed_examples_only_path,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        cols_to_skip=cols_to_skip
        )
    logger.info(f"Num of examples: {len(eval_perturbed_only_dataset)}")

    num_perturbed, num_unperturbed = 0, 0
    for ex in eval_perturbed_only_dataset:
        if ex["is_perturbed"]:
            num_perturbed += 1
        else:
            num_unperturbed += 1
    assert num_perturbed == num_unperturbed, f"Unexpected size mismatch: perturbed {num_perturbed}, unperturbed {num_unperturbed}"
    
    cols_to_remove = [col for col in eval_perturbed_only_dataset.column_names if col not in ["input_ids", "attention_mask", "labels", "ex_id", "is_perturbed"]]
    eval_perturbed_only_dataset = eval_perturbed_only_dataset.remove_columns(cols_to_remove)
    
    ids2inputs = {
        (eid, is_perturbed) : input_ids for is_perturbed, eid, input_ids \
            in zip(eval_perturbed_only_dataset["is_perturbed"], eval_perturbed_only_dataset["ex_id"], eval_perturbed_only_dataset["input_ids"])
            }
    if args.task_name in MULTIPLE_CHOICE_TASKS:
        collator = PerturbedDataCollatorWithPaddingForMultipleChoice(
            tokenizer, pad_to_multiple_of=8 if args.fp16 else None
            )
    else:
        collator = PerturbedDataCollatorWithPadding(
            tokenizer, pad_to_multiple_of=8 if args.fp16 else None
            )

    aggregate_results = {
        "perturbed" : {}, "unperturbed" : {}
    }
    trainer.eval_dataset = eval_perturbed_only_dataset
    trainer.compute_metrics = lambda x: compute_metrics_fairscore(x, aggregate_results, ids2inputs)
    trainer.evaluate()    
    compute_fairscore(aggregate_results, tokenizer, verbose=not args.disable_verbose)

if __name__ == "__main__":
    parser = configargparse.ArgumentParser()
    parser.add_argument("-c", "--config", is_config_file=True)
    parser.add_argument("--output_dir", required=True, help="path to save model checkpoints")
    parser.add_argument("--seed", type=int, default=10)

    # dataset
    parser.add_argument("--train_dataset_name_or_path", type=str, help="provide name of HF dataset or path to perturbed, non-HF dataset")
    parser.add_argument("--eval_unperturbed_dataset_name_or_path", type=str, required=True, help="provide name of HF dataset or path non-HF dataset")
    parser.add_argument(
        "--eval_perturbed_dataset_name_or_path", type=str, required=True, help="provide name (unlikely) or path (much more likely) to perturbed, non-HF dataset"
        )
    parser.add_argument("--train_split_names", nargs="+", help="(at the moment) only applies to unperturbed data")
    parser.add_argument("--eval_unperturbed_split_names", nargs="+", help="(at the moment) only applies to unperturbed data")
    parser.add_argument("--eval_perturbed_split_names", nargs="+", help="(at the moment) only applies to unperturbed data, so we expect this to be None")
    parser.add_argument("--eval_perturbed_examples_only_path", type=str, required=True, help="path to jsonl containing perturbed examples only")
    parser.add_argument("--num_labels", type=int, required=True, help="number of labels for classifier head")
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--task_name", required=True, type=str)

    # model
    parser.add_argument("--model_name_or_path", type=str, default="roberta-large", help="either name of pretrained model or path to saved checkpoint")
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--adam_eps", type=float, default=1e-6)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.98)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--mode", choices=["train", "eval"], default="train")

    parser.add_argument("--max_seq_length", type=int)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--num_logging_steps", type=int, default=250)
    
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--sharded_ddp", default="", choices=["simple", "zero_dp_2", "zero_dp_3", "offload"])
    parser.add_argument("--gradient_accum_steps", default=1, type=int)
    parser.add_argument("--disable_verbose", action="store_true")
    parser.add_argument("--metric_for_best_model", default="accuracy")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    if args.mode == "train":
        assert args.train_dataset_name_or_path is not None, "Must provide training data name or path"

    main(args)