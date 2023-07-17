#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import configargparse
import os
import re
import torch
from transformers import (
    AutoModelForMultipleChoice,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments
)
from transformers.utils import logging
from utils import (
    MULTIPLE_CHOICE_TASKS,
    TASK2KEYS,
    PerturbedDataCollatorWithPadding,
    PerturbedDataCollatorWithPaddingForMultipleChoice,
    load_and_format_dataset
)
from perturbed_trainer import FairscoreTrainerEvalOnly

logging.set_verbosity_info()
logger = logging.get_logger(__name__)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def compute_metrics_fairscore(prediction, aggregate_results, ids2inputs):
    logits = torch.tensor(prediction.predictions, dtype=torch.float)
    labels = torch.tensor(prediction.label_ids, dtype=torch.long)

    probs = torch.softmax(logits, dim=1)
    preds = probs.argmax(dim=1)
    overall_accuracy = (preds == labels).sum() / len(preds)

    # get all predictions
    assert len(prediction.ex_ids) == len(preds) == len(labels)
    for ex_id, prediction, is_perturbed, label in zip(prediction.ex_ids, preds, prediction.is_perturbed, labels):
        data_key = "perturbed" if is_perturbed else "unperturbed"
        correct = prediction == label
        input_ids = ids2inputs[(ex_id, is_perturbed)]
        aggregate_results[data_key][ex_id] = (
            prediction.item(), label.item(), correct.item(), input_ids
            )
    print("acc", overall_accuracy)
    return {"accuracy" : overall_accuracy}

def compute_fairscore(aggregate_results, tokenizer, verbose=True):
    def convert_ids_to_tokens(tokenized_ids):
        if isinstance(tokenized_ids[0], list):
            return " [MULT_CHOICE] ".join([
                    re.sub("Ġ", "", " ".join(tokenizer.convert_ids_to_tokens(tids))) for tids in tokenized_ids
                ])

        else:
            return re.sub(
                "Ġ", "", " ".join(tokenizer.convert_ids_to_tokens(tokenized_ids))
                )

    metrics = {
        "both_correct" : [],
        "neither_correct" : [],
        "different" : []
    }

    assert len(aggregate_results["perturbed"]) == len(aggregate_results["unperturbed"])
    for ex_id in aggregate_results["perturbed"].keys():
        perturbed_pred, perturbed_label, perturbed_correct, perturbed_input_ids = aggregate_results["perturbed"][ex_id]
        unperturbed_pred, unperturbed_label, unperturbed_correct, unperturbed_input_ids = aggregate_results["unperturbed"][ex_id]
        assert perturbed_label == unperturbed_label
        
        metrics["both_correct"].append(int(perturbed_correct and unperturbed_correct))
        metrics["neither_correct"].append(not perturbed_correct and not unperturbed_correct)
        metrics["different"].append(int(perturbed_pred != unperturbed_pred))

        if verbose:
            unperturbed_tokens = convert_ids_to_tokens(unperturbed_input_ids)
            perturbed_tokens = convert_ids_to_tokens(perturbed_input_ids)
            assert unperturbed_input_ids != perturbed_input_ids
            
            if unperturbed_correct and not perturbed_correct:
                print(f"Correct on original, wrong on perturbed\nOriginal: {unperturbed_tokens}\nPerturbed: {perturbed_tokens}\n\n")

            if not unperturbed_correct and perturbed_correct:
                print(f"Wrong on original, correct on perturbed\nOriginal: {unperturbed_tokens}\nPerturbed: {perturbed_tokens}\n\n")
    
    for k,v in metrics.items():
        print(f"{k}: {100 * (sum(v) / len(v))}, {sum(v)} of {len(v)}")

def main(args):
    assert args.task_name in TASK2KEYS, f"Unsupported task: {args.task_name}"

    # -- load model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    
    # -- load datasets
    cols_to_skip = ["ex_id", "is_perturbed"]
    dataset = load_and_format_dataset(
        task_name=args.task_name,
        dataset_name_or_path=args.data_path,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        cols_to_skip=cols_to_skip
        )
    logger.info(f"Num of examples: {len(dataset)}")
    num_perturbed, num_unperturbed = 0, 0
    for ex in dataset:
        if ex["is_perturbed"]:
            num_perturbed += 1
        else:
            num_unperturbed += 1
    assert num_perturbed == num_unperturbed, f"Unexpected size mismatch: perturbed {num_perturbed}, unperturbed {num_unperturbed}"
    
    cols_to_remove = [col for col in dataset.column_names if col not in ["input_ids", "attention_mask", "labels", "ex_id", "is_perturbed"]]
    dataset = dataset.remove_columns(cols_to_remove)
    ids2inputs = {(eid, is_perturbed) : input_ids for is_perturbed, eid, input_ids in zip(dataset["is_perturbed"], dataset["ex_id"], dataset["input_ids"])}

    # -- set up training
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_eval_batch_size=args.eval_batch_size,
        remove_unused_columns=False,
        report_to="tensorboard",
        dataloader_num_workers=args.num_workers,
        fp16=args.fp16,
        sharded_ddp=args.sharded_ddp,
        save_total_limit=3,
        no_cuda=args.no_cuda,
    )
    os.makedirs("results", exist_ok=True)
    
    aggregate_results = {
        "perturbed" : {}, "unperturbed" : {}
    }
    f_auto_model = AutoModelForMultipleChoice if args.task_name in MULTIPLE_CHOICE_TASKS else AutoModelForSequenceClassification
    model = f_auto_model.from_pretrained(
        args.model_name_or_path, num_labels=args.num_labels
        )

    if args.task_name in MULTIPLE_CHOICE_TASKS:
        collator = PerturbedDataCollatorWithPaddingForMultipleChoice(
            tokenizer, pad_to_multiple_of=8 if args.fp16 else None
            )
    else:
        collator = PerturbedDataCollatorWithPadding(
            tokenizer, pad_to_multiple_of=8 if args.fp16 else None
            )
    trainer = FairscoreTrainerEvalOnly(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        eval_dataset=dataset,
        data_collator=collator,
        compute_metrics=lambda x: compute_metrics_fairscore(x, aggregate_results, ids2inputs)
    )
    trainer.evaluate()    
    compute_fairscore(aggregate_results, tokenizer)

if __name__ == "__main__":
    parser = configargparse.ArgumentParser()
    parser.add_argument("-c", "--config", is_config_file=True)
    parser.add_argument("--task_name", required=True, type=str)
    parser.add_argument("--data_path", type=str, required=True, help="path to perturbed (and corresponding original, unperturbed) dataset examples in jsonl format")

    parser.add_argument("--model_name_or_path", type=str, default="roberta-large", help="either name of pretrained model or path to saved checkpoint")
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--max_seq_length", type=int)
    parser.add_argument("--output_dir", required=True, help="path to save model checkpoints")

    parser.add_argument("--num_labels", type=int, required=True, help="number of labels for classifier head")
    parser.add_argument("--num_workers", type=int, default=8)

    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--sharded_ddp", default="", choices=["simple", "zero_dp_2", "zero_dp_3", "offload"])
    args = parser.parse_args()

    if args.no_cuda:
        args.fp16 = None
        args.sharded_ddp = ""

    main(args)