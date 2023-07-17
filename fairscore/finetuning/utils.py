#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from datasets import (
    concatenate_datasets, load_dataset, load_metric
)
from datasets import Dataset as ArrowsDataset
from datasets import DatasetInfo
import itertools
from os import path
import torch
from transformers import DataCollatorWithPadding

MULTIPLE_CHOICE_TASKS = ["copa", "multirc", "record", "wsc"]
REGRESSION_TASKS = ["stsb"]
GLUE_TASKS = ["cola", "mnli", "mrpc", "qnli", "qqp", "stsb", "sst2", "wnli"]
SUPERGLUE_TASKS = ["axb", "axg", "boolq", "cb", "copa", "multirc", "record", "rte", "wic", "wsc", "wsc.fixed"]
TASK2KEYS = {
    "anli" : ["premise", "hypothesis", "label"],
    "boolq" : ["question", "passage", "label"],
    "cb" : ["premise", "hypothesis", "label"],
    "copa" : [("premise", "premise"), ("choice1", "choice2"), "label"],
    "cola" : ["sentence", None, "label"],
    "mnli" : ["premise", "hypothesis", "label"],
    "mrpc" : ["sentence1", "sentence2", "label"],
    "multirc" : ["paragraph", ("question", "answer"), "label"],
    "qqp" : ["question1", "question2", "label"],
    "qnli" : ["question", "sentence", "label"],
    "record" : [("passage", "query"), "entities", "answer"],
    "rte" : ["premise", "hypothesis", "label"],
    "sst2" : ["sentence", None, "label"],
    "stsb" : ["sentence1", "sentence2", "label"],
    "wic" : [("sentence1", "sentence2"), "word", "label"],
    "wnli" : ["sentence1", "sentence2", "label"],
    "wsc" : ["text", ("span1_text", "span2_text"), "label"],
}

class PerturbedDataCollatorWithPadding(DataCollatorWithPadding):
    def __call__(self, features):
        if isinstance(features, list) and "idx" in features[0]:
            indices = [feat.pop("idx") for feat in features]
        else:
            indices = None

        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        if indices is not None:
            batch["indices"] = indices
        if "label" in batch:
            batch["labels"] = batch.pop("label").long()
        if "label_ids" in batch:
            batch["labels"] = batch.pop("label_ids").long()
        return batch

class PerturbedDataCollatorWithPaddingForMultipleChoice(DataCollatorWithPadding):
    # see https://huggingface.co/docs/transformers/tasks/multiple_choice
    def __call__(self, features):
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [
            [{k: v if isinstance(v, int) else v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = sum(flattened_features, [])

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        
        if "label" in batch:
            batch["labels"] = batch.pop("label").long()
        if "label_ids" in batch:
            batch["labels"] = batch.pop("label_ids").long()
        
        # reshape examples to batch_size x num_choices x *
        batch = {k : v.reshape(batch_size, num_choices, -1) for k,v in batch.items()}

        # remove the extra dimension and the duplicates for each answer choice
        for key in ["labels", "is_perturbed", "ex_id"]:
            if key not in batch:
                # is_perturbed will only be present in  fairscore
                continue
            
            batch[key].squeeze_()
            assert torch.equal(batch[key][:, 0], batch[key][:, 1]), f"{key} key doesn't match for each answer choice; some issue with data loading"
            batch[key] = batch[key][:, 0]
        return batch

def load_metric_from_dataset_name(task_name):
    if task_name in GLUE_TASKS:
        return load_metric("glue", task_name)
    elif task_name in SUPERGLUE_TASKS:
        return load_metric("super_glue", task_name)
    else:
        return None
    
def convert_input_ids_to_str(input_ids):
    if isinstance(input_ids[0], list):
        return " ".join([str(i) for i in itertools.chain(*input_ids)])
    else:
        return " ".join([str(i) for i in input_ids])

def load_and_format_dataset(
    task_name, dataset_name_or_path, tokenizer, split_names=None, cols_to_skip=None, max_seq_length=None, description=None
    ):
    if cols_to_skip is not None:
        cols_to_skip.extend(["idx", "input_ids", "input_ids_str", "attention_mask", "labels"])
    else:
        cols_to_skip = ["idx", "input_ids", "input_ids_str", "attention_mask", "labels"]

    # load to dataset format
    if path.exists(dataset_name_or_path):
        # FIXME a little hacky right now, key is always 'train' splits
        dataset = load_dataset("json", data_files=dataset_name_or_path)["train"]
    elif dataset_name_or_path in SUPERGLUE_TASKS:
        dataset = load_dataset("super_glue", dataset_name_or_path, "plain_text")
    elif dataset_name_or_path in GLUE_TASKS:
        dataset = load_dataset("glue", dataset_name_or_path)
    else:
        dataset = load_dataset(dataset_name_or_path, "plain_text")
    dataset.set_format("pandas")

    # combine data across rounds or splits (if needed) then tokenize
    if split_names:
        dataset = concatenate_datasets([dataset[rn] for rn in split_names])

    # TODO merge all of this
    dataset = dataset[:]
    dataset = ArrowsDataset.from_pandas(
        dataset, info=DatasetInfo(description=description)
        )
    
    # where input_key_names can be e.g. hypothesis, premise, e.g.
    text_key_names, text_pair_key_names, label_key_name = TASK2KEYS[task_name]
    combine_method = "list" if task_name in MULTIPLE_CHOICE_TASKS else "str"
    dataset = combine_fields_in_dataset(dataset, text_key_names, "text", combine_method)
    dataset = combine_fields_in_dataset(dataset, text_pair_key_names, "text_pair", combine_method)

    tokenize_single_input = lambda input: tokenizer(
        text=input["text"],
        text_pair=input["text_pair"] if "text_pair" in input else None,
        padding=True,
        truncation=True,
        max_length=max_seq_length
        )

    # convert back to arrows format and tokenize
    dataset = dataset.map(lambda input_ex: tokenize_single_input(input_ex))
    dataset = dataset.map(lambda input_ex: {"input_ids_str" : convert_input_ids_to_str(input_ex["input_ids"])})
    print("d", dataset["text"][0])
    print("d", dataset["text_pair"][0] if "text_pair" in dataset.column_names else None)
    
    dataset = dataset.rename_column(label_key_name, "labels")
    cols_to_remove = [col for col in dataset.column_names if col not in cols_to_skip]
    dataset = dataset.remove_columns(cols_to_remove)
    return dataset

def combine_fields_in_dataset(dataset, fields_to_combine, output_field, combine_method="str"):
    assert combine_method in ["list", "str"], f"Unknown combine method: {combine_method}"
    if fields_to_combine is None:
        return dataset

    def combine_into_list(example, fields):
        output = []
        [output.append(example[field]) for field in fields]
        return output

    if isinstance(fields_to_combine, str):
        fields_to_combine = [fields_to_combine]

    if combine_method == "list":
        dataset = dataset.map(lambda ex: {output_field: combine_into_list(ex, fields_to_combine)})
    else:
        # TODO remove this hard code
        dataset = dataset.map(lambda ex: {output_field: "</s></s>".join(ex[field] for field in fields_to_combine)})
    return dataset

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
