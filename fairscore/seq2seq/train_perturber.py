#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import configargparse
from datasets import load_metric
import json
import pandas as pd
from torch.utils.data import Dataset
from transformers import (
    AutoModelForMaskedLM, AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments
)
import os

SPECIAL_TOKENS = ["<PERT_SEP>" , "<SEP>"]

# remaining config params from ParlAI
#{"-t": ["jsonfile"], "-veps": [0.25], "--log-every-n-secs": [20],
# "--optimizer": ["adam"], "--warmup-updates": [1200],
# "--gradient-clip": [0.1], "-vp": [10], "--max-train-time": [84600], "-vmm": ["max"], "-dynb": ["full"]

class PerturberDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_seq_length=None):
        """
        data_path: path to file containing ParlAI formatted input data
        tokenizer: TODO
        max_seq_length: TODO
        """
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        self.examples = []
        df = pd.read_csv(data_path, sep="|")
        self.examples = list(df.itertuples(index=False))
    
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        original, perturbed = self.examples[idx]
        tokenized_input = self.tokenizer(original, truncation=True, max_length=self.max_seq_length)
        tokenized_perturbed = self.tokenizer(perturbed, truncation=True, max_length=self.max_seq_length)
        tokenized_input["labels"] = tokenized_perturbed["input_ids"]
        return tokenized_input

def parse_args():
    parser = configargparse.ArgumentParser()
    parser.add_argument("-c", is_config_file=True)
    # model args
    parser.add_argument("--model_name_or_path", default="facebook/bart-large")
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--fp16_eval", action="store_true")
    parser.add_argument("-lr", "--learning_rate", default=1e-5)
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    parser.add_argument("--sharded_ddp", default=False, choices=["simple", "zero_dp_2", "zero_dp_3", "offload"])
    parser.add_argument("--evaluation_strategy", default="epoch", choices=["no", "epoch", "steps"])
    parser.add_argument("--save_strategy", default="epoch", choices=["no", "epoch", "steps"])
    # dataset args
    parser.add_argument("--train_data_path", type=str, required=True, help="expecting data in ParlAI format")
    parser.add_argument("--valid_data_path", type=str, required=True, help="expecting data in ParlAI format")
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    return parser.parse_args()

def compute_metrics(tokenizer, eval_prediction, decode_pad_token_id=-100):
    # inputs = eval_prediction.inputs
    batch_label_ids = eval_prediction.label_ids
    batch_pred_ids = eval_prediction.predictions

    # replace pad_token_id used during decoding with tokenizer pad_token_id
    batch_label_ids[batch_label_ids == decode_pad_token_id] = tokenizer.pad_token_id
    batch_pred_ids[batch_pred_ids == decode_pad_token_id] = tokenizer.pad_token_id

    metric = load_metric("bleu", max_order=4) # TODO add to args

    running_label_tokens, running_pred_tokens = [], []
    assert len(batch_label_ids) == len(batch_pred_ids), f"Size mismatch between predictions {len(batch_pred_ids)} and labels {len(batch_label_ids)}"
    for label_ids, pred_ids in zip(batch_label_ids, batch_pred_ids):
        label_tokens = tokenizer.convert_ids_to_tokens(label_ids, skip_special_tokens=True)
        pred_tokens = tokenizer.convert_ids_to_tokens(pred_ids, skip_special_tokens=True)

        running_label_tokens.append([label_tokens])
        running_pred_tokens.append(pred_tokens)
    
    metric_result = metric.compute(predictions=running_pred_tokens, references=running_label_tokens)
    print(metric_result)

def main():
    args = parse_args()

    # load model and add special tokens for perturbation
    model = AutoModelForMaskedLM.from_pretrained(
        args.model_name_or_path, forced_bos_token_id=0 # TODO add to args
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.add_tokens(SPECIAL_TOKENS, special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))

    # load data
    train_dataset = PerturberDataset(args.train_data_path, tokenizer, args.max_seq_length)
    eval_dataset = PerturberDataset(args.valid_data_path, tokenizer, args.max_seq_length)
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=model
    )

    trainer_arguments = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        warmup_steps=1200,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        fp16=args.fp16,
        fp16_full_eval=args.fp16_eval,
        sharded_ddp=args.sharded_ddp,
        evaluation_strategy=args.evaluation_strategy,
        save_strategy=args.save_strategy,
        # include_inputs_for_metrics=True # getting error with sharded_ddp, looks like inputs not being correctly sent to gpu?
        predict_with_generate=True,
        generation_max_length=args.max_seq_length,
        generation_num_beams=1, # TODO add to args
    )
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=trainer_arguments,
        # compute_metrics=lambda eval_pred: compute_metrics(tokenizer, eval_pred)
    )
    os.environ["WANDB_DISABLED"] = "true"
    # import ipdb; ipdb.set_trace()
    # if args.mode == "train":
    trainer.train()

    trainer.evaluate()
    import ipdb;ipdb.set_trace()

    example_english_phrase = "man, woman <PERT_SEP> The man walked down the store."
    batch = tokenizer(example_english_phrase, return_tensors="pt")
    generated_ids = model.generate(batch["input_ids"].to(device="cuda"), max_length=512)
    generated_tokens = tokenizer.convert_ids_to_tokens(generated_ids[0])
    print("g", generated_tokens)
    output_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    print(output_text)

if __name__ == "__main__":
    main()