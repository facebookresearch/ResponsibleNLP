# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the CC BY NC license found in the
# LICENSE-CC-BY-NC file in the project directory.

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch, pdb
from transformers import AutoModelForCausalLM, AutoTokenizer

choices = ["A", "B", "C", "D"]


def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s


def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt


def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
        format_subject(subject)
    )
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt


@torch.no_grad()
def eval(args, subject, model, tokenizer, dev_df, test_df):
    cors = []
    all_probs = []
    answers = choices[: test_df.shape[1] - 2]

    for i in range(test_df.shape[0]):
        # get prompt and make sure it fits
        k = args.ntrain
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end

        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

        label = test_df.iloc[i, test_df.shape[1] - 1]

        logits = model(input_ids=input_ids).logits[0, -1]

        probs = (
            torch.nn.functional.softmax(
                torch.tensor(
                    [
                        logits[tokenizer("A").input_ids[-1]],
                        logits[tokenizer("B").input_ids[-1]],
                        logits[tokenizer("C").input_ids[-1]],
                        logits[tokenizer("D").input_ids[-1]],
                    ]
                ).float(),
                dim=0,
            )
            .detach()
            .cpu()
            .numpy()
        )
        pred = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(probs)]

        cor = pred == label
        cors.append(cor)
        all_probs.append(probs)

    acc = np.mean(cors)
    cors = np.array(cors)

    all_probs = np.array(all_probs)
    print("Average accuracy {:.3f} - {}".format(acc, subject))

    return cors, acc, all_probs


def main(args):

    data_dir = os.path.join(args.data_folder, args.dataset_name)

    if 'gemma-2' in args.model : 
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            load_in_8bit=False,
            low_cpu_mem_usage=True,
            device_map="auto",
            trust_remote_code=True
        )
    else :  
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float16,
            load_in_8bit=False,
            low_cpu_mem_usage=True,
            device_map="auto",
            trust_remote_code=True
        )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model.eval()
    subjects = sorted(
        [
            f.split("_test.csv")[0]
            for f in os.listdir(os.path.join(data_dir, "test"))
            if "_test.csv" in f
        ]
    )

    out_dir_dataset = os.path.join(args.out_dir, args.dataset_name)
    if not os.path.exists(out_dir_dataset):
        os.makedirs(out_dir_dataset)

    save_dir = os.path.join(out_dir_dataset, 'entire_dataset')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    dataset_save_dir = 'results_' + args.model.split('/')[-1]

    if not os.path.exists(os.path.join(save_dir, dataset_save_dir)):
        os.makedirs(os.path.join(save_dir, dataset_save_dir))

    all_cors = []

    for subject in subjects:
        if os.path.exists(os.path.join(data_dir, "dev")):
            dev_df = pd.read_csv(
                os.path.join(data_dir, "dev", subject + "_dev.csv"), header=None
            )[: args.ntrain]
        else:
            print (subject)
            dev_df = pd.read_csv(
                os.path.join(data_dir, "val", subject + "_val.csv"), header=None
            )[: args.ntrain]

        test_df = pd.read_csv(
            os.path.join(data_dir, "test", subject + "_test.csv"), header=None
        )

        cors, acc, probs = eval(args, subject, model, tokenizer, dev_df, test_df)
        all_cors.append(cors)

        test_df["{}_correct".format(args.model)] = cors
        for j in range(probs.shape[1]):
            choice = choices[j]
            test_df["{}_choice{}_probs".format(args.model, choice)] = probs[:, j]
        test_df.to_csv(
            os.path.join(
                save_dir, dataset_save_dir, "{}.csv".format(subject)
            ),
            index=None,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=0)
    parser.add_argument("--dataset_name",type=str, default='ai2_arc')
    parser.add_argument("--data_folder", "-d", type=str, default="../../../datasets")
    parser.add_argument("--out_dir", "-s", type=str, default="../../../results")
    parser.add_argument("--model", "-m", type=str)
    args = parser.parse_args()
    main(args)
