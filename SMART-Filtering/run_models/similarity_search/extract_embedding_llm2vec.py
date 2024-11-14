## This is the approach following "LLM2Vec: Large Language Models Are Secretly Powerful Text Encoders" work

import argparse
import json
import os
import time, pdb
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, AutoConfig
from llm2vec import LLM2Vec
from peft import PeftModel


choices = ["A", "B", "C", "D" , "E"]


def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt


@torch.no_grad()
def eval(args, subject, l2v, tokenizer, test_df) :
    cors = []
    all_embeddings = []

    for i in range(test_df.shape[0]):
        prompt_end = format_example(test_df, i, include_answer=False)
        prompt = [prompt_end]

        embeddings = l2v.encode(prompt)

        all_embeddings.append(embeddings[0].cpu().numpy())

    return all_embeddings


def main(args):

    data_dir = os.path.join(args.data_folder, args.dataset_name)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)

    ## This code is following LLM2VEc huggingface instructions
    model = AutoModel.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        trust_remote_code=True,
        config=config
    )

    model = PeftModel.from_pretrained(model, args.model)

    l2v = LLM2Vec(model, tokenizer, pooling_mode="mean", max_length=512)

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

    emb_dir = os.path.join(out_dir_dataset, 'embeddings')
    if not os.path.exists(emb_dir):
        os.makedirs(emb_dir)

    save_dir = os.path.join(emb_dir, args.model.split('/')[-1])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for subject in subjects:
        test_df = pd.read_csv(
            os.path.join(data_dir, "test", subject + "_test.csv"), header=None
        )

        embeddings = eval(args, subject, l2v, tokenizer, test_df)

        np.save(os.path.join(save_dir, "{}.npy".format(subject)), embeddings)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name",type=str, default='ai2_arc')
    parser.add_argument("--data_folder", "-d", type=str, default="../../datasets")
    parser.add_argument("--out_dir", "-s", type=str, default="../../results")
    parser.add_argument("--model", "-m", default="McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp", type=str)
    args = parser.parse_args()
    main(args)
