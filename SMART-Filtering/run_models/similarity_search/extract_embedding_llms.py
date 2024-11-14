## This method is wrong as decoder only models like LLMs cannot be used to get sentence embeddings. A combine emebedding doesn't
## exists and last token embeddings is modified embeddings of last input token with contexts mixed from all input tokens

import argparse
import json
import os
import time, pdb
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


choices = ["A", "B", "C", "D" , "E"]


def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
    #prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt


@torch.no_grad()
def eval(args, subject, model, tokenizer, test_df) :
    cors = []
    all_embeddings = []

    for i in range(test_df.shape[0]):
        # get prompt and make sure it fits
        prompt_end = format_example(test_df, i, include_answer=False)
        prompt = prompt_end

        inputs = tokenizer.encode_plus(prompt, return_tensors="pt").to(model.device)

        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
        embeddings = hidden_states[:,-1,:]


        all_embeddings.append(embeddings.cpu().numpy())

    return all_embeddings


def main(args):

    data_dir = os.path.join(args.data_folder, args.dataset_name)

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

        embeddings = eval(args, subject, model, tokenizer, test_df)

        np.save(os.path.join(save_dir, "{}.npy".format(subject)), embeddings)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name",type=str, default='ai2_arc')
    parser.add_argument("--data_folder", "-d", type=str, default="../../datasets")
    parser.add_argument("--out_dir", "-s", type=str, default="../../results")
    parser.add_argument("--model", "-m", default="meta-llama/Meta-Llama-3-8B", type=str)
    args = parser.parse_args()
    main(args)
