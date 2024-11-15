import argparse
import json
import os
import time, pdb
import numpy as np
import pandas as pd
import torch
from transformers import BertModel, BertTokenizer
from sentence_transformers import SentenceTransformer

choices = ["A", "B", "C", "D" , "E"]


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
    #prompt += "\nAnswer:"
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


exceed_count = 0
@torch.no_grad()
def eval(args, subject, model, test_df):
    cors = []
    all_embeddings = []

    for i in range(test_df.shape[0]):
        prompt_end = format_example(test_df, i, include_answer=False)
        prompt = prompt_end

        ## Extracting embeddings for one prompt at a time
        embeddings = model.encode(prompt)

        all_embeddings.append(embeddings)

    return all_embeddings


def main(args):

    data_dir = os.path.join(args.data_folder, args.dataset_name)

    model = SentenceTransformer('sentence-transformers/sentence-t5-xxl')
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

    save_dir = os.path.join(emb_dir, 'sbert')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for subject in subjects:

        test_df = pd.read_csv(
            os.path.join(data_dir, "test", subject + "_test.csv"), header=None
        )

        embeddings = eval(args, subject, model, test_df)

        np.save(os.path.join(save_dir, "{}.npy".format(subject)), embeddings)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=0)
    parser.add_argument("--dataset_name",type=str, default='ai2_arc')
    parser.add_argument("--data_folder", "-d", type=str, default="../../datasets")
    parser.add_argument("--out_dir", "-s", type=str, default="../../results")
    args = parser.parse_args()
    main(args)
