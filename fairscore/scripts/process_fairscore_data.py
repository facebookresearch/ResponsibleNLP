#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pandas as pd
import argparse
import math
import random

def format_for_parlai(orig, rewrite):
    return [
        [orig, rewrite]
    ]


PERTURBED_CATEGORY_MAP = {
    # Gender categories
    "Female": "woman",
    "Male": "man",
    "Non-Binary": "non-binary",
    "Woman": "woman",
    "Man": "man",
    # Ethnicity categories
    "Asian": "asian",
    "Black or African American": "black",
    "White": "white",
    "Native Hawaiian or Other Pacific Islander": "pacific-islander",
    "Hispanic or Latino": "hispanic",
    "American Indian or Alaska Native": "native-american",
    # Age categories
    "Child (< 18)": "child",
    "Young (18-44)": "young",
    "Middle-aged (45-64)": "middle-aged",
    "Senior (65+)": "senior",
    "Adult (unspecified)": "adult"
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_data",
        default="/checkpoint/rebeccaqian/fairscore/controlled_gen/datasets/2022-03-16/fairscore_03_16_train.jsonl",
        help="Path to .jsonl training data, containing original and rewrite annotations, with fields for controlled generation."
    )
    parser.add_argument(
        "--output_file",
        default="/checkpoint/rebeccaqian/fairscore/controlled_gen/datasets/2022-03-16/parlai_fairscore_03_16_train.jsonl",
        help="Path to .jsonl output file to write ParlAI formatted inputs."
    )
    args = parser.parse_args()

    train_data = args.train_data
    df = pd.read_json(train_data, lines=True)
    # Process perturbed category as control tokens
    df["perturbed_category_map"] = df["perturbed_category"].apply(lambda x: PERTURBED_CATEGORY_MAP[x])
    # Preparing text with control parameters
    # [selected_word] [perturb_category] [marker] [text_input]
    df["text"] = df["selected_word"] + ", " + df["perturbed_category_map"] + " <TEXT> " + df["original"]
    # Formatting data into dialogue format expected by ParlAI
    df["text_map"] = df["text"].map(lambda x: {"id": "orig", "text": x, "episode_done": False})
    df["rewrite_map"] = df["rewrite"].map(lambda x: {"id": "rewrite", "text": x, "episode_done": False})
    df["dialog"] = df.apply(lambda x: format_for_parlai(orig=x["text_map"], rewrite=x["rewrite_map"]), axis=1)
    # Create train/test/valid splits
    data_len = df.index.size
    train_ratio = 0.7
    valid_ratio = 0.2
    test_ratio = 0.1
    df = df.sample(frac=1)
    # Grab indices
    train_len = math.floor(train_ratio * data_len)
    valid_len = math.floor(valid_ratio * data_len)
    train_split = df[:train_len]
    valid_split = df[train_len:(train_len + valid_len)]
    test_split = df[(train_len + valid_len):]
    # Dump output into .jsonl format
    train_split[["dialog"]].to_json(args.output_file + "_train.jsonl", orient="records", lines=True)
    valid_split[["dialog"]].to_json(args.output_file + "_valid.jsonl", orient="records", lines=True)
    test_split[["dialog"]].to_json(args.output_file + "_test.jsonl", orient="records", lines=True)
