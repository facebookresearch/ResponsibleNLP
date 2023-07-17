#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pandas as pd
from datasets import load_metric
import nltk
from nltk import word_tokenize
import argparse
rouge_score = load_metric("rouge")
bleu_score = load_metric("sacrebleu")


def compute_bleu(predictions, references):
    return bleu_score.compute(
        predictions=[predictions], references=[[references]], lowercase=True
    )['score']

def compute_rouge(predictions, references):
    result = rouge_score.compute(
        predictions=[predictions], references=[references]
    )
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    return result

def levenshtein(predictions, references, normalize=False):
    s1, s2 = predictions, references
    s1 = word_tokenize(s1)
    s2 = word_tokenize(s2)
    if len(s1) < len(s2):
        s1, s2 = s2, s1

    if len(s2) == 0:
        return len(s1)
    
    # Array 0->length of second sequence
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        # Enumerate all the items in the first array
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
        
    if normalize:
        return previous_row[-1]/float(len(s1))
    else:
        return previous_row[-1]

METRICS = [
    {
        'metric_func': compute_bleu,
        'metric_name': 'bleu',
    },
    {
        'metric_func': levenshtein,
        'metric_name': 'levenshtein',
    },
    # Important: Call compute_rogue, reuse result for rogue1/2/L
    {
        'metric_func': compute_rouge,
        'metric_name': 'rouge',
    },
]
MODELS = [
    'augly',
    'textflint',
    'perturber',
]

def compute_metrics(df, metric_func, output_name, model):
    """Applies a transform to a given field and creates a column in dataframe with new metric.
    """
    prediction_name = f"{model}_rewrite"
    df[output_name] = df.apply(lambda x: metric_func(predictions=x[prediction_name], references=x["annotator_rewrite"]), axis=1)
    return df

def compute_outputs(df):
    for model in MODELS:
        for metric in METRICS:
            metric_func = metric['metric_func']
            metric_name = metric['metric_name']
            output_name = f"{metric_name}_{model}"
            compute_metrics(df, metric_func, output_name, model)

def print_outputs(df):
    for metric in METRICS:
        metric_name = metric['metric_name']
        print(f"\n{'*' * 20} {metric_name} scores {'*' * 20}\n")
        for model in MODELS:
            output_name = f"{metric_name}_{model}"
            if metric_name == 'rouge':
                rouge1_name = f"{metric_name}1_{model}"
                df[rouge1_name] = df[output_name].apply(lambda x: x["rouge1"])
                print(f"{rouge1_name}: {df[rouge1_name].mean()}")
                rouge2_name = f"{metric_name}2_{model}"
                df[rouge2_name] = df[output_name].apply(lambda x: x["rouge2"])
                print(f"{rouge2_name}: {df[rouge2_name].mean()}")
                rougeL_name = f"{metric_name}L_{model}"
                df[rougeL_name] = df[output_name].apply(lambda x: x["rougeL"])
                print(f"{rougeL_name}: {df[rougeL_name].mean()}\n")
            else:
                print(f"{model}: {df[output_name].mean()}")      


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv_file",
        required=True,
        help="Path to results of different perturbation methods"
    )
    args = parser.parse_args()
    nltk.download('punkt')
    generated_outputs = pd.read_csv(args.csv_file, sep="|")

    compute_outputs(generated_outputs)
    print_outputs(generated_outputs)

    generated_outputs.to_csv(args.csv_file, sep="|")

if __name__ == "__main__":
    main()