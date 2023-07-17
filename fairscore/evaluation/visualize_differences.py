#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pandas as pd
import nltk
from nltk import word_tokenize
import difflib
import argparse


def render_rewrite(original, rewrite, rewrite_label):
    """Creates visualization highlighting changed words between original and perturbed text.

    see https://docs.python.org/3/library/difflib.html for sequence matching library.
    """
    original_split = word_tokenize(original)
    rewrite_split = word_tokenize(rewrite)

    original_ptr = 0
    rewrite_ptr = 0

    original_highlighted_words = []
    rewrite_highlighted_words = []


    seq = difflib.SequenceMatcher(None, original_split, rewrite_split)

    for block in seq.get_matching_blocks():
        a, b, size = block
        if a > original_ptr:
            original_highlighted_words += [x for x in range(original_ptr, a)]
        if b > rewrite_ptr:
            rewrite_highlighted_words += [x for x in range(rewrite_ptr, b)]
        original_ptr = a + size
        rewrite_ptr = b + size

    original_split_highlight = original_split[:]
    for idx in range(len(original_split)):
        if idx in original_highlighted_words:
            original_split_highlight[idx] = "<mark>" + original_split_highlight[idx] + "</mark>"

    rewrite_split_highlight = rewrite_split[:]
    for idx in range(len(rewrite_split)):
        if idx in rewrite_highlighted_words:
            rewrite_split_highlight[idx] = f"<mark class={rewrite_label.lower()}>" + rewrite_split_highlight[idx] + "</mark>"

    # highlight words that were selected
    print(f"""
    <p><b>[Original] </b>{" ".join(original_split_highlight)}</p>
    <p><b>[{rewrite_label}] </b>{" ".join(rewrite_split_highlight)}</p>
    <br />
    """)


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
    generated_outputs_examples = generated_outputs.sample(10)
    print("""
        <!DOCTYPE html>
        <html>
        <head>
        <style>
        .augly {
            background-color: #ffccf9;
        }
        .textflint {
            background-color: #ffcaaf;
        }
        .perturber {
            background-color: #c4faf8;
        }
        .annotator {
            background-color: #dbffd6;
        }
        </style>
        </head>
        <body>

        <h3>Comparison of Perturbation Methods</h3>
        """
    )

    for _, example in generated_outputs_examples.iterrows():
        original = example["original"]
        annotator_rewrite = example["annotator_rewrite"]
        perturber_rewrite = example["perturber_rewrite"]
        augly_rewrite = example["augly_rewrite"]
        textflint_rewrite = example["textflint_rewrite"]
        render_rewrite(original, augly_rewrite, "AugLy")
        render_rewrite(original, textflint_rewrite, "TextFlint")
        render_rewrite(original, perturber_rewrite, "Perturber")
        render_rewrite(original, annotator_rewrite, "Annotator")
        print(f"""
        <hr />
        """)
        
    print(f"""
        </body>
        </html>
        """
    )

if __name__ == "__main__":
    main()