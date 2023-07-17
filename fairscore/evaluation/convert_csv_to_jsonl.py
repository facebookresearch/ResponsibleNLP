#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import csv
import json
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv_file",
        required=True,
        help="Input CSV containing perturber and annotator rewrites"
    )
    parser.add_argument(
        "--output_file",
        required=True,
        help="File to write .jsonl output"
    )
    args = parser.parse_args()
    gender_tasks = []
    with open(args.csv_file) as csvfile:
        reader = csv.DictReader(csvfile, delimiter="|")
        for row in reader:
            if (row["original_category"] == "Male" or row["original_category"] == "Female") and (row["perturbed_category"] == "Male" or row["perturbed_category"] == "Female"):
                augly_dict = {
                    "uid": row["worker_id"],
                    "original": row["original"],
                    "text_with_params": row["text_with_params"],
                    "annotator_rewrite": row["rewrite"],
                    "perturber_rewrite": row["perturber_rewrite"]
                }
                gender_tasks.append(augly_dict)
        
    with open(args.output_file, "w") as fd:
        for task in gender_tasks:
            json.dump(task, fd)
            fd.write("\n")


if __name__ == "__main__":
    main()