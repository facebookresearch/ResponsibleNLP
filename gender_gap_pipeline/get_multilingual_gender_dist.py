#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import uuid

sys.path.append(".")

from datasets import load_dataset
from gender_gap_pipeline.src.gender_counts import (
    MultilingualGenderDistribution,
    GENDERS,
)
from gender_gap_pipeline.src.util import clean_sample, get_latex_table, reporting


if __name__ == "__main__":

    # Load the pre-trained language identification model
    parser = argparse.ArgumentParser(description="Example of using argparse")

    parser.add_argument("--max_samples", default=None)
    parser.add_argument("--langs", type=str, nargs="+", required=True)

    parser.add_argument("--file_dir", type=str, nargs="+", required=False, default=None)
    parser.add_argument("--file_names", type=str, nargs="+", required=False)
    parser.add_argument("--skip_failed_files", action="store_true", default=True)

    parser.add_argument("--write_dir", type=str, default="reports")
    parser.add_argument(
        "--nouns_format_version", type=str, required=False, default="v1.0"
    )

    parser.add_argument("--dataset", type=str, required=False)
    parser.add_argument("--split", type=str, default="train", required=False)
    parser.add_argument("--first_level_key", type=str, required=False)
    parser.add_argument("--second_level_key", type=str, default=None)

    parser.add_argument("--lang_detect", action="store_true", default=False)

    parser.add_argument("--printout_latex", action="store_true", default=False)

    args = parser.parse_args()
    report = {}
    report_df = {
        "dataset": [],
        "lang": [],
        "masculine": [],
        "feminine": [],
        "unspecified": [],
        "total": [],
        "n_doc_w_match": [],
        "ste_diff_fem_masc": [],
    }

    # Processing HF Dataset
    if args.lang_detect:
        print(
            "Land detect is set to True with --langs provided: the pipeline will check that the identified language is in the list --langs"
        )

    if args.dataset is not None:
        assert (
            len(args.langs) == 1
        ), f"{args.langs} Expecting only 1 language when processing HF dataset"

        hb_counter = MultilingualGenderDistribution(
            store_hb_dir="./tmp",
            langs=args.langs,
            ft_model_path="./fasttext_models/lid.176.bin" if args.lang_detect else None,
            dataset_version=args.nouns_format_version,
        )
        dataset = load_dataset(
            args.dataset
        )  # e.g. "HuggingFaceH4/stack-exchange-preferences"
        hb_counter.process_dataset(
            dataset,
            split=args.split,
            first_level_key=args.first_level_key,  # 'answers'
            second_level_key=args.second_level_key,  # 'text'
            clean_sample=clean_sample,
            max_samples=args.max_samples,
        )

        stat = hb_counter.gender_dist()
        report[
            args.dataset
        ] = f"{args.langs[0]} & {stat['female'][1]:0.3f} & {stat['male'][1]:0.3f} & {stat['neutral'][1]:0.3f} & {stat['total'][0]} \\ % {args.dataset}"
        reporting(report_df, hb_counter, args.langs[0], dataset)
        print(f"REPORT on  {args.dataset}")

    # Processing Text file
    elif args.file_dir is not None:
        if len(args.file_dir) != len(args.file_names):
            args.file_dir = [args.file_dir[0] for _ in args.file_names]

        assert (
            len(args.file_names) == len(args.langs) == len(args.file_dir)
        ), f"{len(args.file_names)} <> {len(args.langs)} "

        for file_dir, file_name, lang in zip(
            args.file_dir, args.file_names, args.langs
        ):
            if "devtest" in file_name and "flores" in file_dir:
                dataset = "flores"
            elif "newstest2019" in file_name and "NTREX" in file_dir:
                dataset = "ntrex"
            elif "oscar" in file_dir:
                dataset = "oscar"
            else:
                dataset = "dataset"
            file_dir = Path(file_dir)

            assert (file_dir / file_name).is_file(), f"{file_dir/file_name} not found"

            hb_counter = MultilingualGenderDistribution(
                store_hb_dir="./tmp",
                langs=[lang],
                ft_model_path="./fasttext_models/lid.176.bin"
                if args.lang_detect
                else None,
                dataset_version=args.nouns_format_version,
            )
            try:
                hb_counter.process_txt_file(
                    file_dir=file_dir / file_name,
                    clean_sample=clean_sample,
                    max_samples=args.max_samples,
                    expected_langs=[lang],
                    return_vec=True,
                )
            except Exception as e:
                if args.skip_failed_files:
                    print(f"Skipping {file_name} Error: {e} ")
                    continue
                else:
                    raise (Exception(e))

            hb_counter.gender_dist(info_file=file_name)

            reporting(report_df, hb_counter, lang, dataset)

    else:
        raise (Exception("Argument missing --file_dir or --dataset missing"))

    report_df = pd.DataFrame(report_df)
    args.write_dir = Path(args.write_dir)
    args.write_dir.mkdir(exist_ok=True)
    write_to = args.write_dir / f"report.csv"
    report_df.to_csv(write_to, index=False)
    print(str(write_to))

    if args.printout_latex:
        get_latex_table(report_df)
