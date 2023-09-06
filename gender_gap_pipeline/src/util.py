#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json

import pandas as pd
from gender_gap_pipeline.src.gender_counts import GENDERS


DEFAULT_DATASET_VERSION = "v1.0"
# Default to the original v1.0 version for compatibility.

RANDOM_SEED = 17

# Template building constants
NONE_STRING = "(none)"  # Use when an attribute is not present
NO_PREFERENCE_DATA_STRING = "no_data"
# Use when it is not known whether a descriptor is preferred

RANK_TO_BOLD_BOS = ["", "\\underline{", "\\bf "]
RANK_TO_BOLD_EOS = ["", "}", ""]


def clean_sample(text: str):
    text = text.replace("\n", " ")
    return text.strip().lower()


def rename_gender(json_dir):
    print("read", json_dir)
    with open(json_dir, "r") as read:
        print("reading", json_dir)
        file = json.load(read)
    if "feminine" not in file:
        file["feminine"] = file["female"]
        del file["female"]
    if "masculine" not in file:
        file["masculine"] = file["male"]
        del file["male"]

    assert len(file) == 3, file
    assert {"feminine", "masculine", "unspecified"} == set(file.keys()), file
    with open(json_dir, "w") as write:
        json.dump(file, write, indent=2, ensure_ascii=False)
    print(f"{write} done")


import numpy as np


def rang_to_bold(rank):
    if rank == 0:
        return RANK_TO_BOLD_BOS[0], RANK_TO_BOLD_EOS[0]
    elif rank == 1:
        return RANK_TO_BOLD_BOS[1], RANK_TO_BOLD_EOS[1]
    elif rank == 2:
        return RANK_TO_BOLD_BOS[2], RANK_TO_BOLD_EOS[2]


def bold(fem, masc, unsp, std, total, lang):
    gender = ["fem", "masc", "unsp"]
    sorting = [fem, masc, unsp]
    sorted = np.argsort(sorting)

    if sorting[sorted[2]] == 0.0:
        gender_2_rank = {gender[pos]: 0 for rank, pos in enumerate(sorted)}
    else:
        gender_2_rank = {gender[pos]: rank for rank, pos in enumerate(sorted)}

    display = [
        RANK_TO_BOLD_BOS[gender_2_rank["fem"]]
        + f"{round(fem, 3):0.3f}"
        + RANK_TO_BOLD_EOS[gender_2_rank["fem"]],
        RANK_TO_BOLD_BOS[gender_2_rank["masc"]]
        + f"{round(masc, 3):0.3f}"
        + RANK_TO_BOLD_EOS[gender_2_rank["masc"]],
        RANK_TO_BOLD_BOS[gender_2_rank["unsp"]]
        + f"{round(unsp, 3):0.3f}"
        + RANK_TO_BOLD_EOS[gender_2_rank["unsp"]],
        f"{round(np.abs(fem-masc), 3):0.3f} ({std:0.4f})",
    ]
    
    latex_line = f" {lang} &  {display[0]} &   {display[1]}  & {display[2]} & {display[3]} & {total}"

    return latex_line


def get_latex_table(report_df):
    for dataset in report_df["dataset"].unique():
        _df = report_df[report_df["dataset"] == dataset]
        _df = _df.sort_values("lang")
        print('lang  & feminine & masculine & unspecified & #words & PCT matched docs.')
        for i in range(_df.shape[0]):
            row = _df.iloc[i]
            display = bold(
                fem=row["feminine"],
                masc=row["masculine"],
                unsp=row["unspecified"],
                std=row["ste_diff_fem_masc"],
                total=row["total"],
                lang=row["lang"],
            )

            print(f" {display} & {row['n_doc_w_match']:0.1f}\\\\")
        print(
            f"avg. &  {_df['feminine'].mean():0.3f} &  {_df['masculine'].mean():0.3f}  &  {_df['unspecified'].mean():0.3f} & {np.mean(np.abs(_df['feminine']-_df['masculine'])):0.3f} ({np.mean(_df['ste_diff_fem_masc']):0.4f})  &  {_df['total'].mean():0.0f} & {_df['n_doc_w_match'].mean():0.1f}\\\\"
        )


def reporting(report_df, hb_counter, lang, dataset):
    for gender in GENDERS:
        report_df[gender].append(hb_counter.stat[gender])

    report_df["total"].append(hb_counter.stat["total"])
    report_df["n_doc_w_match"].append(hb_counter.stat["coverage"])
    report_df["ste_diff_fem_masc"].append(hb_counter.stat["ste_diff_fem_masc"])
    report_df["dataset"].append(dataset)
    report_df["lang"].append(lang)


