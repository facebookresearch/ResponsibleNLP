#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import os
import pandas as pd
from utils import process_jigsaw_1
from utils import process_jigsaw_2
from utils import process_jigsaw_all


"""
The below code generates the AdvPromptSet dataset.
Please refer to README.md for detailed instruction.
"""


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("data", default=".")
    args = parser.parse_args()

    dat_folder = os.path.join(args.data, "jigsaw_data")
    meta_folder = os.path.join(args.data, "metadata")
    out_folder = os.path.join(args.data, "out_data")

    df_jigsaw1 = process_jigsaw_1(dat_folder)
    df_toxic_only, df_has_group = process_jigsaw_2(dat_folder)

    df_jigsaw = process_jigsaw_all(df_jigsaw1, df_toxic_only, df_has_group)

    row_ids = np.load(os.path.join(meta_folder, "advpromptset_rowid.npy"))
    df_jigsaw["row_id"] = row_ids
    df_jigsaw = df_jigsaw[df_jigsaw["row_id"] != -1].reset_index(drop=True).copy()

    df_meta = pd.read_json(os.path.join(meta_folder, "advpromptset_metainfo.jsonl"), lines=True)

    df_advpromptset = df_jigsaw.merge(df_meta, on=["row_id", "id"])

    row_ids_10k = np.load(os.path.join(meta_folder, "advpromptset_rowid_10k.npy"))

    df_advpromptset_10k = df_advpromptset[df_advpromptset["row_id"].isin(row_ids_10k)].reset_index(drop=True).copy()

    with open(os.path.join(out_folder, "advpromptset_final.jsonl"), "w") as f:
        f.write(df_advpromptset.to_json(orient='records', lines=True, force_ascii=False))

    with open(os.path.join(out_folder, "advpromptset_final_10k.jsonl"), "w") as f:
        f.write(df_advpromptset_10k.to_json(orient='records', lines=True, force_ascii=False))
