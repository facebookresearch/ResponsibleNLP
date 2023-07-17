#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import glob
import pandas as pd
import os


split = "train"

for path in glob.glob("/checkpoint/rebeccaqian/fairscore/datasets/roberta/sampled/*valid-perturb-parlai.jsonl"):
    source_text_filename = os.path.basename(path)
    output_path_root = f"/checkpoint/rebeccaqian/fairscore/datasets/roberta/sampled/shard_valid/{source_text_filename}"
    df = pd.read_json(path, orient="records", lines=True)
    # Shard into 10
    chunk_size = int(df.shape[0] / 20)
    for start in range(0, df.shape[0], chunk_size):
        df_subset = df.iloc[start:start + chunk_size]
        df_subset.to_json(f"{output_path_root}-{start}.jsonl", orient="records", lines=True)