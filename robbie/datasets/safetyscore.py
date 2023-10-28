# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

from robbie.datasets._base import Dataset
from robbie.datasets.utils import download_if_missing


def build(args):
    out_dir = os.path.join(args.dataset_dir, "safetyscore")
    remote_files = [
        "https://dl.fbaipublicfiles.com/robbie/datasets/safetyscore/annotated_test_v2.jsonl",
    ]
    download_if_missing(out_dir, remote_files)


Dataset.register(
    name="safetyscore",
    path="safetyscore/annotated_test_v2.jsonl",
    build=build,
    meta=lambda d: {
        k: v for k, v in d.items() if k != "prompt_text"
    }
)
