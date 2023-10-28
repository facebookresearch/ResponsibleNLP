# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from robbie.datasets._base import Dataset


README_URL = "https://github.com/facebookresearch/ResponsibleNLP/blob/main/AdvPromptSet/README.md"

def build(args):
    raise RuntimeError(
        f"Could not find files for AdvPromptSet under '{args.dataset_dir}'. "
        "Some of its components require a user agreement and it can't be constructed automatically. "
        f"Please see {README_URL} for more information."
    )


Dataset.register(
    name="advpromptset",
    path="advpromptset/advpromptset_final.jsonl",
    build=build,
)
