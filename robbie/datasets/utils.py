# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import List

import requests


def download_if_missing(local_dir: str, remote_paths: List[str]):
    os.makedirs(local_dir, exist_ok=True)
    for remote_path in remote_paths:
        local_path = os.path.join(local_dir, os.path.basename(remote_path))
        if os.path.exists(local_path):
            continue
        response = requests.get(remote_path)
        response.raise_for_status()
        with open(local_path, "wb") as f:
            f.write(response.content)
