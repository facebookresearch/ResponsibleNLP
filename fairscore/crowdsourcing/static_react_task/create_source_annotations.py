#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import ast

"""
This script parses a .jsonl file generated by heuristic perturbations
and creates a .csv file where every row contains source text with
a demographic reference.
"""

text_annotations = []

with open("test.txt", "r") as fd:
    nli_annotations = fd.readlines()
    for idx in range(len(nli_annotations)):
        if idx % 4 == 0:
            nli_dict = ast.literal_eval(nli_annotations[idx])
            text_annotations.append(nli_dict["hypothesis"])
        else:
            continue

print(len(text_annotations))
with open("source_text.txt", "w") as fd:
    for row in text_annotations:
        fd.write(row + "\n")