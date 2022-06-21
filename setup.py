#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from setuptools import setup, find_packages
import sys

if sys.version_info < (3, 8):
    sys.exit("Sorry, only Python >=3.8 has been tested to work with ResponsibleNLP.")

with open("README.md", encoding="utf8") as f:
    readme = f.read()

with open("LICENSE") as f:
    license_ = f.read()

setup(
    name="responsible_nlp",
    version="0.1.0",
    description=(
        "Repository of Responsible NLP projects from Meta AI."
    ),
    long_description=readme,
    license=license_,
    python_requires=">=3.8",
    packages=find_packages(),
)
