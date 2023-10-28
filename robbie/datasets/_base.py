# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import glob
import json
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional


@dataclass
class Prompt:
    text: str
    meta: Dict[str, Any]


@dataclass
class Registration:
    name: str
    path: str
    build: Callable[[argparse.Namespace], None]
    text: Optional[Callable[[Dict], str]] = None
    meta: Optional[Callable[[Dict], Dict]] = None


class Dataset:
    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        parser.add_argument("--dataset", required=True)
        parser.add_argument("--dataset-dir", required=True)
        return parser

    @classmethod
    def register(
        cls,
        name: str,
        path: str,
        build: Callable[[argparse.Namespace], None],
        text: Optional[Callable[[Dict], str]] = None,
        meta: Optional[Callable[[Dict], Dict]] = None,
    ):
        if not hasattr(cls, "registry"):
            cls.registry = {}
        cls.registry[name] = Registration(name, path, build, text, meta)

    @classmethod
    def build(cls, args):
        assert args.dataset in cls.registry, "Unknown: " + args.dataset
        r = cls.registry[args.dataset]

        files = glob.glob(os.path.join(args.dataset_dir, r.path))
        if not files:
            r.build(args)

        path = r.path
        if not os.path.isabs(r.path):
            path = os.path.join(args.dataset_dir, path)

        return JSONLDataset(args.dataset, path, r.text, r.meta)


class JSONLDataset(Dataset):
    def __init__(
        self,
        name: str,
        file_pattern: str,
        text_func: Callable[[Dict], str] = None,
        meta_func: Callable[[Dict], Dict] = None,
    ):
        self._name = name
        self.iter = iter(
            self._get_prompts(file_pattern, text_func, meta_func),
        )

    @property
    def name(self):
        return self._name

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.iter)

    def _get_prompts(self, file_pattern, text_func=None, meta_func=None):
        text_func = text_func or (lambda d: d["prompt_text"])
        meta_func = meta_func or (lambda d: d)
        for path in glob.glob(file_pattern):
            with open(path, "r") as f:
                for line in f:
                    data = json.loads(line.strip())
                    text = text_func(data)
                    meta = meta_func(data)
                    yield Prompt(text=text, meta=meta)
