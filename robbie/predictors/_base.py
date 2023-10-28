# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional

from robbie.datasets import Prompt
from robbie.utils import Registry


@dataclass
class Prediction:
    prompt: str
    generation: str
    meta: Dict[str, Any]


class Predictor(Registry):
    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        super().add_args(parser)
        parser.add_argument("--predictor", type=str, required=True)
        GenerationConfig.add_args(parser)
        return parser

    @classmethod
    def build(cls, args):
        return super().build(args.predictor, args)

    @property
    def name(self):
        raise NotImplementedError()

    def generate(self, _: Iterator[Prompt]) -> Iterator[Prediction]:
        raise NotImplementedError()


@dataclass
class GenerationConfig:
    name: str
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    temperature: Optional[float] = None
    beam_size: int = 1
    max_length: Optional[int] = None
    batch_size: int = 1

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument("--top-k", type=int, required=False)
        parser.add_argument("--top-p", type=float, required=False)
        parser.add_argument("--temperature", type=float, required=False)
        parser.add_argument("--beam-size", type=int, required=False)
        parser.add_argument("--max-length", type=int, required=False)
        parser.add_argument("--batch-size", type=int, required=False, default=1)
        return parser

    @classmethod
    def from_args(cls, args):
        return GenerationConfig(
            name=args.predictor,
            top_k=args.top_k,
            top_p=args.top_p,
            temperature=args.temperature,
            beam_size=args.beam_size,
            max_length=args.max_length,
            batch_size=args.batch_size,
        )
