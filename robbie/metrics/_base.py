# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List

from robbie.predictors import Prediction
from robbie.utils import Registry


@dataclass
class MetricConfig:
    batch_size: int = 1

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument("--metric-batch-size", type=int, default=1, required=False)
        return parser

    @classmethod
    def from_args(cls, args):
        return MetricConfig(
            batch_size=args.metric_batch_size,
        )


@dataclass
class Score:
    score: float
    label: str
    prompt: str
    prediction: str
    meta: Dict[str, Any]


@dataclass
class MetricResult:
    scores: List[Score]
    stats: Dict[str, Any]


class Metric(Registry):
    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument("--metric", type=str, required=True)
        parser = MetricConfig.add_args(parser)
        return parser

    @classmethod
    def build(cls, args):
        return super().build(args.metric, args)

    @property
    def name(self):
        raise NotImplementedError()

    def score(self, _: Iterator[Prediction]) -> MetricResult:
        raise NotImplementedError()
