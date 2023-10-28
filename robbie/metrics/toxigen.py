# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from robbie.metrics._base import Metric, MetricConfig
from robbie.metrics.hf import HFClassifierMetric


LABELS = {idx: label for idx, label in enumerate(["0", "1"])}


Metric.register(
    name="toxigen",
    factory=lambda args: HFClassifierMetric.from_args(
        args,
        model_id="tomh/toxigen_roberta",
        labels=LABELS,
        config=MetricConfig.from_args(args),
    ),
)
