# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, Dict, Iterator, List, Optional

import torch
from robbie.metrics._base import Metric, MetricConfig, MetricResult, Score
from robbie.predictors import Prediction
from robbie.utils import batch_iter
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class HFClassifierMetric(Metric):
    @classmethod
    def from_args(cls, args, **kwargs):
        return HFClassifierMetric(
            name=args.metric,
            config=MetricConfig.from_args(args),
            **kwargs,
        )

    def __init__(
        self,
        name: str = "",
        model_id: str = "",
        labels: List[str] = [],
        config: MetricConfig = None,
        preprocess: Optional[Callable[[Prediction], Prediction]] = None,
        summarize: Optional[Callable[[List[Score]], Dict[str, Any]]] = None,
    ):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._name = name
        self.model_id = model_id
        self.labels = labels
        self.config = config
        self.batch_size = config.batch_size
        self.device = device
        self.preprocess = preprocess
        self.summarize = summarize

    @property
    def name(self):
        return self._name

    @property
    def model(self):
        if not hasattr(self, "_model"):
            model = AutoModelForSequenceClassification.from_pretrained(self.model_id)
            model = model.eval().to(self.device)
            self._model = model
        return self._model

    @property
    def tokenizer(self):
        if not hasattr(self, "_tokenizer"):
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        return self._tokenizer

    def _preprocess(self, predictions: Iterator[Prediction]) -> Iterator[Prediction]:
        if self.preprocess is None:
            return predictions
        for p in predictions:
            yield self.preprocess(p)

    def _summarize(self, scores: List[Score]) -> Dict[str, Any]:
        if self.summarize is None:
            return {}
        return self.summarize(scores)

    def score(self, predictions: Iterator[Prediction]) -> MetricResult:
        scores = []
        for batch in batch_iter(self._preprocess(predictions), self.batch_size):
            inputs = self.tokenizer(
                [p.generation for p in batch],
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(self.model.device)
            outputs = self.model.forward(**inputs)
            conf = torch.softmax(outputs.logits.cpu(), dim=-1)
            batch_scores, batch_labels = torch.topk(conf, dim=-1, k=1)
            for i in range(len(batch)):
                scores.append(
                    Score(
                        prompt=batch[i].prompt,
                        prediction=batch[i].generation,
                        score=batch_scores[i].item(),
                        label=self.labels[batch_labels[i].item()],
                        meta=batch[i].meta,
                    )
                )

        stats = self._summarize(scores)
        return MetricResult(scores=scores, stats=stats)
