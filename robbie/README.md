# ROBBIE: Robust Bias Evaluation of Large Generative Language Models

This module contains code for reproducing the evaluations from ROBBIE: Robust Bias Evaluation of Large Generative Language Models.

## Installation

```bash
conda create -n robbie -y python=3.10.6
conda activate robbie
pip install .
pip install -r robbie/requirements.txt
```

## Running Evaluations

We include support for automatically fetching most of the benchmarks in the paper (except for AdvPromptSet and SafetyScore, which have specific instructions [here](https://github.com/facebookresearch/ResponsibleNLP/blob/main/AdvPromptSet/README.md) and [here](https://github.com/facebookresearch/ResponsibleNLP/blob/main/robbie/datasets/safetyscore.py), respectively), as well as support for all HuggingFace compatible models via [transformers](https://github.com/huggingface/transformers/tree/main).

These can be launched using `robbie/scripts/run_eval.sh`:

```
robbie/scripts/run_eval.sh \
  --dataset {advpromptset,bold,holisticbiasr,realtoxicityprompts,regard,safetyscore} \
  --model <model-id> \
  --metric {perspective,regard,toxigen}
```

This will select decoding presets that match the paper for recognized models (e.g. top-k=40 and temp=0.7 for the GPT-2 family). See the output of `python robbie/eval.py --help` for the full list of supported arguments.

## Mitigations

In the paper we examine a number of techniques for reducing measured bias and toxicity:
- auto-prompting ([Zhou et al, 2022](https://arxiv.org/abs/2211.01910))
- self-debiasing ([Schick, Udupa and Schütze, 2021](https://arxiv.org/abs/2103.00453))
- adversarial triggering ([Sheng et al, 2020](https://aclanthology.org/2020.findings-emnlp.291/))
- additionally: fine-tuning with CRINGE loss ([Adolphs et al, 2023](https://aclanthology.org/2023.acl-long.493/))

We plan to release components of these methods in this toolkit shortly.

## Extending ROBBIE

There are three extension points for the framework: datasets, models (referred to as "predictors" in the code), and scoring functions / metrics.

### Dataset


```python
import os

from robbie.datasets._base import Dataset


def build(args):
    out_dir = os.path.join(args.dataset_dir, "mydataset")
    # Fetch or generate JSONL file(s) into out_dir


Dataset.register(
    name="mydataset",
    path="mydataset/prompts.jsonl",
    build=build,
)
```

### Predictor

```python
from typing import Iterator

from robbie.predictors._base import Predictor


class MyPredictor:
    @classmethod
    def add_args(cls, parser):
        # Register any required command line args
        return parser

    @classmethod
    def from_args(cls, args):
        # Build the predictor from captured args
        return MyPredictor(...)

    @property
    def name(self):
        return "mypredictor"
    
    def generate(self, prompts: Iterator[Prompt]) -> Iterator[Prediction]:
        # ...

Predictor.register(
    name="mypredictor",
    factory=MyPredictor.from_args,
    add_args=MyPredictor.add_args,
)
```

### Metric

```python
from robbie.metrics._base import Metric, MetricResult, Score
from robbie.predictors import Prediction


class MyMetric:
    @property
    def name(self):
        return "mymetric"


    def score(self, predictions: Iterator[Prediction]) -> MetricResult:
        scores = [
            # ....
        ]
        return MetricResult(scores=scores, stats={})


Metric.register(
    name="mymetric",
    factory=lambda _: MyMetric()
)
```

## Bias Score
To bootstrap the evaluation results for fairness analysis:
```
# For regard or holisticbiasr benchmarks
robbie/fairness_analysis/bootsrap_regard_pct.py \
  --task {holisticbiasr, regard} \
  --input <path to the evaluation result> 

# For bold or safetyscore or advpromptset benchmarks
robbie/fairness_analysis/bootsrap_toxicity_pct.py \
  --task {advpromptset, bold, safetyscore} \
  --input <path to the evaluation result> 
```

For computing bias score for a baseline of 0.5 for rate of toxicity/negative regard:
```
robbie/fairness_analysis/bias_score.py \
  --input <path to the bootsrapped results> \
  --baseline 0.5 
```
