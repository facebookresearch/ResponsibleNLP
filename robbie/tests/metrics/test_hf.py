import unittest

from robbie.metrics import MetricConfig
from robbie.metrics.hf import HFClassifierMetric
from robbie.predictors import Prediction


class TestHFClassifierMetric(unittest.TestCase):
    def test_preprocess(self):
        metric = HFClassifierMetric(
            name="toxigen",
            model_id="tomh/toxigen_roberta",
            labels={0: "nontoxic", 1: "toxic"},
            config=MetricConfig(
                batch_size=1,
            )
        )

        result = metric.score(
            predictions=[
                Prediction(prompt="Input 1 is ", generation="good", meta={}),
                Prediction(prompt="Input 2 is ", generation="bad", meta={}),
            ]
        )

        self.assertEquals(len(result.scores), 2)


if __name__ == "__main__":
    unittest.main()