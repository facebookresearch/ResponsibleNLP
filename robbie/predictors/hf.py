# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Iterator

import torch
from robbie.datasets import Prompt
from robbie.predictors._base import GenerationConfig, Prediction, Predictor
from robbie.utils import batch_iter
from transformers import AutoModelForCausalLM, AutoTokenizer


def _isset(value):
    return value is not None and value > 0


class HFCausalPredictor(Predictor):
    @classmethod
    def add_args(cls, parser):
        parser.add_argument("--model-id", type=str)
        parser.add_argument("--device", type=str, default="cpu")
        return parser

    @classmethod
    def from_args(cls, args):
        config = GenerationConfig.from_args(args)
        return HFCausalPredictor(
            model_id=args.model_id,
            config=config,
            device=args.device,
        )

    def __init__(
        self,
        model_id: str,
        config: GenerationConfig,
        device: str = "cpu",
    ):
        self.model_id = model_id
        self.config = config
        self.model = AutoModelForCausalLM.from_pretrained(model_id).eval().to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id
            self.model.generation_config.pad_token_id = self.model.config.eos_token_id

    @property
    def name(self):
        return self.model_id.replace("/", "_")

    def _generation_kwargs(self):
        gen_args = {
            "top_k": self.config.top_k,
            "top_p": self.config.top_p,
            "temperature": self.config.temperature,
            "num_beams": self.config.beam_size,
            "max_new_tokens": self.config.max_length,
            "do_sample": (
                _isset(self.config.top_k)
                or _isset(self.config.top_p)
                or _isset(self.config.temperature)
            ),
        }
        return {k: v for k, v in gen_args.items() if v is not None}

    @torch.inference_mode()
    def generate(self, prompts: Iterator[Prompt]) -> Iterator[Prediction]:
        for batch in batch_iter(prompts, self.config.batch_size):
            inputs = self.tokenizer(
                [prompt.text for prompt in batch],
                truncation=True,
                padding=True,
                return_attention_mask=True,
                return_tensors="pt",
            ).to(self.model.device)

            outputs = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                return_dict_in_generate=True,
                **self._generation_kwargs(),
            )

            for i, generation in enumerate(
                self.tokenizer.batch_decode(
                    outputs.sequences.cpu(),
                    skip_special_tokens=True,
                )
            ):
                yield Prediction(
                    prompt=batch[i].text,
                    generation=generation,
                    meta=batch[i].meta,
                )


Predictor.register(
    name="hf_causal",
    factory=HFCausalPredictor.from_args,
    add_args=HFCausalPredictor.add_args,
)
