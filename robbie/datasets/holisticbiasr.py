# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import csv
import json
import os
import random

from robbie.datasets._base import Dataset
from robbie.datasets.regard import PROMPT_TEMPLATES


def build(args):
    from holistic_bias.src.sentences import HolisticBiasSentenceGenerator

    out_dir = os.path.join(args.dataset_dir, "holisticbiasr")
    os.makedirs(out_dir, exist_ok=True)

    HolisticBiasSentenceGenerator(
        save_folder=out_dir,
        dataset_version="v1.1",
        use_small_set=False,
    )

    noun_phrase_data = []
    with open(os.path.join(out_dir, "noun_phrases.csv"), "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for sample in reader:
            # Only include noun phrase types that actually serve as noun phrases:
            # i.e. remove the "descriptor" noun phrase type, which only consists of
            # a descriptor adjective with no noun (e.g. "left-handed")
            if sample["noun_phrase_type"] in [
                "descriptor_noun",
                "fixed_phrase",
                "noun",
                "noun_descriptor",
            ]:
                noun_phrase_data.append(sample)

    noun_set = set(sample["noun"] for sample in noun_phrase_data)
    assert all(
        "-" not in noun for noun in noun_set
    ), "At least one noun has a hyphen, which would be incorrectly removed!"
    assert all(
        noun == noun.lower() for noun in noun_set
    ), "At least one noun has capital letter(s), which would be incorrectly lowercased!"

    out_file = os.path.join(out_dir, "prompts.jsonl")
    with open(out_file, "w") as f:
        for template_sample in PROMPT_TEMPLATES:
            for noun_phrase_sample in noun_phrase_data:
                noun_phrase = noun_phrase_sample["noun_phrase"]

                # Apply stylistic variations
                # (Copied from https://github.com/facebookresearch/ResponsibleNLP/blob/main/holistic_bias/src/sentences.py )
                variant_metadata = {
                    "remove_descriptor_hyphens": random.random() < 0.5,
                    "lowercase_descriptor": random.random() < 0.5,
                }
                variant_noun_phrase = noun_phrase
                # Track the noun phrase as we apply variations
                if variant_metadata["remove_descriptor_hyphens"]:
                    new_variant_noun_phrase = variant_noun_phrase.replace("-", " ")
                    variant_noun_phrase = new_variant_noun_phrase

                if variant_metadata["lowercase_descriptor"]:
                    new_variant_noun_phrase = variant_noun_phrase.lower()
                    variant_noun_phrase = new_variant_noun_phrase

                # Change to a definite article
                for article in ["a ", "an "]:
                    if variant_noun_phrase.startswith(article):
                        formatted_noun_phrase = (
                            "The " + variant_noun_phrase[len(article) :]
                        )
                        break
                else:
                    raise ValueError(
                        f'"{variant_noun_phrase}" does not start with an indefinite article!'
                    )

                prompt_text = template_sample["template"].format(
                    noun_phrase=formatted_noun_phrase,
                )
                data = {
                    "prompt_text": prompt_text,
                    "formatted_noun_phrase": formatted_noun_phrase,
                    **template_sample,
                    **noun_phrase_sample,
                }

                f.write(json.dumps(data) + "\n")


Dataset.register(
    name="holisticbiasr",
    path="holisticbiasr/prompts.jsonl",
    build=build,
    meta=lambda d: {k: v for k, v in d.items() if k != "prompt_text"},
)
