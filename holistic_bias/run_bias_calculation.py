#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Run perplexity measurements on sentences from the HolisticBias dataset, and compile them
to calculate the bias per demographic axis.
"""

import os
from typing import Optional

import numpy as np
from parlai.core.opt import Opt
from parlai.core.params import ParlaiParser
from parlai.core.teachers import register_teacher, DialogTeacher
from parlai.scripts.eval_model import EvalModel
from parlai.tasks.blended_skill_talk.worlds import get_contexts_data
from parlai.utils import logging

from holistic_bias.src.bias_measurements import BiasMeasurementCompiler
from holistic_bias.src.sentences import HolisticBiasSentenceGenerator
from holistic_bias.src.util import DEFAULT_DATASET_VERSION, NONE_STRING, RANDOM_SEED


HOLISTIC_BIAS_TASK = "holistic_bias"


@register_teacher(HOLISTIC_BIAS_TASK)
class HolisticBiasTeacher(DialogTeacher):
    """
    Teacher that selects and loops over the specified number of sentences from
    HolisticBias.
    """

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        super().add_cmdline_args(parser, partial_opt)
        group = parser.add_argument_group("HolisticBiasTeacher arguments")
        group.add_argument(
            "--dataset-version",
            type=str,
            default=DEFAULT_DATASET_VERSION,
            help="Which version of the dataset to load",
        )
        group.add_argument(
            "--use-small-set",
            type="bool",
            default=False,
            help="Use only a small set of descriptors for speed",
        )
        group.add_argument(
            "--use-blenderbot-context",
            type="bool",
            default=False,
            help="Add BlenderBot-style persona strings to the context",
        )
        return parser

    def __init__(self, opt, shared=None):
        opt["datafile"] = "no_file"  # Not needed here
        if opt["world_logs"] is None or opt["world_logs"] == "":
            raise ValueError(
                "--world-logs must be set to specify the output results path!"
            )
        if opt["num_examples"] != -1:
            raise ValueError(
                "--num-examples must be unset so that all sentences can be evaluated!"
            )
        self.id = HOLISTIC_BIAS_TASK
        super().__init__(opt, shared)

    def setup_data(self, path):
        """
        Just respond with the sample message for the model agent to respond to.

        There's only one "turn" to this conversation.
        """
        _ = path  # Unused here

        rng = np.random.default_rng(RANDOM_SEED)

        # Subsample sentences
        sentence_generator_save_folder = os.path.dirname(self.opt["world_logs"])
        sentence_generator = HolisticBiasSentenceGenerator(
            save_folder=sentence_generator_save_folder,
            dataset_version=self.opt["dataset_version"],
            use_small_set=self.opt["use_small_set"],
        )
        filtered_sentences = [
            sentence_metadata
            for sentence_metadata in sentence_generator.sentences
            if (
                sentence_metadata["noun_phrase_type"]
                in [
                    "descriptor_noun",
                    "noun_descriptor",
                ]
                and sentence_metadata["descriptor_gender"] == NONE_STRING
            )
        ]
        # All comparisons should between phrases containing both a noun and a descriptor
        # for the counts to be large enough. We also remove gendered descriptors (e.g.
        # "Latina") because we won't have samples for them for as many nouns as the
        # others.
        logging.info(f"{len(filtered_sentences):d} valid sentences identified.")

        # Load BlendedSkillTalk contexts data
        if self.opt["use_blenderbot_context"]:
            contexts_data = get_contexts_data(self.opt, shared=None)
        else:
            contexts_data = None

        logging.info(
            "Compiling all sentences with optional context. This may take several minutes..."
        )

        for sentence_metadata in filtered_sentences:

            if self.opt["use_blenderbot_context"]:
                # Choose a random BlendedSkillTalk context so that the HolisticBias
                # sentence is in-domain given the bot's training data. Include only the
                # first two lines, which are persona strings, because including Wizard
                # of Wikipedia topics or utterances from another conversation could make
                # the HolisticBias sentence seem non-sensical given that context
                context_pair = rng.choice(contexts_data)
                context = rng.choice(context_pair)
                relevant_context = "\n".join(context.split("\n")[:2])
            else:
                relevant_context = "__SILENCE__"

            modified_sentence_metadata = {
                **sentence_metadata,
                "text": relevant_context,
                "labels": [sentence_metadata["text"]],
            }
            yield modified_sentence_metadata, True


class EvalModelOnHolisticBias(EvalModel):
    @classmethod
    def setup_args(cls):
        parser = super(EvalModelOnHolisticBias, cls).setup_args()
        parser = BiasMeasurementCompiler.add_cmdline_args(parser)
        parser.set_params(task=HOLISTIC_BIAS_TASK, skip_generation=True)
        return parser


if __name__ == "__main__":
    EvalModelOnHolisticBias.main()
    parser_ = EvalModelOnHolisticBias.setup_args()
    opt_ = parser_.parse_args()
    BiasMeasurementCompiler(opt_).compile()
