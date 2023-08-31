#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Class for generating sentences from the HolisticBias dataset.
"""

import json
import os
import random
import re
from typing import Union, List, Dict, Any, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from holistic_bias.src.util import NO_PREFERENCE_DATA_STRING, NONE_STRING, RANDOM_SEED


class HolisticBiasSentenceGenerator:
    """
    Generate sentences from the HolisticBias dataset, with stylistic variations applied.
    """

    # Constants for building templates
    NONE_STRING = NONE_STRING
    NO_PREFERENCE_DATA_STRING = NO_PREFERENCE_DATA_STRING
    NO_NOUN_TEMPLATE = "{descriptor}"
    WITH_NOUN_TEMPLATE = "{article} {descriptor} {{noun}}"
    NOUN_PHRASE_TEMPLATES = [
        NO_NOUN_TEMPLATE,
        WITH_NOUN_TEMPLATE,
    ]
    # "noun" is wrapped in double curly braces because it will be filled second, after
    # the article and descriptor

    # Other constants
    SORT_COLUMNS = [
        "axis",
        "bucket",
        "descriptor",
        "descriptor_gender",
        "descriptor_preference",
        "noun",
        "plural_noun",
        "noun_gender",
        "noun_phrase",
        "plural_noun_phrase",
        "noun_phrase_type",
    ]
    NUM_DESCRIPTORS_IN_SMALL_SET = 100

    # Path to base dataset folder
    BASE_DATASET_FOLDER = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "dataset"
    )

    @classmethod
    def get_dataset_folder(cls, dataset_version: str) -> str:
        """
        Get the path to the folder containing the dataset, given the input version
        string.
        """
        dataset_folder = os.path.join(cls.BASE_DATASET_FOLDER, dataset_version)
        return dataset_folder

    @classmethod
    def get_descriptors(cls, dataset_version: str) -> Dict[str, dict]:
        """
        Get all descriptors, given the input version string.
        """
        dataset_folder = cls.get_dataset_folder(dataset_version)
        descriptors_path = os.path.join(dataset_folder, "descriptors.json")
        with open(descriptors_path) as f:
            descriptors = json.load(f)
        return descriptors

    @classmethod
    def get_nouns(cls, dataset_version: str) -> Dict[str, list]:
        """
        Get all nouns, given the input version string.
        """
        dataset_folder = cls.get_dataset_folder(dataset_version)
        nouns_path = os.path.join(dataset_folder, "nouns.json")
        with open(nouns_path) as f:
            nouns = json.load(f)
        return nouns

    @classmethod
    def get_sentence_templates(cls, dataset_version: str) -> Dict[str, dict]:
        """
        Get all sentence templates, given the input version string.
        """
        dataset_folder = cls.get_dataset_folder(dataset_version)
        sentence_templates_path = os.path.join(
            dataset_folder, "sentence_templates.json"
        )
        with open(sentence_templates_path) as f:
            sentence_templates = json.load(f)
        return sentence_templates

    @classmethod
    def get_standalone_noun_phrases(cls, dataset_version: str) -> Dict[str, list]:
        """
        Get all standalone noun phrases, given the input version string.
        """
        dataset_folder = cls.get_dataset_folder(dataset_version)
        standalone_noun_phrases_path = os.path.join(
            dataset_folder, "standalone_noun_phrases.json"
        )
        with open(standalone_noun_phrases_path) as f:
            standalone_noun_phrases = json.load(f)
        return standalone_noun_phrases

    @classmethod
    def get_compiled_noun_phrases(cls, dataset_version: str) -> pd.DataFrame:
        """
        Create and return all noun phrases, typically formed from combining a descriptor
        and a noun. Takes as input the descriptor list version string.
        """

        all_noun_phrase_metadata = []

        # Add noun phrases with just nouns and no descriptors
        no_descriptor_template = cls.WITH_NOUN_TEMPLATE.replace(" {descriptor}", "")
        # For instance, this will allow for "I'm a man" as a control for "I'm a blind
        # man"
        no_descriptor_noun_phrase_metadata = []
        for group_gender, gender_noun_tuples in cls.get_nouns(dataset_version).items():
            for noun, plural_noun in gender_noun_tuples:
                noun_phrase = no_descriptor_template.format(
                    article=cls._get_article(noun)
                ).format(noun=noun)
                plural_noun_phrase = (
                    no_descriptor_template.format(article="")
                    .lstrip()
                    .format(noun=plural_noun)
                )
                no_descriptor_noun_phrase_metadata.append(
                    {
                        "axis": "null",
                        "bucket": cls.NONE_STRING,
                        "descriptor": cls.NONE_STRING,
                        "descriptor_gender": cls.NONE_STRING,
                        "descriptor_preference": cls.NONE_STRING,
                        "noun": noun,
                        "plural_noun": plural_noun,
                        "noun_gender": group_gender,
                        "noun_phrase": noun_phrase,
                        "plural_noun_phrase": plural_noun_phrase,
                        "noun_phrase_type": "noun",
                    }
                )
        all_noun_phrase_metadata += no_descriptor_noun_phrase_metadata

        # Loop over all demographic axes and enumerate all possible combinations
        for axis, axis_descriptors in cls.get_descriptors(dataset_version).items():

            # Compile noun phrases and metadata
            this_axis_noun_phrase_metadata = []
            for bucket, descriptor_info in axis_descriptors.items():
                for descriptor_obj in descriptor_info:
                    this_axis_noun_phrase_metadata += cls._get_noun_phrase_metadata(
                        descriptor_obj=descriptor_obj,
                        dataset_version=dataset_version,
                    )
                this_axis_noun_phrase_metadata = [
                    {"bucket": bucket, **noun_phrase_metadata}
                    for noun_phrase_metadata in this_axis_noun_phrase_metadata
                ]
            this_axis_noun_phrase_metadata = [
                {"axis": axis, **noun_phrase_metadata}
                for noun_phrase_metadata in this_axis_noun_phrase_metadata
            ]

            all_noun_phrase_metadata += this_axis_noun_phrase_metadata

        # Add in standalone noun phrases
        standalone_noun_phrase_metadata = []
        standalone_noun_phrases = cls.get_standalone_noun_phrases(dataset_version)
        for axis, axis_noun_phrases in standalone_noun_phrases.items():
            for noun_phrase_obj in axis_noun_phrases:

                # Extract out metadata
                if isinstance(noun_phrase_obj, str):
                    # No metadata found
                    noun_phrase_obj = {"noun_phrase": noun_phrase_obj}
                possibly_templated_noun_phrase = noun_phrase_obj["noun_phrase"]
                possibly_templated_plural_noun_phrase = noun_phrase_obj.get(
                    "plural_noun_phrase", possibly_templated_noun_phrase
                )
                noun_phrase_preference = noun_phrase_obj.get(
                    "preference", cls.NO_PREFERENCE_DATA_STRING
                )
                # Lists whether a noun phrase has been labeled as dispreferred or
                # polarizing

                # Fill in the noun if needed, and add gender metadata
                if "{noun}" in possibly_templated_noun_phrase:
                    nouns = cls.get_nouns(dataset_version)
                    noun_phrase_metadata = [
                        {
                            "axis": axis,
                            "bucket": cls.NONE_STRING,
                            "descriptor": possibly_templated_noun_phrase.format(
                                article="", noun=""
                            ).lstrip(),
                            "descriptor_gender": cls.NONE_STRING,
                            "descriptor_preference": noun_phrase_preference,
                            "noun": noun,
                            "plural_noun": plural_noun,
                            "noun_gender": group_gender,
                            "noun_phrase": possibly_templated_noun_phrase.format(
                                article=cls._get_article(noun), noun=noun
                            ),
                            "plural_noun_phrase": possibly_templated_plural_noun_phrase.format(
                                article="", noun=plural_noun
                            ).lstrip(),
                            "noun_phrase_type": "noun_descriptor",
                        }
                        for group_gender, gender_noun_tuples in nouns.items()
                        for noun, plural_noun in gender_noun_tuples
                    ]
                else:
                    noun_phrase_metadata = [
                        {
                            "axis": axis,
                            "bucket": cls.NONE_STRING,
                            "descriptor": possibly_templated_noun_phrase,
                            "descriptor_gender": cls.NONE_STRING,
                            "descriptor_preference": noun_phrase_preference,
                            "noun": cls.NONE_STRING,
                            "plural_noun": cls.NONE_STRING,
                            "noun_gender": "neutral",
                            "noun_phrase": possibly_templated_noun_phrase,
                            "plural_noun_phrase": possibly_templated_plural_noun_phrase,
                            "noun_phrase_type": "fixed_phrase",
                        }
                    ]
                standalone_noun_phrase_metadata += noun_phrase_metadata

        all_noun_phrase_metadata += standalone_noun_phrase_metadata

        noun_phrase_df = pd.DataFrame(all_noun_phrase_metadata)[
            cls.SORT_COLUMNS
        ].sort_values(cls.SORT_COLUMNS)

        return noun_phrase_df

    @classmethod
    def _get_article(cls, descriptor: str) -> str:
        """
        Return the correct indefinite article for the input descriptor phrase.
        """
        if descriptor[0].lower() in "aeiou":
            return "an"
        else:
            return "a"

    @classmethod
    def _get_noun_phrase_metadata(
        cls,
        descriptor_obj: Union[str, dict],
        dataset_version: str,
    ) -> List[Dict[str, Any]]:
        """
        For the given descriptor (maybe accompanied by additional metadata), enumerate
        all possible noun phrases for that descriptor.
        """

        # Extract out metadata
        if isinstance(descriptor_obj, str):
            # No metadata found
            descriptor_obj = {"descriptor": descriptor_obj}
        descriptor = descriptor_obj["descriptor"]
        descriptor_gender = descriptor_obj.get("gender", cls.NONE_STRING)
        # Set the gender associated with the descriptor, if any
        descriptor_article = descriptor_obj.get("article", cls._get_article(descriptor))
        # Allow for manual specification of the correct indefinite article
        descriptor_preference = descriptor_obj.get(
            "preference", cls.NO_PREFERENCE_DATA_STRING
        )
        # Lists whether a term has been labeled as dispreferred or polarizing

        all_noun_phrase_metadata = []
        for template in cls.NOUN_PHRASE_TEMPLATES:

            # Create the raw noun phrase
            if "{article}" in template:
                noun_phrase = template.format(
                    article=descriptor_article,
                    descriptor=descriptor,
                )
                plural_noun_phrase = template.format(
                    article="", descriptor=descriptor
                ).lstrip()
            else:
                noun_phrase = template.format(descriptor=descriptor)
                plural_noun_phrase = template.format(descriptor=descriptor)

            # Fill in the noun if needed, and add gender metadata
            if "{noun}" in template:
                nouns = cls.get_nouns(dataset_version)
                noun_phrase_metadata = [
                    {
                        "descriptor": descriptor,
                        "descriptor_gender": descriptor_gender,
                        "descriptor_preference": descriptor_preference,
                        "noun": noun,
                        "plural_noun": plural_noun,
                        "noun_gender": noun_gender,
                        "noun_phrase": noun_phrase.format(noun=noun),
                        "plural_noun_phrase": plural_noun_phrase.format(
                            noun=plural_noun
                        ),
                        "noun_phrase_type": "descriptor_noun",
                    }
                    for noun_gender, noun_tuples in nouns.items()
                    for noun, plural_noun in noun_tuples
                    if (
                        descriptor_gender == cls.NONE_STRING
                        or descriptor_gender == noun_gender
                    )
                ]
            else:
                noun_phrase_metadata = [
                    {
                        "descriptor": descriptor,
                        "descriptor_gender": descriptor_gender,
                        "descriptor_preference": descriptor_preference,
                        "noun": cls.NONE_STRING,
                        "plural_noun": cls.NONE_STRING,
                        "noun_gender": cls.NONE_STRING,
                        "noun_phrase": noun_phrase,
                        "plural_noun_phrase": plural_noun_phrase,
                        "noun_phrase_type": "descriptor",
                    }
                ]

            all_noun_phrase_metadata += noun_phrase_metadata

        return all_noun_phrase_metadata

    def __init__(
        self,
        save_folder: str,
        dataset_version: str,
        filters: Optional[Dict[str, Any]] = None,
        use_small_set: bool = False,
    ):
        """
        Create a dataframe of *all* possible templated sentences, including minor
        variants, and include extensive metadata about each sentence. Save all noun
        phrases and sentences as CSVs for future use.

        :param save_folder: the folder to save CSVs to
        :param dataset_version: the string specifying which version of the dataset to
            use
        :param filters: any metadata columns to filter sentences on when looping over
            them
        :param use_small_set: if True, use only a small set of descriptors for
            tractability.
        """

        # Inputs
        if filters is None:
            filters = {}
        suffix = "__small_set" if use_small_set else ""

        # Save paths
        os.makedirs(save_folder, exist_ok=True)
        noun_phrase_path = os.path.join(save_folder, f"noun_phrases{suffix}.csv")
        sentence_path = os.path.join(save_folder, f"sentences{suffix}.csv")

        if os.path.exists(sentence_path):

            print(f"Loading existing file of sentences at {sentence_path}.")
            sentence_df = pd.read_csv(sentence_path)

        else:

            # Load noun phrase dataframe
            print("Generating noun phrases.")
            noun_phrase_df = self.get_compiled_noun_phrases(dataset_version)
            print(f"Number of noun phrases generated: {noun_phrase_df.index.size:d}.")

            # Optionally sample a smaller number of descriptors for speed
            if use_small_set:
                print(
                    f"Sampling a set of {self.NUM_DESCRIPTORS_IN_SMALL_SET:d} descriptors."
                )
                all_descriptors = sorted(noun_phrase_df["descriptor"].unique().tolist())
                rng = np.random.default_rng(RANDOM_SEED)
                selected_descriptors = rng.choice(
                    all_descriptors,
                    size=self.NUM_DESCRIPTORS_IN_SMALL_SET,
                    replace=False,
                )
                noun_phrase_df = noun_phrase_df[
                    lambda df: df["descriptor"].isin(selected_descriptors)
                ]

            # Save noun phrase dataframe
            print(f"Saving noun phrases and metadata to {noun_phrase_path}.")
            noun_phrase_df.to_csv(noun_phrase_path, index=False)

            # Loop over noun phrases, templates, and all variants, and create all
            # possible templated sentences
            print("Looping over noun phrases, templates, and all variants:")
            all_sentence_metadata = []
            sentence_templates = self.get_sentence_templates(dataset_version)
            for _, noun_phrase_series in tqdm(noun_phrase_df.iterrows()):
                noun_phrase_metadata = noun_phrase_series.to_dict()
                if noun_phrase_metadata["noun"] == self.NONE_STRING:
                    # There's no noun phrase here (for instance, maybe it's an adjective
                    # like "Deaf"), so don't use templates that require noun phrases
                    template_choices = {
                        template: specs
                        for template, specs in sentence_templates.items()
                        if not specs.get("must_be_noun", False)
                    }
                else:
                    template_choices = sentence_templates
                for template, template_specs in template_choices.items():
                    template_metadata = {
                        "template": template,
                        "first_turn_only": False,
                        "must_be_noun": False,
                        **template_specs,
                    }

                    fill_values = {}

                    # Format the noun phrase
                    if "{noun_phrase}" in template:
                        fill_values["noun_phrase"] = noun_phrase_metadata["noun_phrase"]
                    elif "{plural_noun_phrase}" in template:
                        fill_values["plural_noun_phrase"] = noun_phrase_metadata[
                            "plural_noun_phrase"
                        ]
                    else:
                        raise ValueError(
                            f'A noun phrase field is not present in the template "{template}"!'
                        )

                    # Fill in all blanks in the template
                    sentence = template.format(**fill_values)

                    # Compile and validate metadata
                    sentence_metadata = {
                        "text": sentence,
                        **noun_phrase_metadata,
                        **template_metadata,
                    }
                    assert not any([val is None for val in sentence_metadata])
                    assert not any([val == "" for val in sentence_metadata])
                    # None or empty-string values are hard to parse when
                    # analyzing

                    all_sentence_metadata.append(sentence_metadata)

            print("Creating a dataframe of all sentences and metadata.")
            sentence_df = pd.DataFrame(all_sentence_metadata)
            print(f"Number of sentences generated: {sentence_df.index.size:d}")

            print(f"Saving sentences and metadata to {sentence_path}.")
            sentence_df.to_csv(sentence_path, index=False)

        for column, allowable_values in filters.items():
            sentence_df = sentence_df[sentence_df[column].isin(allowable_values)]
        print(
            f"Number of sentences remaining after applying any filters: "
            f"{sentence_df.index.size:d}"
        )
        self.sentences = sentence_df.to_dict(orient="records")

    def get_sentence(self) -> Dict[str, Any]:
        """
        Return a randomly selected sentence. Randomly apply a few stylistic variations
        to this sentence.
        """

        selected_sentence_metadata = random.choice(self.sentences).copy()
        sentence = selected_sentence_metadata["text"]
        template = selected_sentence_metadata["template"]
        noun = selected_sentence_metadata["noun"]
        if "{plural_noun_phrase}" in template:
            noun_phrase = selected_sentence_metadata["plural_noun_phrase"]
        elif "{noun_phrase}" in template:
            noun_phrase = selected_sentence_metadata["noun_phrase"]
        else:
            raise ValueError(
                f'Noun phrase pluralization cannot be determined from the template "{template}"!'
            )

        # Apply stylistic variations
        variant_metadata = {
            "remove_im_contraction": random.random() < 0.5,
            "remove_descriptor_hyphens": random.random() < 0.5,
            "lowercase_descriptor": random.random() < 0.5,
            "remove_final_period": random.random() < 0.5,
        }
        variant_noun_phrase = noun_phrase
        # Track the noun phrase as we apply variations

        if variant_metadata["remove_im_contraction"]:
            sentence = re.sub(r"\b{}\b".format(re.escape("I'm")), "I am", sentence)

        if variant_metadata["remove_descriptor_hyphens"]:
            assert (
                "-" not in noun
            ), "The hyphen in the noun will be incorrectly removed!"
            new_variant_noun_phrase = variant_noun_phrase.replace("-", " ")
            sentence = re.sub(
                r"\b{}\b".format(re.escape(variant_noun_phrase)),
                new_variant_noun_phrase,
                sentence,
            )
            variant_noun_phrase = new_variant_noun_phrase

        if variant_metadata["lowercase_descriptor"]:
            assert noun == noun.lower(), "The noun will be incorrectly lowercased!"
            new_variant_noun_phrase = variant_noun_phrase.lower()
            sentence = re.sub(
                r"\b{}\b".format(re.escape(variant_noun_phrase)),
                new_variant_noun_phrase,
                sentence,
            )
            variant_noun_phrase = new_variant_noun_phrase

        if variant_metadata["remove_final_period"]:
            sentence = sentence.rstrip(".")

        selected_sentence_metadata["text"] = sentence

        assert (
            variant_noun_phrase in sentence
        ), "Misalignment between supposed and actual noun phrase!"

        return {
            **selected_sentence_metadata,
            **variant_metadata,
            "variant_noun_phrase": variant_noun_phrase,
        }