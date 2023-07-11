#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from mephisto.operations.hydra_config import RunScriptConfig, register_script_config
import os
import nltk
import shutil
import ast
import subprocess
from mephisto.operations.operator import Operator
from mephisto.tools.scripts import load_db_and_process_config
from mephisto.abstractions.blueprints.static_react_task.static_react_blueprint import (
    BLUEPRINT_TYPE_STATIC_REACT,
)
from mephisto.abstractions.blueprints.abstract.static_task.static_blueprint import (
    SharedStaticTaskState,
)
from mephisto.data_model.worker import Worker
from mephisto.utils.qualifications import find_or_create_qualification
from mephisto.utils.qualifications import make_qualification_dict
from mephisto.data_model.qualification import QUAL_EXISTS
from mephisto.tools.scripts import load_db_and_process_config

import hydra
from omegaconf import DictConfig
from dataclasses import dataclass, field
from typing import List, Any


TASK_DIRECTORY = os.path.dirname(os.path.abspath(__file__))

defaults = [
    {"mephisto/blueprint": BLUEPRINT_TYPE_STATIC_REACT},
    {"mephisto/architect": "heroku"},
    {"mephisto/provider": "mturk_sandbox"},
    {"conf": "onboarding_example"},
]


@dataclass
class TestScriptConfig(RunScriptConfig):
    defaults: List[Any] = field(default_factory=lambda: defaults)
    task_dir: str = TASK_DIRECTORY


register_script_config(name="scriptconfig", module=TestScriptConfig)


# TODO it would be nice if this was automated in the way that it
# is for ParlAI custom frontend tasks
def build_task(task_dir):
    """Rebuild the frontend for this task"""

    frontend_source_dir = os.path.join(task_dir, "webapp")
    frontend_build_dir = os.path.join(frontend_source_dir, "build")

    return_dir = os.getcwd()
    os.chdir(frontend_source_dir)
    if os.path.exists(frontend_build_dir):
        shutil.rmtree(frontend_build_dir)
    packages_installed = subprocess.call(["npm", "install"])
    if packages_installed != 0:
        raise Exception(
            "please make sure npm is installed, otherwise view "
            "the above error for more info."
        )

    webpack_complete = subprocess.call(["npm", "run", "dev"])
    if webpack_complete != 0:
        raise Exception(
            "Webpack appears to have failed to build your "
            "frontend. See the above error for more information."
        )
    os.chdir(return_dir)

def get_worker_from_name(db, worker_name: str, provider_type: str):
    workers = db.find_workers(worker_name=worker_name)
    if workers == []:
        # Register worker - this is because we want to whitelist some workers by MTurk ID who are not in our Mephisto DB
        worker = Worker._register_worker(db=db, worker_name=worker_name, provider_type=provider_type)
        # Return the Mephisto worker ID that was just created
        return worker.db_id
    return workers[0].db_id

def grant_qualification(db, allowlist, qualification, provider_type):
    for worker_name in allowlist:
        worker_id = get_worker_from_name(db, worker_name, provider_type)
        worker = Worker.get(db, worker_id)
        worker.grant_qualification(qualification)
        print(f'Granting qualification to MTurk worker {worker_name} with Mephisto ID {worker_id}')

@hydra.main(config_path="hydra_configs", config_name="scriptconfig")
def main(cfg: DictConfig) -> None:
    task_dir = cfg.task_dir
    provider_type = cfg.mephisto.provider._provider_type

    def select_words_onboarding(onboarding_data):
        print("Validating onboarding\n Received input: {}".format(onboarding_data))
        worker_answers = onboarding_data["outputs"]
        try:
            assert set(worker_answers["spans"]["gender"]) == set(['Jessica', 'her', 'his'])
            assert (worker_answers["spans"]["race"] == [] or worker_answers["spans"]["race"] == ['Jessica'])
            assert worker_answers["spans"]["age"] == ['young']
        except AssertionError:
            print("Onboarding attempt failed.")
            return False
        return True

    def identify_attributes_onboarding(onboarding_data):
        print("Validating onboarding\n Received input: {}".format(onboarding_data))
        worker_answers = onboarding_data["outputs"]
        try:
            assert worker_answers["spans"]["gender"]["Jessica"]['2'] == 'woman'
            assert worker_answers["spans"]["gender"]["her"]['7'] == 'woman'
            assert worker_answers["spans"]["gender"]["she"]['10'] == 'woman'
            assert worker_answers["spans"]["race"]["for"]['6'] == 'none'
            assert worker_answers["spans"]["race"]["Japanese"]['14'] == 'asian'
            assert worker_answers["spans"]["age"]["toddlers"]['8'] == 'child'
        except AssertionError:
            print("Onboarding attempt failed.")
            return False
        return True

    def rewrite_sentence_onboarding(onboarding_data):
        """Checking that workers are able to:
        - Perturb gendered markers, eg. "king" -> "queen"
        - Perturb gendered names
        - Write non-binary pronouns
        - Mostly preserve the original text snippets.
        """
        print("Validating onboarding\n Received input: {}".format(onboarding_data))
        worker_answers = onboarding_data["outputs"]
        try:
            q1_rewrite = worker_answers["q1_rewrite"].strip().lower()
            # Check that the worker did not copy and paste the assigned perturbed category
            assert "female" not in q1_rewrite
            # Check correctness of individual tokens
            q1_rewrite_tokens = q1_rewrite.split(" ")
            assert q1_rewrite_tokens[0] == "queen"
            assert q1_rewrite_tokens[1] != "james"
            assert " ".join(q1_rewrite_tokens[2:]) == "has a daughter with her second wife."

            q2_rewrite = worker_answers["q2_rewrite"].strip().lower()
            # Check that the worker did not copy and paste the assigned perturbed category
            assert "non-binary" not in q2_rewrite
            q2_rewrite_tokens = q2_rewrite.split(" ")
            assert q2_rewrite_tokens[0] not in ["he", "she", "it"]
            assert q2_rewrite_tokens[6] not in ["himself", "herself", "itself", "it"]
            assert " ".join(q2_rewrite_tokens[1:6]) == "went to the mall by"
            assert " ".join(q2_rewrite_tokens[7:]) == "and bought some video games."

            q3_rewrite = worker_answers["q3_rewrite"].strip().lower()
            assert q3_rewrite == "people came to copy old photos of loved ones."

        except AssertionError as e:
            print("Onboarding attempt failed.")
            print(e)
            return False
        return True

    def process_words_pair(task_data, gender_spans, gender_options, label, original, tokens, original_newlines):
        try:
            if gender_spans == {}:
                return
            for attribute in gender_spans:
                # idx is the id of the highlighted word in the text, when tokenized with nltk.word_tokenize
                for idx in gender_spans[attribute]:
                    value = gender_spans[attribute][idx]
                    if value == "none":
                        continue
                    for gender_category in gender_options:
                        if gender_category == value or gender_category == "none":
                            continue
                        # Figure out which line the token to highlight is in
                        newline_lengths = [len(x) for x in original_newlines]
                        curr_num_toks = 0
                        x_pos = 0
                        y_pos = 0
                        for line_num, _ in enumerate(newline_lengths):
                            line_length = newline_lengths[line_num]
                            # cast index to int
                            idx = int(idx)
                            if (curr_num_toks + line_length) > idx:
                                # the index is in this row
                                x_pos = line_num
                                y_pos = idx - curr_num_toks

                                task_data.append({
                                    "text": original,
                                    "text_newlines": original_newlines,
                                    "tokens": tokens,
                                    "demographic_axis": label,
                                    "selected_word": attribute,
                                    "selected_word_index": idx,
                                    "x_pos": int(x_pos),
                                    "y_pos": int(y_pos),
                                    "selected_word_category": gender_options[value],
                                    "perturbed_word_category": gender_options[gender_category],
                                })
                                break;
                            curr_num_toks += line_length
        except ValueError as e:
            print(e)
            return

    def process_data_for_phase_3():
        """Phase 3 is the rewrites task, where workers are given a snippet, demographic category
        and highlighted term, and asked to perturb all references to that term.
        """
        task_data = []
        # perturbable categories
        attribute_map = {
            "gender": {
                "man": "Male",
                "woman": "Female",
                "non-binary": "Non-Binary",
                "none": "None or Unspecified",
            },
            "race": {
                "hispanic": "Hispanic or Latino",
                "pacific-islander": "Native Hawaiian or Other Pacific Islander",
                "white": "White",
                "black": "Black or African American",
                "native-american": "American Indian or Alaska Native",
                "asian": "Asian",
                "none": "None or Unspecified",
                "race_words": "None or Unspecified",
            },
            "age": {
                "child": "Child (< 18)",
                "young": "Young (18-44)",
                "middle-aged": "Middle-aged (45-64)",
                "senior": "Senior (65+)",
                "adult": "Adult (unspecified)",
                "none": "None or Unspecified",
            }
        }
        # NOTE: source files are in crowdsourcing/static_react_task
        with open("../../../{}".format(cfg.mephisto.task.source_text), "r") as fd:
            annotations = fd.readlines()

            for row in annotations:
                try:
                    results_dict = ast.literal_eval(row)
                    if not results_dict or results_dict["data"]["inputs"] == None:
                        continue
                    if results_dict['status'] == 'timeout':
                        continue
                    if results_dict["data"]["outputs"] == None:
                        continue
                    original = results_dict["data"]["inputs"]["text"]
                    original_newlines = [nltk.word_tokenize(x) for x in original.split("\n")]
                    spans_data = results_dict["data"]["outputs"]["final_data"]["spans"]
                    gender_spans = spans_data["gender"]
                    race_spans = spans_data["race"]
                    age_spans = spans_data["age"]
                    
                    tokens = results_dict["data"]["inputs"]["tokens"]
                    process_words_pair(task_data, gender_spans, attribute_map["gender"], "gender", original, tokens, original_newlines)
                    process_words_pair(task_data, race_spans, attribute_map["race"], "race", original, tokens, original_newlines)
                    process_words_pair(task_data, age_spans, attribute_map["age"], "age", original, tokens, original_newlines)
                    
                    if len(task_data) == 0:
                        return
                except TypeError as e:
                    print(e)
                    return
            return task_data

    def process_data_for_phase_2():
        task_data = []

        with open("../../../{}".format(cfg.mephisto.task.source_text), "r") as fd:
            annotations = fd.readlines()

            for row in annotations:
                try:
                    results_dict = ast.literal_eval(row)
                    if not results_dict or results_dict["data"]["inputs"] == None:
                        continue
                    if results_dict['status'] == 'timeout':
                        continue
                    original = results_dict["data"]["inputs"]["text"]
                    spans_data = results_dict["data"]["outputs"]["final_data"]["spans"]
                        
                    gender_spans = spans_data["gender"]
                    gender_spans.sort(key=lambda x: x[1])

                    race_spans = spans_data["race"]
                    race_spans.sort(key=lambda x: x[1])

                    age_spans = spans_data["age"]
                    age_spans.sort(key=lambda x: x[1])
                    tokens = results_dict["data"]["inputs"]["tokens"]

                    # If there are no words to annotate, skip
                    if len(gender_spans) == 0 and len(race_spans) == 0 and len(age_spans) == 0:
                        continue

                    task_data.append({
                        "text": original,
                        "tokens": tokens,
                        "spans": {
                            "gender": gender_spans,
                            "race": race_spans,
                            "age": age_spans
                        }
                    })

                except TypeError as e:
                    print(e)
                    return

        # Read source text annotations for bookcorpus only
        # with open("../../../{}".format(cfg.mephisto.task.source_text)) as csvfile:
        #     reader = csv.DictReader(csvfile, delimiter="|")
        #     task_data = []
            # for row in reader:
            #     tokens = row["tokens"]
            #     gender_words = row['gender_spans'].split(",")
            #     race_words = row['race_spans'].split(",")
            #     age_words = row['age_spans'].split(",")
            #     gender_words = [[word, tokens.index(word)] for word in gender_words if word != '']
            #     race_words = [[word, tokens.index(word)] for word in race_words if word != '']
            #     age_words = [[word, tokens.index(word)] for word in age_words if word != '']
            #     task_data.append({
            #         "text": row['original'],
            #         "spans": {
            #             "gender": gender_words,
            #             "race": race_words,
            #             "age": age_words
            #         }
            #     })

        return task_data

    def process_data_for_phase_1():
        with open("../../../{}".format(cfg.mephisto.task.source_text), "rb") as fd:
            source_text = fd.read()

        # Convert to byte strings to ignore newlines
        source_text_lines = source_text.split(b'\r')
        # Decode to string
        source_text_lines = [x.decode('utf-8') for x in source_text_lines]
        task_data = [
            {
                "text": annotation,
                "tokens": nltk.word_tokenize(annotation), 
                "spans": {"gender": [], "race": [], "age": []}
            } 
            for annotation
            in source_text_lines 
            if annotation != ""
        ] 
        return task_data   

    if cfg.mephisto.task.fairscore_task_id == 1:
        task_data = process_data_for_phase_1()
        onboarding_task = select_words_onboarding
    elif cfg.mephisto.task.fairscore_task_id == 2:
        task_data = process_data_for_phase_2()
        onboarding_task = identify_attributes_onboarding
    elif cfg.mephisto.task.fairscore_task_id == 3:
        task_data = process_data_for_phase_3()
        onboarding_task = rewrite_sentence_onboarding
    else:
        print("Invalid FairScore task ID specified. Exiting.")
        return

    # if len(task_data) == 0:
    #     print("No tasks to schedule. Exiting.")
    #     return

    # Include onboarding if specified in the task config
    if "onboarding_qualification" in cfg.mephisto.blueprint:
        shared_state = SharedStaticTaskState(
            static_task_data=task_data,
            validate_onboarding=onboarding_task,
        )    
    else:
        shared_state = SharedStaticTaskState(
            static_task_data=task_data,
        )

    if "qualification" in cfg.mephisto:
        assert "qualification_name" in cfg.mephisto.qualification
        ALLOWLIST_QUALIFICATION = cfg.mephisto.qualification.qualification_name
        db, cfg = load_db_and_process_config(cfg)
        find_or_create_qualification(db, ALLOWLIST_QUALIFICATION)
        allowlist_path = cfg.mephisto.qualification.allowlist_path
        with open(allowlist_path) as fd:
            allowlist = [x for x in fd.read().splitlines()]
            grant_qualification(db, allowlist, ALLOWLIST_QUALIFICATION, provider_type)

        shared_state.qualifications = [
            make_qualification_dict(
                ALLOWLIST_QUALIFICATION,
                QUAL_EXISTS,
                None
            ),
        ]
        import ipdb; ipdb.set_trace()

    build_task(task_dir)

    db, cfg = load_db_and_process_config(cfg)
    operator = Operator(db)

    operator.validate_and_run_config(cfg.mephisto, shared_state)
    operator.wait_for_runs_then_shutdown(skip_input=True, log_rate=30)


if __name__ == "__main__":
    main()
