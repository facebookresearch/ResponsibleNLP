#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import ast
import ipdb
import argparse

def parse_results(input_file, output_file, task_id, is_intermediate):
    with open(input_file, "r") as fd:
        annotations = fd.readlines()

    if task_id == 1:
        with open(output_file, "w") as f:
            f.write("worker_id|original|gender_spans|race_spans|age_spans|gender|race|age|\n")
    elif task_id == 2:
        with open(output_file, "w") as f:
            f.write("worker_id|original|gender_spans|race_spans|age_spans|gender|race|age|\n")
    else:
        with open(output_file, "w") as f:
            f.write("worker_id|original|rewrite|demographic_axis|selected_word|original_category|perturbed_category|gender|race|age|feedback|\n")

    for row in annotations:
        try:
            results_dict = ast.literal_eval(row)
            worker_id = results_dict["worker_id"]

            if not results_dict or results_dict["data"]["inputs"] == None:
                continue
            if results_dict['status'] == 'timeout':
                continue
            original = results_dict["data"]["inputs"]["text"].replace("\n", " ")
            original = original.replace("|", "-")
            survey_data = results_dict["data"]["outputs"]["final_data"]["survey"]
            gender = "\t".join(survey_data["gender"])
            race = "\t".join(survey_data["race"]) 
            if "age" in survey_data:
                age = survey_data["age"]
            else:
                age = "" 

            # Check if task results are from the Phase 1 task (word selection)
            if task_id == 1:
                spans_data = results_dict["data"]["outputs"]["final_data"]["spans"]
                gender_spans = spans_data["gender"]
                gender_spans.sort(key=lambda x: x[1])

                race_spans = spans_data["race"]
                race_spans.sort(key=lambda x: x[1])

                age_spans = spans_data["age"]
                age_spans.sort(key=lambda x: x[1])

                # Writing a prettified version for inspection
                gender_spans_str = ",".join([x[0] for x in gender_spans])
                race_spans_str = ",".join([x[0] for x in race_spans])
                age_spans_str = ",".join([x[0] for x in age_spans])

                with open(output_file, "a") as f:
                    f.write(f"{worker_id}|{original}|{gender_spans_str}|{race_spans_str}|{age_spans_str}|{gender}|{race}|{age}|\n")

            # Check if task results are from the Phase 2 task (attribute identification)
            elif task_id == 2:

                spans_data = results_dict["data"]["outputs"]["final_data"]["spans"]
                gender_spans = []
                for word in spans_data["gender"]:
                    for instance in spans_data["gender"][word]:
                        category = spans_data["gender"][word][instance]
                        gender_spans.append("{}={}".format(word, category))
                gender_spans_str = ",".join(gender_spans)

                race_spans = []
                for word in spans_data["race"]:
                    for instance in spans_data["race"][word]:
                        category = spans_data["race"][word][instance]
                        race_spans.append("{}={}".format(word, category))
                race_spans_str = ",".join(race_spans)

                age_spans = []
                for word in spans_data["age"]:
                    for instance in spans_data["age"][word]:
                        category = spans_data["age"][word][instance]
                        age_spans.append("{}={}".format(word, category))
                age_spans_str = ",".join(age_spans)

                with open(output_file, "a") as f:
                    f.write(f"{worker_id}|{original}|{gender_spans_str}|{race_spans_str}|{age_spans_str}|{gender}|{race}|{age}|\n")
            elif task_id == 3:
                rewrite = results_dict["data"]["outputs"]["final_data"]["rewrite"].replace("\n", " ")
                rewrite = rewrite.replace("|", "-")
                demographic_axis = results_dict["data"]["inputs"]["demographic_axis"].strip()
                selected_word = results_dict["data"]["inputs"]["selected_word"].strip()
                original_category = results_dict["data"]["inputs"]["selected_word_category"].strip()
                perturbed_category = results_dict["data"]["inputs"]["perturbed_word_category"].strip()
                feedback = results_dict["data"]["outputs"]["final_data"]["feedback"].replace("\n", " ")

                with open(output_file, "a") as f:
                    f.write(f"{worker_id}|{original}|{rewrite}|{demographic_axis}|{selected_word}|{original_category}|{perturbed_category}|{gender}|{race}|{age}|{feedback}|\n")
        except TypeError as e:
            print(e)
            continue


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        type=str,
    )
    parser.add_argument(
        "--output_file",
        type=str,
    )
    # Indicate data collection phase 1, 2 or 3
    parser.add_argument(
        "--task_id",
        type=int,
    )
    # Indicate whether this is an intermediate file
    parser.add_argument(
        "--is_intermediate",
        action="store_true",
        default=False
    )
    args = parser.parse_args()
    print("Most recent task names:")
    parse_results(args.input_file, args.output_file, args.task_id, args.is_intermediate)


if __name__ == "__main__":
    main()