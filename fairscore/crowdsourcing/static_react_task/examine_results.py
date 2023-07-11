#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from mephisto.abstractions.databases.local_database import LocalMephistoDB
from mephisto.tools.data_browser import DataBrowser as MephistoDataBrowser
import argparse

mephisto_db = LocalMephistoDB()
data_browser = MephistoDataBrowser(db=mephisto_db)

# Accept all the submitted HITs
def accept_hits(task_name, output_file):
    task_units = data_browser.get_units_for_task_name(task_name)
    print(f"Approving {len(task_units)} HITs from {task_name}")
    for unit in task_units:
        try:
            if unit.get_status() == 'soft_rejected':
                continue
            data = data_browser.get_data_from_unit(unit)
            with open(output_file, "a") as fd:
                fd.write("{}\n".format(data))
            if unit.get_assigned_agent().db_status in ("approved", "rejected"):
                continue
            if unit.get_status() != 'accepted':
                unit.get_assigned_agent().approve_work()
        except AssertionError as e:
            import ipdb; ipdb.set_trace()
            print(e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task_name",
        type=str,
    )
    parser.add_argument(
        "--output_file",
        type=str
    )
    args = parser.parse_args()
    print("Most recent task names:")
    accept_hits(args.task_name, args.output_file)


if __name__ == "__main__":
    main()