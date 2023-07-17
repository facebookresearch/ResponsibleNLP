#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from mephisto.data_model.worker import Worker
from mephisto.abstractions.databases.local_database import LocalMephistoDB
import argparse

def get_name_from_id(db, allowlist_path):
    with open(allowlist_path) as fd:
        allowlist = [x for x in fd.read().splitlines()]

    for worker_id in allowlist:
        worker = Worker.get(db, worker_id)
        worker_name = worker.worker_name
        print(worker_name)


def get_worker_from_name(db, allowlist_path):
    with open(allowlist_path) as fd:
        allowlist = [x for x in fd.read().splitlines()]

    worker_ids = []
    for worker_name in allowlist:
        workers = db.find_workers(worker_name=worker_name)
        if len(workers) == 0:
            # Register worker - this is because we want to whitelist some workers by MTurk ID who are not in our Mephisto DB
            worker = Worker._register_worker(db=db, worker_name=worker_name, provider_type="mturk")
            # Return the Mephisto worker ID that was just created
            worker_ids.append(worker.db_id)
        else:
            worker_ids.append(workers[0].db_id)
    return worker_ids

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mturk_names_list",
        type=str,
        help="List of newline separated MTurk worker IDs."
    )
    args = parser.parse_args()
    db = LocalMephistoDB()
    get_name_from_id(db, args.mturk_names_list)
