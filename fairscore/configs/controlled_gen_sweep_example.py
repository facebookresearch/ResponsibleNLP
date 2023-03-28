#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai_internal.projects.param_sweep_utils.param_sweep import ez_grid

# Change these values for your run
PROJECT_NAME = "perturber-90k-04-04"
TOTAL_GPUS = 8
HOURS = 12
# Define param grid
GRID = {
    '-t': ['jsonfile'],
    '-veps': [0.25],
    '--batchsize': [8],
    '--model': ['bart', 'hugging_face/t5'],
    '--fp16': [True],
    '--label-truncate': [512],
    '--log-every-n-secs': [20],
    '-lr': [1e-5],
    '--optimizer': ['adam'],
    '--save-after-valid': [True],
    '--text-truncate': [512],
    '--warmup-updates': [1200],
    '--update-freq': [1],
    '--gradient-clip': [0.1],
    '--skip-generation': [False],
    '-vp': [10],
    '--max-train-time': [HOURS * 60 * 60 - 30 * 60],
    '-vmt': ['bleu-4'],
    '-vmm': 'max',
    '-dynb': ['full'],
    '--load-from-checkpoint': ['true'],
    '--jsonfile-datapath': ['PATH_TO_DATA_1', 'PATH_TO_DATA_2'],
    '--jsonfile-datatype-extension': 'True',
    '--inference': ['nucleus', 'greedy', 'beam'],
    '--topk': [5],
    '--topp': [0.9]
}


if __name__ == '__main__':
    ez_grid(GRID, project_name=PROJECT_NAME, total_gpus=TOTAL_GPUS, hours=HOURS)
