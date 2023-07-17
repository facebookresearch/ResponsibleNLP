#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

echo $SLURMD_NODENAME $SLURM_JOB_ID
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo number of gpus: $NUM_GPUS
torchrun --nproc_per_node=$NUM_GPUS run_finetune.py $@
