#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

## SLURM scripts have a specific format.

### Section1: SBATCH directives to specify job configuration

## job name
#SBATCH --job-name=sample
## filename for job standard output (stdout)
## %j is the job id, %u is the user id
#SBATCH --output=/checkpoint/%u/jobs/sample-%j.out
## filename for job standard error output (stderr)
#SBATCH --error=/checkpoint/%u/jobs/sample-%j.err

## partition name
#SBATCH --partition=learnfair
## number of nodes
#SBATCH --nodes=1

## number of tasks per node
#SBATCH --ntasks-per-node=8


# set up environment
module purge
module load anaconda3/2020.11
module load cuda/11.1
module load cudnn/v8.0.3.33-cuda.11.0
module load NCCL/2.8.3-1-cuda.11.0
conda activate conda_parlai

# run job
NUM_TRAINERS=8
srun --label wrapper_finetune.sh $@ &