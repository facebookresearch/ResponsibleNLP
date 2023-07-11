
#!/bin/bash

echo $SLURMD_NODENAME $SLURM_JOB_ID
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo number of gpus: $NUM_GPUS
torchrun --nproc_per_node=$NUM_GPUS run_finetune.py $@
