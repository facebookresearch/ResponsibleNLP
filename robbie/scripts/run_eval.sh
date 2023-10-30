#!/usr/bin/env bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case $1 in
    -d|--dataset)
      DATASET="$2"
      shift 2
      ;;
    -m|--model)
      MODEL="$2"
      shift 2
      ;;
    -s|metric)
      METRIC="$2"
      shift 2
      ;;
    *)
      EXTRA_ARGS+=($1)
      shift 1
      ;;
  esac
done


if [[ $MODEL =~ "gpt2" ]]; then
  PREDICTOR_ARGS="--predictor hf_causal --model-id $MODEL --top-k 40 --temperature 0.7"
elif [[ $MODEL =~ "bloom" ]]; then
  PREDICTOR_ARGS="--predictor hf_causal --model-id $MODEL --top-k 40 --temperature 0.7"
elif [[ $MODEL =~ "opt" ]]; then
  PREDICTOR_ARGS="--predictor hf_causal --model-id $MODEL --top-p 0.9 --temperature 1.0"
fi

declare -A DATASET_ARGS
DATASET_ARGS["advpromptset"]="--dataset advpromptset --max-length 100"
DATASET_ARGS["bold"]="--dataset bold --max-length 100"
DATASET_ARGS["holisticbiasr"]="--dataset holisticbiasr --max-length 100"
DATASET_ARGS["realtoxicityprompts"]="--dataset realtoxicityprompts --max-length 100"
DATASET_ARGS["regard"]="--dataset regard --max-length 30"
DATASET_ARGS["safetyscore"]="--dataset safetyscore --max-length 100"

python robbie/eval.py ${DATASET_ARGS[$DATASET]} $PREDICTOR_ARGS --metric $METRIC "${EXTRA_ARGS[@]}"
