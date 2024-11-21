# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the CC BY NC license found in the
# LICENSE-CC-BY-NC file in the project directory.

CONFIG = {
    'DATA_DIR': '../datasets/ai2_arc/test/',
    'MODEL_RESULTS_DIR': '../results/ai2_arc/entire_dataset',
    'NO_QUES_RESULTS_DIR': '../results/ai2_arc/data_contamination',
    'CLUSTER_DIR': '../results/ai2_arc/embeddings/clusters', 
    'EASY_FILTER_THRESHOLD': 0.8,
    'HARD_FILTER_THRESHOLD': 0.8,
    'NOTA_THRESHOLD': 0.8,#None of the above data cont. test threshold
    'MODELS_LIST_FOR_FILTER': ['results_Meta-Llama-3.1-70B-Instruct/', 'results_Yi-34B/', 'results_Qwen2-72B-Instruct/', 'results_Mixtral-8x22B-Instruct-v0.1/', 'results_gemma-2-27b-it/', 'results_dbrx-instruct', 'results_Phi-3-medium-4k-instruct'],
    'NO_QUES_MODELS_FOR_FILTER': ['results_Meta-Llama-3.1-70B-Instruct/', 'results_Yi-34B/', 'results_Qwen2-72B-Instruct/', 'results_Mixtral-8x22B-Instruct-v0.1/', 'results_gemma-2-27b-it/', 'results_dbrx-instruct', 'results_Phi-3-medium-4k-instruct'],
    'MODEL_LIST_FOR_TESTING': ['results_Llama-2-70b-hf', 'results_Meta-Llama-3.1-70B-Instruct', 'results_Mixtral-8x22B-Instruct-v0.1', 'results_OLMo-1.7-7B-hf', 'results_Qwen-7B', 'results_Qwen2-72B-Instruct', 'results_Yi-34B', 'results_falcon-40b', 'results_gemma-2-9b-it', 'results_internlm2_5-20b-chat', 'results_Meta-Llama-3-70B-Instruct', 'results_Mistral-7B-Instruct-v0.2', 'results_Mixtral-8x7B-Instruct-v0.1', 'results_Phi-3-medium-4k-instruct', 'results_Qwen-7B-Chat', 'results_Qwen2-7B-Instruct', 'results_Yi-34B-Chat', 'results_falcon-40b-instruct', 'results_gemma-7b', 'results_internlm2_5-7b-chat', 'results_Meta-Llama-3-8B-Instruct', 'results_Mistral-7B-v0.3', 'results_Mixtral-8x7B-v0.1', 'results_Phi-3.5-MoE-instruct', 'results_Qwen1.5-32B-Chat', 'results_Yi-1.5-9B-Chat', 'results_dbrx-instruct', 'results_gemma-2-27b-it', 'results_gemma-7b-it'],
    'ANS_IDX': 5,
    'TRUE_FALSE_BOOL_IDX_IN_RESULT': 6,
    'MCQ_PROB_START_IDX': 7,
    'MCQ_PROB_END_IDX': 10,
    'QUESTION_IDX': 0,#Index for question string in the dataset 
    'LOG_DIR': '/path/to/logs',
    'ANOMOLOUS_DATASETS': [],#name of anomolous categories
    'COSINE_DISTANCES': ['cosine_sbert_knn=100.h5'],
    'COSINE_KNNS': 100,
    # Add more paths as needed
}