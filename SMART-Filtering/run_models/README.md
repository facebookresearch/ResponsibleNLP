# Evaluating LLMs for 3 criterias

This folder contains scripts to evaluate Large Language Models (LLMs) on entire evaluation datasets, test for data contamination and getting cosine distances between examples. The evaluation code is a modified version of the MMLU codebase.


## Assumptions 
1. The dataset consists of multiple-choice questions. 
2. The dataset is stored in the `../datasets` folder.


## Running the Evaluation Script

To evaluate a model on the entire dataset and store the output probabilities for each answer option, follow these steps:

```
cd <dataset_name>/entire_dataset

python evaluate_hf.py --model <model_name_as_on_huggingface>
```
Replace <model_name_as_on_huggingface> with the actual name of the model on the Hugging Face hub. The model list tested for SMART paper is present [here](models_tested.txt).

We used these results for filtering easy examples. The model list tested for 7 models used for filtering easy examples for SMART paper is present [here](models_used_for_smart.txt).


## Running Models to Test for Data Contamination

To test the models for data contamination, first modify the dataset using the following commands:

```
cd <dataset_name>/data_contamination
python remove_question.py
```

The above script makes the question string empty. We will evaluate the models with no question input and passing the options in the same order. The modified dataset is stored in `data_contamination/dataset_no_ques` folder.

To evaluate a model on the above no_question inputs and store the output probabilities for each answer option, follow these steps:

```
cd data_contamination

python evaluate_hf.py --model <model_name_as_on_huggingface>
```
Replace <model_name_as_on_huggingface> with the actual name of the model on the Hugging Face hub. The model list tested for 7 models used for testing data contamination for SMART paper is present [here](models_used_for_smart.txt).


## Customizing the Dataset

The current codebase can be evaluated for any Multiple Choice QA dataset. Here we have evaluated 4-choice QA (MMLU and ARC) and 5-choice QA (CommonsenseQA) datasets. The code can be extended to any type of dataset with some edits. To run the code on a different dataset:

1. Ensure the dataset is present in the `../../../datasets` folder.
2. Copy ARC/Commonsense-QA folder and make any minor changes if needed. Please note if you are evaluating 4-choice QA, just changing dataset_name in evaluation scripts in ARC folder will work. Similarly for 5-choice QA, just changing dataset_name argument in evaluation scripts in CommonsenseQA will work

```
python evaluate_hf.py --model <model_name_as_on_huggingface> --dataset_name <dataset_name>

python remove_question.py --dataset_name <dataset_name>
```

The outputs are stored in `../../../results` folder.

# Testing question similarity using cosine embeddings

We calculate bert embeddings for all inputs and save them in results folder. Then we calculate cosine distances between each pair of embeddings.

```
cd similarity_search

python extract_embedding_bert.py --dataset_name <dataset_name>

python cosine_distances.py --embedding_dir ../../results/<dataset_name>/embeddings/sbert --out_dir ../../results/<dataset_name>/embeddings/clusters
```

## Downloading results using git lfs

In place of generating results and embeddings for all models, which is compute extensive, you can download the results for mmlu, arc and commonsenqa using git lfs.

If haven't installed already run ``git lfs install`` and clone the repository again.

Then run 

```
git lfs pull
```

This will download all results in [results](../results) folder. Then unzip the downloaded files using
```
unzip mmlu.zip
unzip ai2_arc.zip
unzip commonsense_qa.zip
```

