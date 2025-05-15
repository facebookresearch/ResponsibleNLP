# SMART Filtering Repository
This module contains the code for reproducing the resuls and running SMART Filtering from [Improving Model Evaluation using SMART Filtering of Benchmark
Datasets](https://arxiv.org/abs/2410.20245)

### SMART-Filtered Datasets 
- https://huggingface.co/datasets/vipulgupta/arc-smart
- https://huggingface.co/datasets/vipulgupta/mmlu-smart
- https://huggingface.co/datasets/vipulgupta/commonsense_qa_smart

## Starting

This repository uses Git Large File Storage (LFS) to store large files (these are results files and are optional to download, you can skip this step). To access these files, we need to install git lfs:
```
git lfs install
```

Clone the repository:
```
git clone git@github.com:facebookresearch/ResponsibleNLP.git
cd SMART-Filtering
```


## Installation

```
conda create -n smart -y python=3.10.14
conda activate smart
pip install -r requirements.txt
pip install flash-attn
```


## Getting Started with SMART Filtering

Our methodology consists of four main steps:
1. **Dataset Conversion**: Convert your dataset to our standardized format.
2. **Model Evaluation**: Evaluate models on your dataset and for data contamination.
3. **Cosine Distance Calculation**: Calculate cosine distances in higher embedding spaces.
4. **Filtering**: Filter out low-quality examples from your dataset.


## Step 1: Dataset Conversion

To ensure a smooth pipeline, we convert all datasets to a standardized format, similar to the MMLU dataset. You can find conversion scripts for ARC and CommonsenseQA in the `datasets/scripts` folder.

To convert a new dataset:
- Copy the [ARC](https://www.kaggle.com/datasets/thedevastator/arc-grade-school-science-questions/data) (4-choice QA) or CommonsenseQA (5-choice QA) script.
- Modify the script to match your custom format.
- Save the converted dataset in the `dataset/<dataset_name>` folder.


## Step 2-3: Model Evaluation and Cosine Distance Calculation

Evaluate models on your dataset and measure cosine distances between examples by following steps in [README](run_models/README.md) inside `run_models` folder.

## Step 4: Filtering

After evaluating models and measuring cosine distances, filter out low-quality examples by running:

```
cd filtering
python main.py --dataset <dataset_name>
```

**Configuring a New Dataset**

To filter a new dataset, create a copy of the ``config_mmlu.py`` file and modify it according to your needs.


By following these steps, you'll be able to refine your dataset using the SMART filtering methodology.

## Indexes for SMART-Filtered Datasets 

The original datasets are present in [datasets](datasets) folder and indexes that are present in SMART-Filtered version are present in [here](results/smart_filtered) 

