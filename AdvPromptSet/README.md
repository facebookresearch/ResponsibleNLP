## AdvPromptSet

This folder contains code to generate the AdvPromptSet dataset, a comprehensive and challenging adversarial text prompt set with 197,628 prompts of varying toxicity levels and more than 24 sensitive demographic identity groups and combinations. 


### Download the two Jigsaw datasets from Kaggle

- [Toxic Comment Classification Challenge](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/data)
- [Jigsaw Unintended Bias in Toxicity Classification](https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification/data).

Organize and unzip the two Jigsaw datasets. Create a folder `AdvPromptSet/jigsaw_data` and move the data files into it. The directory tree structure is shown below. 

```
├── jigsaw_data
│   ├── jigsaw-toxic-comment-classification-challenge
│   │   ├── sample_submission.csv
│   │   ├── sample_submission.csv.zip
│   │   ├── test.csv
│   │   ├── test.csv.zip
│   │   ├── test_labels.csv
│   │   ├── test_labels.csv.zip
│   │   ├── train.csv
│   │   └── train.csv.zip
│   └── jigsaw-unintended-bias-in-toxicity-classification
│       ├── all_data.csv
│       ├── identity_individual_annotations.csv
│       ├── sample_submission.csv
│       ├── test.csv
│       ├── test_private_expanded.csv
│       ├── test_public_expanded.csv
│       ├── toxicity_individual_annotations.csv
│       └── train.csv
```


### Download the metadata of AdvPromptSet

- [Archive Download - metadata.zip](https://dl.fbaipublicfiles.com/AdvPromptSet/metadata.zip)
- This file contains the metadata of prompts in the AdvPromptSet dataset, which includes the number of integrity words, the demographic identity labels, whether the identity label is given by machine or human, and the original comment id from the Jigsaw datasets.

Create a folder called `AdvPromptSet/metadata` and unzip `metadata.zip` into the folder. The directory tree structure is shown below. 

```
├── metadata
│   ├── advpromptset_metainfo.jsonl
│   ├── advpromptset_rowid.npy
│   └── advpromptset_rowid_10k.npy
```


### Installation

Starting in the root folder of the repo, `ResponsibleNLP/`:

```
conda create -n advpromptset -y python=3.10.6 && conda activate advpromptset
pip install .
pip install -r AdvPromptSet/requirements.txt
```


### Generating the dataset

Create a folder `AdvPromptSet/out_data` to store the generated dataset. Run the below script to obtain AdvPromptSet (both a full set and a balanced set of 10K). The output files are in `out_data` folder.

```
mkdir AdvPromptSet/out_data
python AdvPromptSet/main.py
```

The final structure of the directory is shown below.
```
.
├── README.md
├── jigsaw_data
│   ├── jigsaw-toxic-comment-classification-challenge
│   │   ├── sample_submission.csv
│   │   ├── sample_submission.csv.zip
│   │   ├── test.csv
│   │   ├── test.csv.zip
│   │   ├── test_labels.csv
│   │   ├── test_labels.csv.zip
│   │   ├── train.csv
│   │   └── train.csv.zip
│   └── jigsaw-unintended-bias-in-toxicity-classification
│       ├── all_data.csv
│       ├── identity_individual_annotations.csv
│       ├── sample_submission.csv
│       ├── test.csv
│       ├── test_private_expanded.csv
│       ├── test_public_expanded.csv
│       ├── toxicity_individual_annotations.csv
│       └── train.csv
├── main.py
├── metadata
│   ├── advpromptset_metainfo.jsonl
│   ├── advpromptset_rowid.npy
│   └── advpromptset_rowid_10k.npy
├── out_data
│   ├── advpromptset_final.jsonl
│   └── advpromptset_final_10k.jsonl
└── utils.py
```
