# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the CC BY NC license found in the
# LICENSE-CC-BY-NC file in the project directory.

from datasets import get_dataset_config_names 
import os, csv, pdb

'''
Download ARC dataset from Kaggle: [https://www.kaggle.com/datasets/thedevastator/arc-grade-school-science-questions/data](https://www.kaggle.com/datasets/thedevastator/arc-grade-school-science-questions/data)
and save it in dataset_name folder
'''
dataset_name = 'allenai/ai2_arc'

out_dir = '../' + dataset_name.split('/')[-1]

##Creating dataset folder
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

subsets = get_dataset_config_names(dataset_name)

'''
out_file = full path of csv file to be saved
dataset = list of lists of data to be saved in csv
'''
def write_csv(dataset, out_file) :
    with open(out_file, 'w+', encoding='UTF8', newline='') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerows(dataset)

## This will vary across datasets
## We need to convert answers to A,B,C,D 
ans_mapping = {'1': 'A', '2': 'B', '3': 'C', '4': 'D'}

## The column names might be different in different datasets
## Change the key names for data_entry accordingly 
for subset in subsets:
    print (f'Saving for {subset}')
    dataset = load_dataset(dataset_name, subset)

    splits = dataset.keys()

    for split in splits:
        split_data = dataset[split]
        if split in ['validation', 'val']:
            split = 'val'
        save_dir = os.path.join(out_dir, split) 
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        dataset_mmlu_format = []
        for idx in range(len(split_data)):
            data_entry = split_data[idx]
            mmlu_entry = []
            mmlu_entry.append(data_entry['question'])
            if len(data_entry['choices']['text']) != 4:
                continue
            mmlu_entry += data_entry['choices']['text']
            ans = data_entry['answerKey']
            if ans not in ['A', 'B', 'C', 'D']:
                ans = ans_mapping[ans]
            mmlu_entry += [ans]            
            mmlu_entry += [data_entry['id']]

            dataset_mmlu_format.append(mmlu_entry)


        write_csv(dataset_mmlu_format, os.path.join(save_dir, subset + '_' + split + '.csv'))