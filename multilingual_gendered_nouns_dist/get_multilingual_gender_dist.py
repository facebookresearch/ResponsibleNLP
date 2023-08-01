#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


"""
MultilingualGenderDistribution loads the gendered noun lists and compute gender of given datasets.

The dataset can be loaded from the HF datasets library or from a text file. 


# HF
e.g.  python multilingual_gendered_nouns_dist/get_multilingual_gender_dist.py --dataset "Anthropic/hh-rlhf" --first_level_key 'chosen' --split test --max_samples 10


python multilingual_gendered_nouns_dist/get_multilingual_gender_dist.py  --file_dir /private/home/benjaminmuller/dev/biases/data/flores200_dataset/devtest/ \
    --file_names arb_Arab.devtest bel_Cyrl.devtest vie_Latn.devtest por_Latn.devtest eng_Latn.devtest spa_Latn.devtest  \
    --langs arb bel vie por eng spa \
    --max_samples 100
 """

import argparse
import sys
import pandas as pd
from pathlib import Path

sys.path.append('.')

from datasets import load_dataset
from multilingual_gendered_nouns_dist.src.gender_counts import MultilingualGenderDistribution
from multilingual_gendered_nouns_dist.src.gender_counts import LANGISO
from multilingual_gendered_nouns_dist.src.util import clean_sample



if __name__ == '__main__':
    
    # Load the pre-trained language identification model
    

    parser = argparse.ArgumentParser(description='Example of using argparse')

    parser.add_argument('--max_samples', default=None)
    parser.add_argument('--langs', type=str, nargs='+', required=True)

    #parser.add_argument('--file_dir', type=str, required=False)
    parser.add_argument('--file_dir', type=str, nargs='+', required=False)
    parser.add_argument('--file_names', type=str, nargs='+', required=False)
    parser.add_argument('--nouns_format_version', type=str, required=False, default='v1.0')
    
    parser.add_argument('--dataset', type=str, required=False)
    parser.add_argument('--split', type=str, default='train', required=False)
    parser.add_argument('--first_level_key', type=str, required=False)
    parser.add_argument('--second_level_key', type=str, default=None)

    parser.add_argument('--count_demographics', action='store_true', default=False)
    parser.add_argument('--lang_detect', action='store_true', default=False)

    
    args = parser.parse_args()
    report = {} 
    report_df = {'dataset':[], 'lang':[], 'masculine':[], 'feminine':[],  'unspecified':[], 'total':[]}

    # Processing HF Dataset
    if args.lang_detect: 
        print('Land detect is set to True with --langs provided: the pipeline will check that the identified language is in the list --langs')
    if args.dataset is not None:
        assert len(args.langs) == 1, f'{args.langs} should be of len 1 when processing HF dataset'
        
        hb_counter = MultilingualGenderDistribution(store_hb_dir='./tmp', langs=args.langs, 
                                       ft_model_path='./fasttext_models/lid.176.bin' if args.lang_detect else None, 
                                       dataset_version=args.nouns_format_version)
        dataset = load_dataset(args.dataset) # e.g. "HuggingFaceH4/stack-exchange-preferences"    
        hb_counter.process_dataset(dataset, split=args.split, 
                                first_level_key=args.first_level_key, # 'answers'
                                second_level_key=args.second_level_key,  # 'text'
                                clean_sample=clean_sample, 
                                max_samples=args.max_samples)

        stat = hb_counter.gender_dist()
        report[args.dataset] = f"{LANGISO.get(args.langs[0], args.langs[0])} & {stat['female'][1]:0.3f} & {stat['male'][1]:0.3f} & {stat['neutral'][1]:0.3f} & {stat['total'][0]} \\ % {args.dataset}"

        print(f'REPORT on  {args.dataset}')
        
    # Processing Text file 
    elif args.file_dir is not None:
        if len(args.file_dir) != len(args.file_names):
            args.file_dir = [args.file_dir[0] for _ in args.file_names]

        assert len(args.file_names) == len(args.langs) == len(args.file_dir)
        
        for file_dir, file_name, lang in zip(args.file_dir, args.file_names, args.langs):
            if 'devtest' in file_name and 'flores' in file_dir:
                dataset = 'flores'
            elif 'newstest2019' in file_name and 'NTREX' in file_dir:
                dataset = 'ntrex'
            else:
                dataset = 'NA'
            file_dir = Path(file_dir)
            if not (file_dir/file_name).is_file():
                print(f'Warning:  {file_dir/file_name} not found so skipping {lang}')
                continue
            
            if not Path(f'./multilingual_gendered_nouns_dist/dataset/{args.nouns_format_version}/{lang}_nouns.json').is_file():
                print(f'WARNING: Gender list for {lang} was not found ./holistic_bias/dataset/{args.nouns_format_version}/{lang}_nouns.json: skipping {lang}')
                continue
            with open(file_dir/file_name) as file:
                hb_counter = MultilingualGenderDistribution(store_hb_dir='./tmp', langs=[lang], ft_model_path='./fasttext_models/lid.176.bin' if  args.lang_detect else None, 
                                               dataset_version=args.nouns_format_version)
                hb_counter.process_txt(file=file, clean_sample=clean_sample, max_samples=args.max_samples, expected_langs=[lang])
                    
            stat = hb_counter.gender_dist()
            
            report[file_name] = f"{lang} & "
            
            for gender in stat.columns:
                if gender != 'total':
                    report[file_name] += f" {stat[gender][1]:0.3f} &"
                    report_df[gender].append(stat[gender][1])
                else:
                    report[file_name] += f" {stat['total'][0]} \\\\ % {file_name}"
                    report_df['total'].append(stat['total'][0])
            report_df['dataset'].append(dataset)
            report_df['lang'].append(lang)
    report_df = pd.DataFrame(report_df)
    
    for dataset in report_df['dataset'].unique():
        print(f'\% of words in each gender group {dataset} \n')
        _df = report_df[report_df['dataset']==dataset]
        _df = _df.sort_values("lang")
        for i in range(_df.shape[0]):
            row = _df.iloc[i]
            print(f" {row['lang']} &  {row['feminine']:0.3f}  &   {row['masculine']:0.3f}  & {row['unspecified']:0.3f} & {row['total']}\\\\" )
            
        print('MEAN')
        print(f"avg. &  {_df['feminine'].mean():0.3f} ({_df['feminine'].std():0.2f})  &  {_df['masculine'].mean():0.3f} ({_df['masculine'].std():0.2f}) &  {_df['unspecified'].mean():0.3f} ({_df['unspecified'].std():0.2f})& \\bf {row['total']} \\\\")

        #_df.to_csv(f'report_{dataset}.csv', index=None)
        print(f'report_{dataset}.csv copied')
    
    if args.count_demographics:
        hb_counter.printout_summary_demographics()