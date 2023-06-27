#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


"""
CountHolisticBias loads the holistic biases templates and gender nouns collections and compute gender and demographics distribution of given datasets.

The dataset can be loaded from the HF datasets library or from a text file. 



# HF
e.g.  python holistic_bias/get_stats_hb.py --dataset "Anthropic/hh-rlhf" --first_level_key 'chosen' --split test --max_samples 10

# FILE 
 python holistic_bias/get_stats_hb.py  --file_dir /private/home/benjaminmuller/dev/biases/data/NTREX/NTREX-128 --file_names newstest2019-src.eng.txt newstest2019-ref.spa.txt --langs en spa --max_samples 100
 python holistic_bias/get_stats_hb.py  --file_dir /private/home/benjaminmuller/dev/biases/data/flores200_dataset/dev  /private/home/benjaminmuller/dev/biases/data/flores200_dataset/dev /private/home/benjaminmuller/dev/biases/data/NTREX/NTREX-128 /private/home/benjaminmuller/dev/biases/data/NTREX/NTREX-128 --file_names spa_Latn.dev eng_Latn.dev   newstest2019-ref.spa.txt newstest2019-src.eng.txt --langs spa en  spa en


 """

import argparse
import sys

from pathlib import Path

sys.path.append('.')

from datasets import load_dataset
from holistic_bias.src.hb_counts import CountHolisticBias
from holistic_bias.src.hb_counts import LANGISO


def clean_sample(text: str):
    text = text.replace('\n', ' ')
    return text.strip()


if __name__ == '__main__':
    
    # Load the pre-trained language identification model
    

    parser = argparse.ArgumentParser(description='Example of using argparse')

    

    parser.add_argument('--max_samples', default=None)
    parser.add_argument('--langs', type=str, nargs='+', required=True)

    #parser.add_argument('--file_dir', type=str, required=False)
    parser.add_argument('--file_dir', type=str, nargs='+', required=False)
    parser.add_argument('--file_names', type=str, nargs='+', required=False)
    
    
    parser.add_argument('--dataset', type=str, required=False)
    parser.add_argument('--split', type=str, default='train', required=False)
    parser.add_argument('--first_level_key', type=str, required=False)
    parser.add_argument('--second_level_key', type=str, default=None)

    parser.add_argument('--count_demographics', action='store_true', default=False)

    

    
    args = parser.parse_args()
    report = {} 
    # Processing HF Dataset
    if args.dataset is not None:
        assert len(args.langs) == 1, f'{args.langs} should be of len 1 when processing HF dataset'
        hb_counter = CountHolisticBias(store_hb_dir='./tmp', langs=args.langs, ft_model_path='./fasttext_models/lid.176.bin', only_gender=not args.count_demographics)
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
            file_dir = Path(file_dir)
            with open(file_dir/file_name) as file:
                hb_counter = CountHolisticBias(store_hb_dir='./tmp', langs=[lang], ft_model_path='./fasttext_models/lid.176.bin', only_gender=not args.count_demographics)
                hb_counter.process_txt(file=file, clean_sample=clean_sample, max_samples=args.max_samples, expected_langs=[lang])
        
        #hb_counter.printout_summary()
            
            stat = hb_counter.gender_dist()
            report[file_name] = f"{LANGISO.get(lang, lang)} & {stat['female'][1]:0.3f} & {stat['male'][1]:0.3f} & {stat['neutral'][1]:0.3f} & {stat['total'][0]} \\ % {file_name}"

        print(f'REPORT on  {args.file_names}')
    
    print(f" \\bf Lang & \\bf Female & \\bf Male & \\bf Neutral   & \\bf # words  ")
    for row in report:
        print(report[row])
            
    
    
    if args.count_demographics:
        hb_counter.printout_summary_demographics()