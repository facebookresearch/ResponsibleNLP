#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.



import argparse
import sys
import uuid

import numpy as np
import pandas as pd

from datasets import load_dataset
from pathlib import Path
sys.path.insert(0, '.')

from gender_gap_pipeline.src.gender_counts import GenderGAP, GENDERS
from gender_gap_pipeline.src.util import clean_sample, get_latex_table, reporting


if __name__ == '__main__':
    
    # Load the pre-trained language identification model
    parser = argparse.ArgumentParser(description='Example of using argparse')

    parser.add_argument('--max_samples', default=None)
    parser.add_argument('--langs', type=str, nargs='+', required=False)

    parser.add_argument('--file_dir', type=str, nargs='+', required=False, default=None)
    parser.add_argument('--file_names', type=str, nargs='+', required=False)
    parser.add_argument('--skip_failed_files', action='store_true', default=False)

    parser.add_argument('--write_dir', type=str, default='reports')
    parser.add_argument('--nouns_format_version', type=str, required=False, default='v1.0')
    
    parser.add_argument('--hf_datasets', type=str, nargs='+', required=False, default=None)
    parser.add_argument('--split', type=str, nargs='+', required=False, default=['test'])
    parser.add_argument('--first_level_key', type=str, nargs='+', required=False, default=None)
    parser.add_argument('--second_level_key', type=str, nargs='+', required=False, default=None)
    
    parser.add_argument('--lang_detect', action='store_true', default=False)
    
    parser.add_argument('--printout_latex', action='store_true', default=False)

    
    args = parser.parse_args()
    report = {} 
    report_df = {'dataset':[], 'lang':[], 'masculine':[], 'feminine':[],  'unspecified':[], 'total':[], 'n_doc_w_match':[], 'ste_diff_fem_masc': []}

    if args.langs is None:
        print('INFO: lang detect activated because no --langs were provided.')
        args.lang_detect = True
    elif args.lang_detect:
        print('INFO: --lang_detect ignored because --langs was provided')
        args.lang_detect = False
        
    
    if not args.lang_detect: 
        assert len(args.langs) > 0 
    else:
        assert args.langs is None
    
    # Processing HF Datasets
    if args.hf_datasets is not None:
        assert args.first_level_key is not None, '--args.first_level_key is required for hf_datasets: Provide one key per hf_datasets provided in --hg_datasets'
        if not args.lang_detect:
            
            if len(args.hf_datasets) < len(args.first_level_key):
                assert len(args.hf_datasets) == 1, '--hf_datasets should be one dataset or a list of datasets as long as --first_level_key'
                # if only one language provided, language assumed to be the same for all files and equal to --langs[0]
                args.hf_datasets = [args.hf_datasets[0] for _ in args.first_level_key]
            if len(args.split) < len(args.first_level_key):
                assert len(args.split) == 1, '--split should be one split or a list of datasets as long as --first_level_key'
                # if only one language provided, language assumed to be the same for all files and equal to --langs[0]
                args.split = [args.split[0] for _ in args.first_level_key]
            if not len(args.hf_datasets) == len(args.langs):
                # if only one language provided, language assumed to be the same for all files and equal to --langs[0]
                args.langs = [args.langs[0] for _ in args.hf_datasets]
        assert len(args.first_level_key) == len(args.hf_datasets) == len(args.split), 'The same number of --hf_datasets, --first_level_key and --split should be provided.'
        if args.second_level_key is not None:
            assert len(args.first_level_key) == len(args.second_level_key), '--second_level_key is not null, it should include as many entries as --first_level_key'
        for i_dataset, dataset_name in enumerate(args.hf_datasets):
            hb_counter = GenderGAP(store_hb_dir='./tmp', lang=args.langs[i_dataset] if args.langs is not None else None, 
                                ft_model_path='./fasttext_models/lid.176.bin' if args.lang_detect else None, 
                                dataset_version=args.nouns_format_version)
            try:
                dataset = load_dataset(dataset_name) 
                hb_counter.process_dataset(dataset, split=args.split[i_dataset], 
                                           first_level_key=args.first_level_key[i_dataset], 
                                           second_level_key=args.second_level_key[i_dataset] if args.second_level_key else None, 
                                           clean_sample=clean_sample, 
                                           max_samples=args.max_samples, 
                                          return_vec=True)
            except Exception as e:    
                if args.skip_failed_files:
                    print(f'WARNING: Skipping {dataset_name} Error: {e} ')
                    continue
                else:
                    raise(e)
            stat = hb_counter.gender_dist(info_file=dataset_name)
            label = dataset_name+'-'+args.split[i_dataset]+'-'+args.first_level_key[i_dataset]
            if args.second_level_key is not None:
                label += '-'+args.second_level_key[i_dataset]
            reporting(report_df, hb_counter, args.langs[i_dataset] if args.langs is not None else 'Predicted', label)
            print(f'REPORT on {dataset_name}')
        
    # Processing text files or .txt.gz files
    elif args.file_dir is not None:
        if len(args.file_dir) != len(args.file_names):
            args.file_dir = [args.file_dir[0] for _ in args.file_names]
        if not args.lang_detect:
            if not len(args.file_names) == len(args.langs):
                # if only one language provided, language assumed to be the same for all files and equal to --langs[0]
                args.langs = [args.langs[0] for _ in args.file_names]
        assert len(args.file_names) == len(args.file_dir), f'{len(args.file_names)} <> {len(args.file_dir)} '
        
        for i_file, (file_dir, file_name) in enumerate(zip(args.file_dir, args.file_names)):
            
            if 'devtest' in file_name and 'flores' in file_dir:
                dataset = 'flores'
            elif 'newstest2019' in file_name and 'NTREX' in file_dir:
                dataset = 'ntrex'
            else:
                dataset = Path(file_dir).name
            file_dir = Path(file_dir)
            
            assert (file_dir/file_name).is_file(), f'{file_dir/file_name} not found'
            
            hb_counter = GenderGAP(store_hb_dir='./tmp', lang=args.langs[i_file] if args.langs is not None else None, 
                                   ft_model_path='./fasttext_models/lid.176.bin' if  args.lang_detect else None, 
                                   dataset_version=args.nouns_format_version)
            try:
                hb_counter.process_txt_file(file_dir=file_dir/file_name, clean_sample=clean_sample,  
                                            max_samples=args.max_samples, 
                                             return_vec=True)
            except Exception as e:    
                if args.skip_failed_files:
                    print(f'WARNING: Skipping {file_name} Error: {e} ')
                    continue
                else:
                    raise(Exception(e))
                    
            hb_counter.gender_dist(info_file=file_name)

            reporting(report_df, hb_counter, args.langs[i_file] if args.langs is not None else 'Predicted', dataset)
            
            
    else:
        raise(Exception('Argument missing --file_dir or --dataset missing'))
    

    report_df = pd.DataFrame(report_df)
    args.write_dir = Path(args.write_dir)    
    args.write_dir.mkdir(exist_ok=True)
    write_to = args.write_dir/f'report.csv'
    report_df.to_csv(write_to, index=False)
    print(str(write_to))
    
    if args.printout_latex:
        get_latex_table(report_df)