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

python multilingual_gendered_nouns_dist/get_multilingual_gender_dist.py --file_names eng_Latn.devtest arb_Arab.devtest asm_Beng.devtest bel_Cyrl.devtest ben_Beng.devtest bul_Cyrl.devtest cat_Latn.devtest ces_Latn.devtest ckb_Arab.devtest cym_Latn.devtest dan_Latn.devtest deu_Latn.devtest ell_Grek.devtest est_Latn.devtest fin_Latn.devtest fra_Latn.devtest gle_Latn.devtest hin_Deva.devtest hun_Latn.devtest ind_Latn.devtest ita_Latn.devtest jpn_Jpan.devtest kan_Knda.devtest kat_Geor.devtest khk_Cyrl.devtest kir_Cyrl.devtest kor_Hang.devtest lit_Latn.devtest lug_Latn.devtest lvs_Latn.devtest mar_Deva.devtest \
        mlt_Latn.devtest nld_Latn.devtest  pan_Guru.devtest pes_Arab.devtest pol_Latn.devtest por_Latn.devtest ron_Latn.devtest rus_Cyrl.devtest slk_Latn.devtest slv_Latn.devtest spa_Latn.devtest swe_Latn.devtest swh_Latn.devtest tam_Taml.devtest tel_Telu.devtest tgl_Latn.devtest tha_Thai.devtest tur_Latn.devtest urd_Arab.devtest uzn_Latn.devtest vie_Latn.devtest yue_Hant.devtest zul_Latn.devtest zho_Hans.devtest \
    --langs eng arb asm bel ben bul cat ces ckb cym dan deu ell est fin fra gle hin hun ind ita jpn kan kat khk kir kor lit lug lvs mar mlt nld  pan pes pol por ron rus slk slv spa swe swh tam tel tgl tha tur urd uzn vie yue zul cmn\
    --file_dir $DATA/flores200_dataset/devtest/ > ./log_flrores_devtest_new_tok.txt
    
 """

import argparse
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import uuid

sys.path.append('.')

from datasets import load_dataset
from multilingual_gendered_nouns_dist.src.gender_counts import MultilingualGenderDistribution
from multilingual_gendered_nouns_dist.src.util import clean_sample, get_latex_table



if __name__ == '__main__':
    
    # Load the pre-trained language identification model
    

    parser = argparse.ArgumentParser(description='Example of using argparse')

    parser.add_argument('--max_samples', default=None)
    parser.add_argument('--langs', type=str, nargs='+', required=True)

    parser.add_argument('--file_dir', type=str, nargs='+', required=False, default=None)
    parser.add_argument('--file_names', type=str, nargs='+', required=False)
    parser.add_argument('--write_dir', type=str, default='reports')
    parser.add_argument('--nouns_format_version', type=str, required=False, default='v1.0')
    
    parser.add_argument('--dataset', type=str, required=False)
    parser.add_argument('--split', type=str, default='train', required=False)
    parser.add_argument('--first_level_key', type=str, required=False)
    parser.add_argument('--second_level_key', type=str, default=None)

    parser.add_argument('--lang_detect', action='store_true', default=False)
    
    parser.add_argument('--printout_latex', action='store_true', default=True)

    
    args = parser.parse_args()
    report = {} 
    report_df = {'dataset':[], 'lang':[], 'masculine':[], 'feminine':[],  'unspecified':[], 'total':[], 'n_doc_w_match':[]}

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
        report[args.dataset] = f"{args.langs[0]} & {stat['female'][1]:0.3f} & {stat['male'][1]:0.3f} & {stat['neutral'][1]:0.3f} & {stat['total'][0]} \\ % {args.dataset}"

        print(f'REPORT on  {args.dataset}')
        
    # Processing Text file 
    elif args.file_dir is not None:
        if len(args.file_dir) != len(args.file_names):
            args.file_dir = [args.file_dir[0] for _ in args.file_names]

        assert len(args.file_names) == len(args.langs) == len(args.file_dir), f'{len(args.file_names)} <> {len(args.langs)} '
        
        for file_dir, file_name, lang in zip(args.file_dir, args.file_names, args.langs):
            if 'devtest' in file_name and 'flores' in file_dir:
                dataset = 'flores'
            elif 'newstest2019' in file_name and 'NTREX' in file_dir:
                dataset = 'ntrex'
            elif 'oscar' in file_dir:
                dataset = 'oscar'
            else:
                dataset = 'NA'
            file_dir = Path(file_dir)
            
            assert (file_dir/file_name).is_file(), f'{file_dir/file_name} not found'
            
            hb_counter = MultilingualGenderDistribution(store_hb_dir='./tmp', langs=[lang], 
                                                        ft_model_path='./fasttext_models/lid.176.bin' if  args.lang_detect else None, 
                                                        dataset_version=args.nouns_format_version)
            try:
                hb_counter.process_txt_file(file_dir=file_dir/file_name, clean_sample=clean_sample, 
                                        max_samples=args.max_samples, 
                                        expected_langs=[lang])
            except Exception as e:
                print(e)
                print('skipping', file_name)
                continue
                    
            stat = hb_counter.gender_dist(info_file=file_name)
            coverage = pd.Series(hb_counter.n_doc_w_match)
            assert len(coverage.value_counts())>1, f'Lang {lang} No match found: {coverage}'
            
            coverage_stat = coverage.value_counts()[1]/len(coverage)*100
            print(f'LANG: {lang} {len(hb_counter.n_words_per_match)}: {np.mean(hb_counter.n_words_per_match)} avg. words with match \n% {coverage_stat} % doc covered.')

            report[file_name] = f"{lang} & "
            
            for gender in stat.columns:
                if gender != 'total':
                    report[file_name] += f" {stat[gender][1]:0.3f} &"
                    report_df[gender].append(stat[gender][1])
                else:
                    report[file_name] += f" {stat['total'][0]} \\\\ % {file_name}"
                    report_df['total'].append(stat['total'][0])
            report_df['n_doc_w_match'].append(coverage_stat)
            report_df['dataset'].append(dataset)
            report_df['lang'].append(lang)
    else:
        raise(Exception('Argument missing --file_dir or --dataset missing'))
    
    report_df = pd.DataFrame(report_df)
    args.write_dir = Path(args.write_dir)
    short_id = str(uuid.uuid4().int)[:4]
    write_to = (args.write_dir/f'report-{short_id}')
    write_to.mkdir(exist_ok=True)
    write_to = write_to/f'report.csv'
    report_df.to_csv(write_to, index=False)
    print(str(write_to))
    if args.printout_latex:
        get_latex_table(report_df)