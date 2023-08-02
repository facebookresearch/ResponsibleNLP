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

python multilingual_gendered_nouns_dist/get_multilingual_gender_dist.py \
    --file_names newstest2019-src.eng.txt newstest2019-ref.arb.txt newstest2019-ref.bel.txt newstest2019-ref.ben.txt newstest2019-ref.bul.txt newstest2019-ref.cat.txt newstest2019-ref.ces.txt newstest2019-ref.ckb-Arab.txt newstest2019-ref.cym.txt newstest2019-ref.dan.txt newstest2019-ref.deu.txt newstest2019-ref.ell.txt newstest2019-ref.est.txt newstest2019-ref.fin.txt newstest2019-ref.fra.txt newstest2019-ref.gle.txt newstest2019-ref.hin.txt newstest2019-ref.hun.txt newstest2019-ref.ind.txt newstest2019-ref.ita.txt newstest2019-ref.jpn.txt newstest2019-ref.kan.txt newstest2019-ref.kat.txt \
        newstest2019-ref.kir.txt newstest2019-ref.kor.txt newstest2019-ref.lit.txt newstest2019-ref.mar.txt newstest2019-ref.mlt.txt newstest2019-ref.nld.txt newstest2019-ref.pan.txt newstest2019-ref.pol.txt newstest2019-ref.por.txt newstest2019-ref.ron.txt newstest2019-ref.rus.txt newstest2019-ref.slk.txt newstest2019-ref.slv.txt newstest2019-ref.spa.txt newstest2019-ref.swe.txt newstest2019-ref.tam.txt newstest2019-ref.tel.txt newstest2019-ref.tha.txt newstest2019-ref.tur.txt newstest2019-ref.urd.txt newstest2019-ref.vie.txt newstest2019-ref.yue.txt newstest2019-ref.zul.txt newstest2019-ref.zho-CN.txt \
    --langs eng arb bel ben bul cat ces ckb cym dan deu ell est fin fra gle hin hun ind ita jpn kan kat kir kor lit mar mlt nld pan pol por ron rus slk slv spa swe tam tel tha tur urd vie yue zul cmn \
    --file_dir $DATA/NTREX/NTREX-128 > ./log_ntrex_final_new_tok.txt

    
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

sys.path.append('.')

from datasets import load_dataset
from multilingual_gendered_nouns_dist.src.gender_counts import MultilingualGenderDistribution
from multilingual_gendered_nouns_dist.src.util import clean_sample, bold



if __name__ == '__main__':
    
    # Load the pre-trained language identification model
    

    parser = argparse.ArgumentParser(description='Example of using argparse')

    parser.add_argument('--max_samples', default=None)
    parser.add_argument('--langs', type=str, nargs='+', required=True)

    #parser.add_argument('--file_dir', type=str, required=False)
    parser.add_argument('--file_dir', type=str, nargs='+', required=False, default=None)
    parser.add_argument('--file_names', type=str, nargs='+', required=False)
    parser.add_argument('--nouns_format_version', type=str, required=False, default='v1.0')
    
    parser.add_argument('--dataset', type=str, required=False)
    parser.add_argument('--split', type=str, default='train', required=False)
    parser.add_argument('--first_level_key', type=str, required=False)
    parser.add_argument('--second_level_key', type=str, default=None)

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
            else:
                dataset = 'NA'
            file_dir = Path(file_dir)
            
            assert (file_dir/file_name).is_file(), f'{file_dir/file_name} not found'
            
            with open(file_dir/file_name) as file:
                hb_counter = MultilingualGenderDistribution(store_hb_dir='./tmp', langs=[lang], ft_model_path='./fasttext_models/lid.176.bin' if  args.lang_detect else None, 
                                               dataset_version=args.nouns_format_version)
                hb_counter.process_txt(file=file, clean_sample=clean_sample, max_samples=args.max_samples, expected_langs=[lang])
                    
            stat = hb_counter.gender_dist(info_file=file_name)
            
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
    else:
        raise(Exception('Argument missing --file_dir or --dataset missing'))
    
    report_df = pd.DataFrame(report_df)
    
    for dataset in report_df['dataset'].unique():
        print(f'\% of words in each gender group {dataset} \n')
        _df = report_df[report_df['dataset']==dataset]
        _df = _df.sort_values("lang")
        for i in range(_df.shape[0]):
            row = _df.iloc[i]            
            display = bold(fem=row['feminine'], masc=row['masculine'], unsp=row['unspecified'], total=row['total'], lang=row['lang'])
            print(f" {display}\\\\" )
        print(f"avg. &  {_df['feminine'].mean():0.3f} ({_df['feminine'].std():0.2f})  &  {_df['masculine'].mean():0.3f} ({_df['masculine'].std():0.2f}) &  {_df['unspecified'].mean():0.3f} ({_df['unspecified'].std():0.2f})& \\bf {_df['total'].mean()} \\\\")

        print(f'report_{dataset}.csv copied')
    