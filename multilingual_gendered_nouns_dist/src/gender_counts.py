#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import re
import sys
import typing as tp

from collections import Counter
from pathlib import Path
from tqdm import tqdm

import os 
import requests
import pandas as pd
import fasttext
import stanza 
import json 



LANGISO = {'arb': 'ar','bel':'be','eng': 'en','ben': 'bn', 'cym': 'cy', 'hun': 'hu', 'lit': 'lt', 'pes': 'fa', 'tam': 'ta', 'urd': 'ur', 'bul': 'bg', 'deu': 'de', 'ind': 'id', 'lug': 'lg', 'por': 'pt', 'tel': 'te', 'vie': 'vi', 'cat': 'ca', 'est': 'et', 'ita': 'it', 'mar': 'mr', 'slv': 'sl', 'tgl': 'tl', 'zul': 'zu', 'ckb': 'ckb', 'fra': 'fr', 'kan': 'kn', 'mlt': 'mt', 'spa': 'es', 'tha': 'th', 'cmn': 'zh', 'hin': 'hi', 'kat': 'ka', 'pan': 'pa', 'swh': 'sw', 'tur': 'tr'}
LANG = {'ar': 'arb','be':'bel', 'en': 'eng','bn': 'ben', 'cy': 'cym', 'hu': 'hun', 'lt': 'lit', 'fa': 'pes', 'ta': 'tam', 'ur': 'urd', 'bg': 'bul', 'de': 'deu', 'id': 'ind', 'lg': 'lug', 'pt': 'por', 'te': 'tel', 'vi': 'vie', 'ca': 'cat', 'et': 'est', 'it': 'ita', 'mr': 'mar', 'sl': 'slv', 'tl': 'tgl', 'zu': 'zul', 'ckb': 'ckb', 'fr': 'fra', 'kn': 'kan', 'mt': 'mlt', 'es': 'spa', 'th': 'tha', 'zh': 'cmn', 'hi': 'hin', 'ka': 'kat', 'pa': 'pan', 'sw': 'swh', 'tr': 'tur'}


GENDER_LS = ['masculine', 'feminine', 'unspecified']


def count_lines(filename):
    with open(filename, 'r') as file:
        return sum(1 for line in file)


class MultilingualGenderDistribution(object):
    """
    This is the core class for computing Holistic Biases distribution. 
    It does the counting based on Holistic_Bias list of nouns and noun phrases, return counters

    """

    def __init__(
        self,
        store_hb_dir: str,
        ft_model_path: str,
        langs: list = ['en'], 
        dataset_version='v1.0'
        
    ) -> None:
        
        store_hb_dir = Path(store_hb_dir)
        store_hb_dir.mkdir(exist_ok=True)
                
        if ft_model_path:
            self.lang_detect_model = fasttext.load_model(ft_model_path) 
        else:
            print('WARNING: self.lang_detect_model set to None cause ft_model_path is None')
            self.lang_detect_model = None
        


        self.noun_phrases = {lang: {} for lang in langs}
        self.gender_regs = {lang: {} for lang in langs}
        self.gender_ls =  {lang: {} for lang in langs}
        self.gender_counters =  {lang: {} for lang in langs}
        base_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dataset")
        dataset_folder = os.path.join(base_folder, dataset_version)
        
        
        self.supported_langs = langs
        self.stanza_tokenizer = {}
        self.nouns = {}
        for lang in self.supported_langs:
            
                
            lang_iso = LANGISO.get(lang, lang)
            
            try:
                stanza.download(LANGISO.get(lang, lang))
                self.stanza_tokenizer[lang] = stanza.Pipeline(lang=lang_iso, processors='tokenize', tokenize_no_ssplit=True)
            except requests.exceptions.ConnectionError as e:
                print('WARNING: Stanza tokenizer coult not be loaded due to Connection Error so white-space tokenization instead')
                self.stanza_tokenizer[lang] = None
            except:
                print('WARNING: Stanza tokenizer cound not be loaded so white-space tokenization instead')
                self.stanza_tokenizer[lang] = None

            with open(os.path.join(dataset_folder, f"{lang}_nouns.json") ) as f:
                self.nouns[lang] = json.load(f)
            for group_gender, gender_noun_tuples in self.nouns[lang].items():
                
                if group_gender not in self.gender_ls[lang]:
                    self.gender_ls[lang][group_gender] = []

                if dataset_version == 'v1.0':
                    for nouns_dict in gender_noun_tuples:     
                        assert 'singular' in nouns_dict or 'plural' in nouns_dict, f'boken dictionary: {nouns_dict}'
                        if 'singular' in nouns_dict:
                            self.gender_ls[lang][group_gender].append(nouns_dict['singular'].lower())
                        if 'plural' in nouns_dict:
                            self.gender_ls[lang][group_gender].append(nouns_dict['plural'].lower())
                else:
                    raise(Exception(f'{dataset_version} not supported'))
                
                self.gender_counters[lang][group_gender] = Counter(self.gender_ls[lang][group_gender])
        
        # Language agnostic counters
        
        self.count_np: Counter = Counter()
        self.count_gender: Counter = Counter()


    def count_demographics(self, line: str, lang: str)-> None:
        sentence = line.strip()
        # lines counter
        assert lang in self.supported_langs, f'{lang} not in {self.supported_langs}'

        # for gender, we count words instead of lines, so we do basic tokenization (eng only)
        if  self.stanza_tokenizer[lang]:
            sentences = self.stanza_tokenizer[lang](sentence).sentences
            # verify that segmentation lead to a single sentence
            if len(sentences) == 0:
                print(f'Warning: empty sentence: {lang}')
                return 
            assert len(sentences) == 1, sentences
        
            tokenized_sentence = sentences[0]
            cc = len(tokenized_sentence.tokens)
            tokenized_sentence = [token.text.lower() for token in tokenized_sentence.tokens]
            assert len(tokenized_sentence) == cc
        else:
            tokenized_sentence = sentence.split(' ')
        
        
        self.count_gender["_total"] += len(tokenized_sentence) # count words

        curr_dic = {key: 0 for key in self.gender_counters[lang]}
        for group_gender in self.gender_counters[lang].keys():            
            # match to list of nouns that are feminine and masculine 
            total_match = 0
            #if len(reg.findall(sentence)):
            tokenized_sentence_counts = Counter(tokenized_sentence)
            common_elements = set(tokenized_sentence_counts) & set(self.gender_counters[lang][group_gender])
                
            for element in common_elements: 
                total_match += tokenized_sentence_counts[element]
            
            self.count_gender[group_gender] += total_match
            curr_dic[group_gender] += total_match
        return curr_dic
            

    def detect_language(self, text: str):
        # Predict the language using the FastText model
        predictions = self.lang_detect_model.predict(text)
        # Extract the predicted label (language code)
        
        label = predictions[0][0].replace('__label__', '')
        return label
    

    def process_lines(self, lines_with_number: tp.Iterator[tp.Tuple[int, str]], lang: str) -> None:
        # iterate over lines
        for line in lines_with_number:
            #TODO: implement lang detect
            self.count_demographics(line, lang)


    def process_dataset(self, dataset: tp.Iterator[tp.Tuple[int, str]], split: str, 
                        first_level_key: str, second_level_key: str=None, 
                        clean_sample: tp.Callable=None, 
                        max_samples: int=None,
                        verbose: int =1) -> None:
        # iterate over lines
        n_sample_counted = 0
        if max_samples is not None:
            max_samples = int(max_samples)
        for i, sample in tqdm(enumerate(dataset[split])):
            
            first_level_val = sample[first_level_key]
            if second_level_key:
                for sample in first_level_val:
                    sample = sample[second_level_key]
                    if clean_sample is not None:
                        sample = clean_sample(sample) 
                    n_sample_counted+= 1
                    lang_detected = self.detect_language(text=sample)
                    self.count_demographics(sample, lang_detected)
            else:
                if clean_sample is not None:
                    first_level_val = clean_sample(first_level_val)
                n_sample_counted+= 1
                if self.lang_detect_model is not None:
                    lang_detected = self.detect_language(text=first_level_val)
                else:
                    print("WARNING: default lang 'en' ")
                    lang_detected = 'en'
                
                lang_detected = LANG.get(lang_detected, lang_detected)                
                self.count_demographics(first_level_val, lang_detected)
            if max_samples is not None:
                if n_sample_counted>=max_samples:
                    break

        if verbose:
            print(f'{n_sample_counted} samples were counted')
    
    def process_txt(self, file,
                    clean_sample: tp.Callable=None, 
                    max_samples: int=None,
                    expected_langs: list = None,
                    verbose: int =1) -> None:
        # iterate over lines
        n_sample_counted = 0
        if max_samples is not None:
            max_samples = int(max_samples)
        for i, sample in tqdm(enumerate(file), desc=f'Processing {file.name}'):
            n_sample_counted+=1
            sample = clean_sample(sample)
            
            if self.lang_detect_model:
                lang_detected = self.detect_language(text=sample)
                lang_detected = LANG.get(lang_detected, lang_detected)
            else:
                assert len(expected_langs) == 1
                lang_detected = expected_langs[0]
            
            if lang_detected not in expected_langs:
                print(f'Lang detected {lang_detected} of {sample}  not in {expected_langs} skipped')
                continue 
                
            self.count_demographics(sample, lang_detected)
            if max_samples is not None:
                if n_sample_counted>=max_samples:
                    break

        if verbose:
            print(f'{n_sample_counted} sentences were processed')
        
    
    

    def final_result(self) -> tp.Tuple[str, Counter, Counter]:
        return (self.supported_langs, self.count_gender, self.count_np)


    def gender_dist(self):
        final_count = self.final_result()
        lang = final_count[0]
        gender_count = final_count[1]
        
        summary = f"Report for lang {lang}\n\n"
        summary += f"Out of {gender_count['_total']} words: \n" 

        report = {}
        
        for gender_cat in GENDER_LS: 
            summary += f"{gender_cat} words amounts for {gender_count[gender_cat]} ({gender_count[gender_cat]/gender_count['_total']*100:0.1f}%), "
            report[gender_cat] = [gender_count[gender_cat], gender_count[gender_cat]/gender_count['_total']*100]
        report['total'] = [gender_count['_total'], 100]
        report = pd.DataFrame(report)
        
        return report
    

    def printout_summary_demographics(self, printout=True, write_dir: str=None):
        final_count = self.final_result()
        lang = final_count[0]
        gender_count = final_count[1]
        demographics = final_count[2]
        
        summary = f"Report for lang {lang}\n\n"
        
        summary += f"Out of {gender_count['_total']} words: \n" 
        for gender_cat in GENDER_LS:
            summary += f"{gender_cat} words amounts for {gender_count[gender_cat]} ({gender_count[gender_cat]/gender_count['_total']*100:0.1f}%), "
        
        summary+="\n\n"

        summary += f"Out of {demographics['_total']} samples: \n" 
        for demog in demographics:
            if demog != '_total':   
                bucket = demog.split('\t')[1]
                axis = demog.split('\t')[0]
                
                gender = demog.split('\t')[2]
                if axis == '(none)' and bucket == 'null':
                    assert gender in GENDER_LS
                    continue
                count = demographics[demog]
                summary += f"{bucket}-{axis} samples amounts for {count} ({count/demographics['_total']*100:0.1f}%),\n"

        if printout:
            print(summary)
        if write_dir is not None:
            with open(write_dir, 'w') as f:
                f.write(summary)
                print(f'Summary was written {write_dir}')
        
