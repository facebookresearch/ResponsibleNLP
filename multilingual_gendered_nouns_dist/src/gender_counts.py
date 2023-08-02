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
import nltk
import json 

from nltk.tokenize import word_tokenize
from pythainlp.tokenize import word_tokenize as word_tokenize_thai
ISO_639_3_TO_1 = {'aar': 'aa', 'abk': 'ab', 'ave': 'ae', 'afr': 'af', 'aka': 'ak', 'amh': 'am', 'arg': 'an', 'arb': 'ar', 'asm': 'as', 'ava': 'av', 'aym': 'ay', 'aze': 'az', 'bak': 'ba', 'bel': 'be', 'bul': 'bg', 'bis': 'bi', 'bam': 'bm', 'ben': 'bn', 'bod': 'bo', 'bre': 'br', 'bos': 'bs', 'cat': 'ca', 'che': 'ce', 'cha': 'ch', 'cos': 'co', 'cre': 'cr', 'ces': 'cs', 'chu': 'cu', 'chv': 'cv', 'cym': 'cy', 'dan': 'da', 'deu': 'de', 'div': 'dv', 'dzo': 'dz', 'ewe': 'ee', 'ell': 'el', 'eng': 'en', 'epo': 'eo', 'spa': 'es', 'est': 'et', 'eus': 'eu', 'fas': 'fa', 'ful': 'ff', 'fin': 'fi', 'fij': 'fj', 'fao': 'fo', 'fra': 'fr', 'fry': 'fy', 'gle': 'ga', 'gla': 'gd', 'glg': 'gl', 'grn': 'gn', 'guj': 'gu', 'glv': 'gv', 'hau': 'ha', 'heb': 'he', 'hin': 'hi', 'hmo': 'ho', 'hrv': 'hr', 'hat': 'ht', 'hun': 'hu', 'hye': 'hy', 'her': 'hz', 'ina': 'ia', 'ind': 'id', 'ile': 'ie', 'ibo': 'ig', 'iii': 'ii', 'ipk': 'ik', 'ido': 'io', 'isl': 'is', 'ita': 'it', 'iku': 'iu', 'jpn': 'ja', 'jav': 'jv', 'kat': 'ka', 'kon': 'kg', 'kik': 'ki', 'kua': 'kj', 'kaz': 'kk', 'kal': 'kl', 'khm': 'km', 'kan': 'kn', 'kor': 'ko', 'kau': 'kr', 'kas': 'ks', 'kur': 'ku', 'kom': 'kv', 'cor': 'kw', 'kir': 'ky', 'lat': 'la', 'ltz': 'lb', 'lug': 'lg', 'lim': 'li', 'lin': 'ln', 'lao': 'lo', 'lit': 'lt', 'lub': 'lu', 'lav': 'lv', 'mlg': 'mg', 'mah': 'mh', 'mri': 'mi', 'mkd': 'mk', 'mal': 'ml', 'mon': 'mn', 'mar': 'mr', 'msa': 'ms', 'mlt': 'mt', 'mya': 'my', 'nau': 'na', 'nob': 'nb', 'nde': 'nd', 'nep': 'ne', 'ndo': 'ng', 'nld': 'nl', 'nno': 'nn', 'nor': 'no', 'nbl': 'nr', 'nav': 'nv', 'nya': 'ny', 'oci': 'oc', 'oji': 'oj', 'orm': 'om', 'ori': 'or', 'oss': 'os', 'pan': 'pa', 'pli': 'pi', 'pol': 'pl', 'pus': 'ps', 'por': 'pt', 'que': 'qu', 'roh': 'rm', 'run': 'rn', 'ron': 'ro', 'rus': 'ru', 'kin': 'rw', 'san': 'sa', 'srd': 'sc', 'snd': 'sd', 'sme': 'se', 'sag': 'sg', 'hbs': 'sh', 'sin': 'si', 'slk': 'sk', 'slv': 'sl', 'smo': 'sm', 'sna': 'sn', 'som': 'so', 'sqi': 'sq', 'srp': 'sr', 'ssw': 'ss', 'sot': 'st', 'sun': 'su', 'swe': 'sv', 'swa': 'sw', 'tam': 'ta', 'tel': 'te', 'tgk': 'tg', 'tha': 'th', 'tir': 'ti', 'tuk': 'tk', 'tgl': 'tl', 'tsn': 'tn', 'ton': 'to', 'tur': 'tr', 'tso': 'ts', 'tat': 'tt', 'twi': 'tw', 'tah': 'ty', 'uig': 'ug', 'ukr': 'uk', 'urd': 'ur', 'uzb': 'uz', 'ven': 've', 'vie': 'vi', 'vol': 'vo', 'wln': 'wa', 'wol': 'wo', 'xho': 'xh', 'yid': 'yi', 'yor': 'yo', 'zha': 'za', 'zho': 'zh', 'cmn': 'zh', 'zul': 'zu'}

SUPPORTED_LANGS = ['eng', 'arb', 'asm', 'bel', 'ben', 'bul', 'cat', 'ces', 'ckb', 'cmn', 'cym', 'dan', 'deu', 'ell', 'est', 'fin', 'fra', 'gle', 'hin', 'hun', 'ind', 'ita', 'jpn', 'kan', 'kat', 'khk', 'kir', 'kor', 'lit', 'lug', 'lvs', 'mar', 'mlt', 'nld', 'nou', 'pan', 'pes', 'pol', 'por', 'ron', 'rus', 'slk', 'slv', 'spa', 'swe', 'swh', 'tam', 'tel', 'tgl', 'tha', 'tur', 'urd', 'uzn', 'vie', 'yue', 'zul']

GENDERS = ['feminine', 'masculine',  'unspecified']


def count_lines(filename):
    with open(filename, 'r') as file:
        return sum(1 for line in file)


def load_tokenizer(lang: str):
    if lang == 'th':
        print('Tokenizer: for thai using pythainlp')
        return word_tokenize_thai, 'pythainlp'
    else:
        try:    
            if lang == 'yue':
                print('Tokenizer: for yue using zh stanza tokenization')
                lang = 'zh'
            stanza.download(lang)
            word_tokenizer = stanza.Pipeline(lang=lang, processors='tokenize', tokenize_no_ssplit=True)
            return word_tokenizer, 'stanza'
        except Exception as e:
            print(f'Stanza model not available {e}: using punctuation tokenizer from nltk')
            print(f'Tokenizer: fall back on nltk {lang}')
            return word_tokenize, 'nltk'
    



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
        self.gender_ls =  {lang: {} for lang in langs}
        self.gender_counters =  {lang: {} for lang in langs}
        base_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dataset")
        dataset_folder = os.path.join(base_folder, dataset_version)
        
        
        self.langs = langs
        self.tokenizer = {}
        self.tokenizer_type = {}
        self.nouns = {}

        if langs:
            for lang in langs:
                assert lang in SUPPORTED_LANGS, f'{lang} not supported by the pipeline'
        # loading nltk as fall back option if stanza not supported
        nltk.download('punkt')
        for lang in self.langs:
            if lang not in ISO_639_3_TO_1:
                print(f'WARNING lang code {lang}')
            lang_iso = ISO_639_3_TO_1.get(lang, lang)
            
            self.tokenizer[lang], self.tokenizer_type[lang] = load_tokenizer(lang_iso)

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
        sentence = line.strip().lower()
        # lines counter
        assert lang in self.langs, f'{lang} not in {self.langs}'

        # for gender, we count words instead of lines, so we do basic tokenization (eng only)
        if  self.tokenizer_type[lang] == 'stanza':
            sentences = self.tokenizer[lang](sentence).sentences
            if len(sentences) == 0:
                print(f'Warning: empty sentence: {lang}')
                return 
            tokenized_sentence = sentences[0]
            tokenized_sentence = [token.text for token in tokenized_sentence.tokens]
            # verify that segmentation lead to a single sentence
            
        elif self.tokenizer_type[lang] in ['nltk', 'pythainlp']:
            tokenized_sentence = self.tokenizer[lang](sentence)
        
        assert len(sentence)>0, 'empty sentence'
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
        return (self.langs, self.count_gender, self.count_np)


    def gender_dist(self, info_file: str):
        final_count = self.final_result()
        lang = final_count[0]
        gender_count = final_count[1]
        
        summary = f"Report for lang {lang}\n\n"
        summary += f"Out of {gender_count['_total']} words: \n" 

        report = {}
        
        for gender_cat in GENDERS: 
            summary += f"{gender_cat} words amounts for {gender_count[gender_cat]} ({gender_count[gender_cat]/gender_count['_total']*100:0.1f}%), "
            report[gender_cat] = [gender_count[gender_cat], gender_count[gender_cat]/gender_count['_total']*100]
            if gender_count[gender_cat]==0:
                print(f'WARNING {lang} {gender_cat}: 0 match file {info_file}')
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
        for gender_cat in GENDERS:
            summary += f"{gender_cat} words amounts for {gender_count[gender_cat]} ({gender_count[gender_cat]/gender_count['_total']*100:0.1f}%), "
        
        summary+="\n\n"

        summary += f"Out of {demographics['_total']} samples: \n" 
        for demog in demographics:
            if demog != '_total':   
                bucket = demog.split('\t')[1]
                axis = demog.split('\t')[0]
                
                gender = demog.split('\t')[2]
                if axis == '(none)' and bucket == 'null':
                    assert gender in GENDERS
                    continue
                count = demographics[demog]
                summary += f"{bucket}-{axis} samples amounts for {count} ({count/demographics['_total']*100:0.1f}%),\n"

        if printout:
            print(summary)
        if write_dir is not None:
            with open(write_dir, 'w') as f:
                f.write(summary)
                print(f'Summary was written {write_dir}')
        
