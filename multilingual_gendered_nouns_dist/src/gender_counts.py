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
import gzip
import stanza 
import nltk
import json 
import numpy as np
from nltk.tokenize import word_tokenize
from pythainlp.tokenize import word_tokenize as word_tokenize_thai

ISO_639_3_TO_1 = {'aar': 'aa', 'abk': 'ab', 'ave': 'ae', 'afr': 'af', 'aka': 'ak', 'amh': 'am', 'arg': 'an', 'arb': 'ar', 'asm': 'as', 'ava': 'av', 'aym': 'ay', 'aze': 'az', 'bak': 'ba', 'bel': 'be', 'bul': 'bg', 'bis': 'bi', 'bam': 'bm', 'ben': 'bn', 'bod': 'bo', 'bre': 'br', 'bos': 'bs', 'cat': 'ca', 'che': 'ce', 'cha': 'ch', 'cos': 'co', 'cre': 'cr', 'ces': 'cs', 'chu': 'cu', 'chv': 'cv', 'cym': 'cy', 'dan': 'da', 'deu': 'de', 'div': 'dv', 'dzo': 'dz', 'ewe': 'ee', 'ell': 'el', 'eng': 'en', 'epo': 'eo', 'spa': 'es', 'est': 'et', 'lvs':'lv', 'eus': 'eu', 'pes': 'fa', 'ful': 'ff', 'fin': 'fi', 'fij': 'fj', 'fao': 'fo', 'fra': 'fr', 'fry': 'fy', 'gle': 'ga', 'gla': 'gd', 'glg': 'gl', 'grn': 'gn', 'guj': 'gu', 'glv': 'gv', 'hau': 'ha', 'heb': 'he', 'hin': 'hi', 'hmo': 'ho', 'hrv': 'hr', 'hat': 'ht', 'hun': 'hu', 'hye': 'hy', 'her': 'hz', 'ina': 'ia', 'ind': 'id', 'ile': 'ie', 'ibo': 'ig', 'iii': 'ii', 'ipk': 'ik', 'ido': 'io', 'isl': 'is', 'ita': 'it', 'iku': 'iu', 'jpn': 'ja', 'jav': 'jv', 'kat': 'ka', 'kon': 'kg', 'kik': 'ki', 'kua': 'kj', 'kaz': 'kk', 'kal': 'kl', 'khm': 'km', 'kan': 'kn', 'kor': 'ko', 'kau': 'kr', 'kas': 'ks', 'kur': 'ku', 'kom': 'kv', 'cor': 'kw', 'kir': 'ky', 'lat': 'la', 'ltz': 'lb', 'lug': 'lg', 'lim': 'li', 'lin': 'ln', 'lao': 'lo', 'lit': 'lt', 'lub': 'lu',  'mlg': 'mg', 'mah': 'mh', 'mri': 'mi', 'mkd': 'mk', 'mal': 'ml', 'khk': 'mn', 'mar': 'mr', 'msa': 'ms', 'mlt': 'mt', 'mya': 'my', 'nau': 'na', 'nob': 'nb', 'nde': 'nd', 'nep': 'ne', 'ndo': 'ng', 'nld': 'nl', 'nno': 'nn', 'nor': 'no', 'nbl': 'nr', 'nav': 'nv', 'nya': 'ny', 'oci': 'oc', 'oji': 'oj', 'orm': 'om', 'ori': 'or', 'oss': 'os', 'pan': 'pa', 'pli': 'pi', 'pol': 'pl', 'pus': 'ps', 'por': 'pt', 'que': 'qu', 'roh': 'rm', 'run': 'rn', 'ron': 'ro', 'rus': 'ru', 'kin': 'rw', 'san': 'sa', 'srd': 'sc', 'snd': 'sd', 'sme': 'se', 'sag': 'sg', 'hbs': 'sh', 'sin': 'si', 'slk': 'sk', 'slv': 'sl', 'smo': 'sm', 'sna': 'sn', 'som': 'so', 'sqi': 'sq', 'srp': 'sr', 'ssw': 'ss', 'sot': 'st', 'sun': 'su', 'swe': 'sv', 'swh': 'sw', 'tam': 'ta', 'tel': 'te', 'tgk': 'tg', 'tha': 'th', 'tir': 'ti', 'tuk': 'tk', 'tgl': 'tl', 'tsn': 'tn', 'ton': 'to', 'tur': 'tr', 'tso': 'ts', 'tat': 'tt', 'twi': 'tw', 'tah': 'ty', 'uig': 'ug', 'ukr': 'uk', 'urd': 'ur', 'uzb': 'uz', 'ven': 've', 'vie': 'vi', 'vol': 'vo', 'wln': 'wa', 'wol': 'wo', 'xho': 'xh', 'yid': 'yi', 'yor': 'yo', 'zha': 'za', 'zho': 'zh', 'cmn': 'zh', 'zul': 'zu'}
ISO_639_1_TO_3 = {iso1:iso3 for iso3, iso1 in ISO_639_3_TO_1.items()}

LANG2Script = {'arb': 'Arab', 'asm': 'Beng', 'bel': 'Cyrl', 'ben': 'Beng', 'bul': 'Cyrl', 'cat': 'Latn', 'ces': 'Latn', 'ckb': 'Arab', 'cmn': 'Hans', 'cym': 'Latn', 'dan': 'Latn', 'deu': 'Latn', 'ell': 'Grek', 'eng': 'Latn', 'est': 'Latn', 'fin': 'Latn', 'fra': 'Latn', 'gle': 'Latn', 'hin': 'Deva', 'hun': 'Latn', 'ind': 'Latn', 'ita': 'Latn', 'jpn': 'Jpan', 'kat': 'Geor', 'khk': 'Cyrl', 'kir': 'Cyrl', 'lit': 'Latn', 'lug': 'Latn', 'lvs': 'Latn', 'mar': 'Deva', 'mlt': 'Latn', 'nld': 'Latn', 'pan': 'Guru', 'pes': 'Arab', 'pol': 'Latn', 'por': 'Latn', 'ron': 'Latn', 'rus': 'Cyrl', 'slk': 'Latn', 'slv': 'Latn', 'spa': 'Latn', 'swe': 'Latn', 'swh': 'Latn', 'tam': 'Taml', 'tha': 'Thai', 'tur': 'Latn', 'ukr': 'Cyrl', 'urd': 'Arab', 'uzn': 'Latn', 'vie': 'Latn', 'yue': 'Hant', 'kan': 'Knda', 'tel': 'Telu', 'tgl': 'Latn', 'zul': 'Latn'}

GENDERS = ['feminine', 'masculine',  'unspecified']


def count_lines(file_dir: str):
    
    if str(file_dir).endswith('.txt.gz'):
        try:
            return sum(1 for _ in gzip.open(file_dir,'rt'))
        except EOFError as e:
            print(e, file_dir)
    else:
        return sum(1 for _ in open(file_dir))


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
            if lang == 'ar':
                word_tokenizer = stanza.Pipeline(lang=lang, processors='tokenize,mwt', tokenize_no_split=True)
                suffix = '-mwt'
            else:
                word_tokenizer = stanza.Pipeline(lang=lang, processors='tokenize', tokenize_no_split=True)
                suffix = ''
            tokenizer_name = f'stanza{suffix}'
            print('Stanza tokenizer loaded:', tokenizer_name)
            return word_tokenizer, tokenizer_name
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
            self.lang_detect_model = None
        
        self.noun_phrases = {lang: {} for lang in langs}
        self.gender_ls =  {lang: {} for lang in langs}
        self.gender_counters =  {lang: {} for lang in langs}
        base_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dataset")
        dataset_folder = os.path.join(base_folder, dataset_version)
        
        self.stat = {gender: None for gender in GENDERS+['total', 'coverage', 'ste_diff_fem_masc']}
        self.gender_dic_vec = None
        self.langs = langs
        self.tokenizer = {}
        self.tokenizer_type = {}
        self.nouns = {}
        script_path = Path(__file__).resolve().parent
        base_folder = script_path/'..'/'dataset'/dataset_version
        SUPPORTED_LANGS = [file.stem.split('_')[0] for file in base_folder.iterdir() if file.is_file() if file.suffix == '.json']


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
        self.n_words_per_match = []
        self.n_words_with_no_match = 0
        self.n_doc_w_match = []


    def count_demographics(self, line: str, lang: str, return_terms=False, return_vec=False)-> None:
        sentence = line.strip().lower()
        if len(sentence) == 0:
            print(f'Warning: empty sentence: {lang}')
            return
        gender_vec = None
        # lines counter
        assert lang in self.langs, f'{lang} not in {self.langs}'

        # for gender, we count words instead of lines, so we do basic tokenization (eng only)
        if  self.tokenizer_type[lang].startswith('stanza'):
            sentences = self.tokenizer[lang](sentence).sentences
            if len(sentences) == 0:
                print(f'Warning: empty sentence: {lang}')
                return 
            tokenized_sentence = sentences[0]
            
            if self.tokenizer_type[lang].endswith('mwt'):
                tokenized_sentence = [word.text for token in tokenized_sentence.tokens for word in token.words]
            else:
                tokenized_sentence = [token.text for token in tokenized_sentence.tokens]
            # verify that segmentation lead to a single sentence
            
        elif self.tokenizer_type[lang] in ['nltk', 'pythainlp']:
            tokenized_sentence = self.tokenizer[lang](sentence)
        
        
        self.count_gender["_total"] += len(tokenized_sentence) # count words
        self.n_words_with_no_match += len(tokenized_sentence) 
        curr_dic = {}
        curr_dic_terms = {} if return_terms else None
        matching = 0
        if return_vec:
            masc_vec = [int(word in self.gender_counters[lang]['masculine']) for word in tokenized_sentence]
            fem_vec = [int(word in self.gender_counters[lang]['feminine']) for word in tokenized_sentence]
            gender_vec = {'masculine': masc_vec, 'feminine': fem_vec}

        for group_gender in self.gender_counters[lang].keys():            
            # match to list of nouns that are feminine and masculine 
            total_match = 0
            tokenized_sentence_counts = Counter(tokenized_sentence)
            common_elements = set(tokenized_sentence_counts) & set(self.gender_counters[lang][group_gender])
                
            for element in common_elements: 
                total_match += tokenized_sentence_counts[element]
            if total_match > 0:
                matching = 1
            self.count_gender[group_gender] += total_match
            curr_dic[group_gender] = total_match    
            if return_terms:
                curr_dic_terms[group_gender] =  ', '.join(common_elements)
        self.n_words_per_match.append(self.n_words_with_no_match-len(tokenized_sentence))
        self.n_words_with_no_match = 0
        self.n_doc_w_match.append(matching)
        return curr_dic, curr_dic_terms, gender_vec
            

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
                
                lang_detected = ISO_639_3_TO_1.get(lang_detected, lang_detected)                
                self.count_demographics(first_level_val, lang_detected)
            if max_samples is not None:
                if n_sample_counted>=max_samples:
                    break

        if verbose:
            print(f'{n_sample_counted} samples were counted')
    
    def count_lines(self, file_dir: str):
        
        if str(file_dir).endswith('.txt.gz'):
            return sum(1 for _ in gzip.open(file_dir,'rt'))
        else:
            return sum(1 for _ in open(file_dir))

    def process_txt_file(self, file_dir: str,
                        clean_sample: tp.Callable=None, 
                        max_samples: int=None,
                        expected_langs: list = None,
                        return_vec: bool= False,
                        verbose: int =1) -> None:
        # iterate over lines
        n_sample_counted = 0
        if max_samples is not None:
            max_samples = int(max_samples)
        
        n_samples = count_lines(file_dir)
        if verbose:
            print(f'{n_samples} lines in {file_dir.name}')
        if str(file_dir).endswith('.txt.gz'):
            file = gzip.open(file_dir, 'rt')
        else:
            file = open(file_dir)
        
        gender_dic = {'masculine': [], 'feminine': []}
        for _, sample in tqdm(enumerate(file), desc=f'Processing {file.name}', total=n_samples):
            n_sample_counted+=1
            sample = clean_sample(sample)
            
            if self.lang_detect_model:
                lang_detected = self.detect_language(text=sample)
                lang_detected = ISO_639_3_TO_1.get(lang_detected, lang_detected)
            else:
                assert len(expected_langs) == 1
                lang_detected = expected_langs[0]
            
            if lang_detected not in expected_langs:
                print(f'Lang detected {lang_detected} of {sample}  not in {expected_langs} skipped')
                continue 
                
            _, _, _gender_dic = self.count_demographics(sample, lang_detected, return_vec=return_vec)
            if return_vec:
                gender_dic['masculine'].extend(_gender_dic['masculine'])
                gender_dic['feminine'].extend(_gender_dic['feminine'])
            if max_samples is not None:
                if n_sample_counted>=max_samples:
                    break

        if verbose:
            print(f'{n_sample_counted} sentences were processed')
        if return_vec:
            gender_dic['masculine'] = np.array(gender_dic['masculine'])
            gender_dic['feminine'] = np.array(gender_dic['feminine'])
        self.gender_dic_vec = gender_dic
    

    def final_result(self) -> tp.Tuple[str, Counter, Counter]:
        return (self.langs, self.count_gender, self.count_np)


    def gender_dist(self, info_file: str):
        final_count = self.final_result()
        lang = final_count[0]
        gender_count = final_count[1]
        
        
        report = {}
        
        for gender_cat in GENDERS: 
            report[gender_cat] = [gender_count[gender_cat], gender_count[gender_cat]/gender_count['_total']*100]
            if gender_count[gender_cat]==0:
                print(f'WARNING {lang} {gender_cat}: 0 match file {info_file}')
        report['total'] = [gender_count['_total'], 100]
        report = pd.DataFrame(report)
        
        # fill in self.stat for reporting
        assert self.stat['masculine'] is None
        self.stat['masculine'] = report['masculine'][1]
        self.stat['feminine'] = report['feminine'][1]
        self.stat['unspecified'] = report['unspecified'][1]
        self.stat['total'] = report['total'][0]
        
        
        # compute std of the difference masculine vs. feminine
        assert len(report['masculine']) == len(report['feminine'])
        size = np.size(self.gender_dic_vec['masculine'])
        self.stat['ste_diff_fem_masc'] = np.std(self.gender_dic_vec['masculine']-self.gender_dic_vec['feminine'], ddof=1)/np.sqrt(size)
        # compute coverage statistics
        self.n_doc_w_match = pd.Series(self.n_doc_w_match)
        coverage = len(self.n_doc_w_match[self.n_doc_w_match  == 1])/len(self.n_doc_w_match)*100

        #try:
        #    assert len(coverage.value_counts())>1, f'Lang {lang} No match found: {coverage}'
        #except Exception as e:
        #    print(f'Warning {e}')
        
        #coverage_stat = coverage.value_counts()[1]/len(coverage)*100 if len(coverage.value_counts())>1 else 0
        
        self.stat['coverage'] = coverage
    

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
        
