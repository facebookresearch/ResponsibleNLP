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

import pandas as pd
import fasttext

from holistic_bias.src.sentences import HolisticBiasSentenceGenerator
import stanza 

LANG = {'es': 'spa'}
LANGISO = {'spa': 'es'}

class CountHolisticBias(object):
    """
    This is the core class for computing Holistic Biases distribution. 
    It does the counting based on Holistic_Bias list of nouns and noun phrases, return counters

    """

    def __init__(
        self,
        store_hb_dir: str,
        ft_model_path: str,
        only_gender : bool = False,
        langs: list = ['en'], 
        
    ) -> None:
        
        store_hb_dir = Path(store_hb_dir)
        store_hb_dir.mkdir(exist_ok=True)
        dataset_version = 'v1.1'
        self.only_gender = only_gender
        self.lang_detect_model = fasttext.load_model(ft_model_path)



        self.noun_phrases = {lang: {} for lang in langs}
        self.gender_regs = {lang: {} for lang in langs}
        holistic_bias_data = HolisticBiasSentenceGenerator(store_hb_dir, dataset_version=dataset_version) 
        self.supported_langs = langs
        self.stanza_tokenizer = {}
        for lang in self.supported_langs:
            lang_iso = LANGISO.get(lang, lang)
            stanza.download(LANGISO.get(lang, lang))
            self.stanza_tokenizer[lang] = stanza.Pipeline(lang=lang_iso, processors='tokenize', tokenize_no_ssplit=True)
            # only loading English data right now: make multilingual when data ready
            if not only_gender:
                self.noun_phrases[lang] = holistic_bias_data.get_compiled_noun_phrases(dataset_version=dataset_version, lang=lang)
                # we want to match noun phrases disregarding the undefinite article
                # that's added by default, hack around it (ideally this is an option in HB)
                reg = re.compile(r"^an? ", re.IGNORECASE)
                self.noun_phrases[lang]["noun_phrase_simple"] = self.noun_phrases[lang]["noun_phrase"].apply(lambda s: reg.sub("", s))
                self.noun_phrases[lang]["noun_phrase_re"] = self.noun_phrases[lang].apply(lambda r: re.compile(f"\\b({r['noun_phrase_simple']}|{r['plural_noun_phrase']})\\b",re.IGNORECASE,), axis=1)

            # build a regexp to find gendered noun mentions
            # one regexp per group_gender:
            # 'female' => r"..."
            # 'male' => r"..."
            NOUNS = holistic_bias_data.get_nouns(dataset_version, lang=lang)
            for group_gender, gender_noun_tuples in NOUNS.items():
                r_string = "\\b("
                for noun, plural_noun in gender_noun_tuples:
                    if len(noun) == 0:
                        # singular empty
                        assert len(plural_noun)
                        r_string += f"{re.escape(plural_noun)}|"
                    elif len(plural_noun) == 0: 
                        # plural empty
                        assert len(noun)
                        r_string += f"{re.escape(noun)}|"
                    else:
                        r_string += f"{re.escape(noun)}|{re.escape(plural_noun)}|"
                r_string = r_string[:-1]
                r_string += ")\\b"
                self.gender_regs[lang][group_gender] = re.compile(r_string, re.IGNORECASE)
            
        # Language agnostic counters
        
        self.count_np: Counter = Counter()
        self.count_gender: Counter = Counter()


    def count_demographics(self, line: str, lang: str)-> None:
        sentence = line.strip()
        # lines counter
        assert lang in self.supported_langs, f'{lang} not in {self.supported_langs}'
        if not self.only_gender:
            self.count_np["_total"] += 1
        # for gender, we count words instead of lines, so we do basic tokenization (eng only)
        sentences = self.stanza_tokenizer[lang](sentence).sentences
        assert len(sentences) == 1

        self.count_gender["_total"] += len(sentences[0].tokens) # count words
        if not self.only_gender:
            for _idx, w in self.noun_phrases[lang].iterrows():
                # for each noun_phrases: e.g. (working class man |) (e.g match both plural/singular: bro who is an amputee|bros who are amputees)
                # ==> count if the phrase includes it: if it does: append counter
                if w["noun_phrase_re"].search(sentence):
                    key = (
                        f"{w['bucket']}\t"
                        f"{w['axis']}\t"
                        f"{w['noun_gender']}\t"
                        f"{w['descriptor_preference']}"
                    )
                    # count the occurence for the category of what was matched (bucket, gender, ) : bucket: type of descriptor; value in the bucket ; 
                    self.count_np[key] += 1
        
        for group_gender, reg in self.gender_regs[lang].items():            
            # match to list of nouns that are feminine and masculine 
            self.count_gender[group_gender] += len(reg.findall(sentence))

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
                lang_detected = self.detect_language(text=first_level_val)
                
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
            lang_detected = self.detect_language(text=sample)
            lang_detected = LANG.get(lang_detected, lang_detected)
            
            if lang_detected not in expected_langs:
                print(f'Lang detected {lang_detected} of {sample}  not in {expected_langs} skipped')
                continue 
            self.count_demographics(sample, lang_detected)
            if max_samples is not None:
                if n_sample_counted>=max_samples:
                    break

        if verbose:
            print(f'{n_sample_counted} sententes were counted')

    def final_result(self) -> tp.Tuple[str, Counter, Counter]:
        return (self.supported_langs, self.count_gender, self.count_np)
    
    def gender_dist(self):
        final_count = self.final_result()
        lang = final_count[0]
        gender_count = final_count[1]
        
        summary = f"Report for lang {lang}\n\n"
        summary += f"Out of {gender_count['_total']} words: \n" 

        report = {}
        for gender_cat in ['neutral', 'male', 'female']:
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
        for gender_cat in ['neutral', 'male', 'female']:
            summary += f"{gender_cat} words amounts for {gender_count[gender_cat]} ({gender_count[gender_cat]/gender_count['_total']*100:0.1f}%), "
        summary+="\n\n"

        summary += f"Out of {demographics['_total']} samples: \n" 
        for demog in demographics:
            if demog != '_total':   
                bucket = demog.split('\t')[1]
                axis = demog.split('\t')[0]
                
                gender = demog.split('\t')[2]
                if axis == '(none)' and bucket == 'null':
                    assert gender in ['male', 'female', 'neutral']
                    continue
                count = demographics[demog]
                summary += f"{bucket}-{axis} samples amounts for {count} ({count/demographics['_total']*100:0.1f}%),\n"

        if printout:
            print(summary)
        if write_dir is not None:
            with open(write_dir, 'w') as f:
                f.write(summary)
                print(f'Summary was written {write_dir}')
        
