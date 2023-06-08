
import re
import sys
import typing as tp

from collections import Counter
from pathlib import Path
from tqdm import tqdm

import fasttext

from holistic_bias.src.sentences import HolisticBiasSentenceGenerator



class CountHolisticBias(object):
    """
    This is the core class for computing Holistic Biases distribution. 
    It does the counting based on Holistic_Bias list of nouns and noun phrases, return counters

    Based on https://github.com/fairinternal/seamless_common/blob/hb_count/stopes/eval/holistic_bias/callback.py
    """

    def __init__(
        self,
        store_hb_dir: str,
        ft_model_path: str,
        langs: list = ['en'], 
    ) -> None:
        assert len(langs) == 1 , f'' 
        self.langs = langs 
        store_hb_dir = Path(store_hb_dir)
        store_hb_dir.mkdir(exist_ok=True)
        dataset_version = 'v1.1'

        self.lang_detect_model = fasttext.load_model(ft_model_path)

        self.noun_phrases = {lang: {} for lang in langs}
        self.gender_regs = {lang: {} for lang in langs}
        for lang in langs:
            # only loading English data right now: make multilingual when data ready
            holistic_bias_data = HolisticBiasSentenceGenerator(store_hb_dir, dataset_version=dataset_version) 
            self.noun_phrases[lang] = holistic_bias_data.get_compiled_noun_phrases(dataset_version=dataset_version)
            # we want to match noun phrases disregarding the undefinite article
            # that's added by default, hack around it (ideally this is an option in HB)
            reg = re.compile(r"^an? ", re.IGNORECASE)
            self.noun_phrases[lang]["noun_phrase_simple"] = self.noun_phrases[lang]["noun_phrase"].apply(lambda s: reg.sub("", s))
            self.noun_phrases[lang]["noun_phrase_re"] = self.noun_phrases[lang].apply(lambda r: re.compile(f"\\b({r['noun_phrase_simple']}|{r['plural_noun_phrase']})\\b",re.IGNORECASE,), axis=1)
            # build a regexp to find gendered noun mentions
            # one regexp per group_gender:
            # 'female' => r"..."
            # 'male' => r"..."
            NOUNS = holistic_bias_data.get_nouns(dataset_version)
            for group_gender, gender_noun_tuples in NOUNS.items():
                r_string = "\\b("
                for noun, plural_noun in gender_noun_tuples:
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
        self.count_np["_total"] += 1
        # for gender, we count words instead of lines, so we do basic tokenization (eng only)
        self.count_gender["_total"] += len(sentence.split()) # count words
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
                assert lang_detected == 'en'
                self.count_demographics(first_level_val, lang_detected)
            if max_samples is not None:
                if n_sample_counted>=max_samples:
                    break

        if verbose:
            print(f'{n_sample_counted} samples were counted')

    def final_result(self) -> tp.Tuple[str, Counter, Counter]:
        return (self.langs, self.count_gender, self.count_np)
    
    def printout_summary(self, printout=True, write_dir: str=None):
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
                count = demographics[demog]
                summary += f"{bucket}-{axis} samples amounts for {count} ({count/demographics['_total']*100:0.1f}%),\n"

        if printout:
            print(summary)
        if write_dir is not None:
            with open(write_dir, 'w') as f:
                f.write(summary)
                print(f'Summary was written {write_dir}')
        
