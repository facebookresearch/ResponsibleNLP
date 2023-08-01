#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json


DEFAULT_DATASET_VERSION = "v1.0"
# Default to the original v1.0 version for compatibility.

RANDOM_SEED = 17

# Template building constants
NONE_STRING = "(none)"  # Use when an attribute is not present
NO_PREFERENCE_DATA_STRING = "no_data"
# Use when it is not known whether a descriptor is preferred

ISO_639_3_TO_1 = {'aar': 'aa', 'abk': 'ab', 'ave': 'ae', 'afr': 'af', 'aka': 'ak', 'amh': 'am', 'arg': 'an', 'ara': 'ar', 'asm': 'as', 'ava': 'av', 'aym': 'ay', 'aze': 'az', 'bak': 'ba', 'bel': 'be', 'bul': 'bg', 'bis': 'bi', 'bam': 'bm', 'ben': 'bn', 'bod': 'bo', 'bre': 'br', 'bos': 'bs', 'cat': 'ca', 'che': 'ce', 'cha': 'ch', 'cos': 'co', 'cre': 'cr', 'ces': 'cs', 'chu': 'cu', 'chv': 'cv', 'cym': 'cy', 'dan': 'da', 'deu': 'de', 'div': 'dv', 'dzo': 'dz', 'ewe': 'ee', 'ell': 'el', 'eng': 'en', 'epo': 'eo', 'spa': 'es', 'est': 'et', 'eus': 'eu', 'fas': 'fa', 'ful': 'ff', 'fin': 'fi', 'fij': 'fj', 'fao': 'fo', 'fra': 'fr', 'fry': 'fy', 'gle': 'ga', 'gla': 'gd', 'glg': 'gl', 'grn': 'gn', 'guj': 'gu', 'glv': 'gv', 'hau': 'ha', 'heb': 'he', 'hin': 'hi', 'hmo': 'ho', 'hrv': 'hr', 'hat': 'ht', 'hun': 'hu', 'hye': 'hy', 'her': 'hz', 'ina': 'ia', 'ind': 'id', 'ile': 'ie', 'ibo': 'ig', 'iii': 'ii', 'ipk': 'ik', 'ido': 'io', 'isl': 'is', 'ita': 'it', 'iku': 'iu', 'jpn': 'ja', 'jav': 'jv', 'kat': 'ka', 'kon': 'kg', 'kik': 'ki', 'kua': 'kj', 'kaz': 'kk', 'kal': 'kl', 'khm': 'km', 'kan': 'kn', 'kor': 'ko', 'kau': 'kr', 'kas': 'ks', 'kur': 'ku', 'kom': 'kv', 'cor': 'kw', 'kir': 'ky', 'lat': 'la', 'ltz': 'lb', 'lug': 'lg', 'lim': 'li', 'lin': 'ln', 'lao': 'lo', 'lit': 'lt', 'lub': 'lu', 'lav': 'lv', 'mlg': 'mg', 'mah': 'mh', 'mri': 'mi', 'mkd': 'mk', 'mal': 'ml', 'mon': 'mn', 'mar': 'mr', 'msa': 'ms', 'mlt': 'mt', 'mya': 'my', 'nau': 'na', 'nob': 'nb', 'nde': 'nd', 'nep': 'ne', 'ndo': 'ng', 'nld': 'nl', 'nno': 'nn', 'nor': 'no', 'nbl': 'nr', 'nav': 'nv', 'nya': 'ny', 'oci': 'oc', 'oji': 'oj', 'orm': 'om', 'ori': 'or', 'oss': 'os', 'pan': 'pa', 'pli': 'pi', 'pol': 'pl', 'pus': 'ps', 'por': 'pt', 'que': 'qu', 'roh': 'rm', 'run': 'rn', 'ron': 'ro', 'rus': 'ru', 'kin': 'rw', 'san': 'sa', 'srd': 'sc', 'snd': 'sd', 'sme': 'se', 'sag': 'sg', 'hbs': 'sh', 'sin': 'si', 'slk': 'sk', 'slv': 'sl', 'smo': 'sm', 'sna': 'sn', 'som': 'so', 'sqi': 'sq', 'srp': 'sr', 'ssw': 'ss', 'sot': 'st', 'sun': 'su', 'swe': 'sv', 'swa': 'sw', 'tam': 'ta', 'tel': 'te', 'tgk': 'tg', 'tha': 'th', 'tir': 'ti', 'tuk': 'tk', 'tgl': 'tl', 'tsn': 'tn', 'ton': 'to', 'tur': 'tr', 'tso': 'ts', 'tat': 'tt', 'twi': 'tw', 'tah': 'ty', 'uig': 'ug', 'ukr': 'uk', 'urd': 'ur', 'uzb': 'uz', 'ven': 've', 'vie': 'vi', 'vol': 'vo', 'wln': 'wa', 'wol': 'wo', 'xho': 'xh', 'yid': 'yi', 'yor': 'yo', 'zha': 'za', 'zho': 'zh', 'zul': 'zu'}

def clean_sample(text: str):
    text = text.replace('\n', ' ')
    return text.strip().lower()


def rename_gender(json_dir): 
    print('read', json_dir)
    with open(json_dir, 'r') as read:
        print('reading', json_dir)
        file = json.load(read)
    if 'feminine' not in file:
        file['feminine'] = file['female']
        del file['female']
    if 'masculine' not in file:
        file['masculine'] = file['male']
        del file['male']
    
    
    
    assert len(file) == 3, file
    assert {'feminine', 'masculine', 'unspecified'} == set(file.keys()), file
    with open(json_dir, 'w') as write:
        json.dump(file, write, indent=2, ensure_ascii=False)
    print(f'{write} done')

#LANGISO = {'arb', 'asm', 'bel', 'ben', 'bul', 'cat', 'ces', 'ckb', 'cmn', 'cym', 'dan', 'deu', 'ell', 'eng', 'est', 'fin', 'fra', 'gle', 'hin', 'hun', 'ind', 'ita', 'jpn', 'kan', 'kat', 'khk', 'kir', 'kor', 'lit', 'lug', 'mar', 'mlt', 'nld', 'pan', 'pes', 'pol', 'por', 'ron', 'rus', 'slk', 'slv', 'spa', 'swe', 'swh', 'tam', 'tel', 'tgl', 'tha', 'tur', 'urd', 'uzn', 'vie', 'yue', 'zul'}

#for lang in LANGISO:
#    rename_gender(f'/private/home/benjaminmuller/dev/biases/ResponsibleNLP/multilingual_gendered_nouns_dist/dataset/v1.0/{lang}_nouns.json')
