#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

DEFAULT_DATASET_VERSION = "v1.0"
# Default to the original v1.0 version for compatibility.

RANDOM_SEED = 17

# Template building constants
NONE_STRING = "(none)"  # Use when an attribute is not present
NO_PREFERENCE_DATA_STRING = "no_data"
# Use when it is not known whether a descriptor is preferred
import json

def clean_sample(text: str):
    text = text.replace('\n', ' ')
    return text.strip().lower()


def rename_gender(json_dir): 
    print('read', json_dir)
    with open(json_dir, 'r') as read:
        file = json.load(read)
    file['feminine'] = file['female']
    file['masculine'] = file['male']
    del file['male']
    del file['female']
    with open(json_dir, 'w') as write:
        json.dump(file, write, indent=2, ensure_ascii=False)
    print(f'{write} done')

LANGISO = {#'ben': 'bn', 'cym': 'cy', 'hun': 'hu', 'lit': 'lt', 'pes': 'fa', 'tam': 'ta', 'urd': 'ur', 'bul': 'bg', 'deu': 'de', 'ind': 'id', 'lug': 'lg', 'por': 'pt', 'tel': 'te', 'vie': 'vi', 'cat': 'ca', 'est': 'et', 'ita': 'it', 'mar': 'mr', 'slv': 'sl', 'tgl': 'tl', 'zul': 'zu', 'ckb': 'ckb', 'fra': 'fr', 
           'kan': 'kn', 'mlt': 'mt', 'spa': 'es', 'tha': 'th', 'cmn': 'zh', 'hin': 'hi', 'kat': 'ka', 'pan': 'pa', 'swh': 'sw', 'tur': 'tr'}

#for lang in LANGISO:
#    rename_gender(f'/private/home/benjaminmuller/dev/biases/ResponsibleNLP/multilingual_gendered_nouns_dist/dataset/v1.0/{lang}_nouns.json')
