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

LANGISO = {'arb', 'asm', 'bel', 'ben', 'bul', 'cat', 'ces', 'ckb', 'cmn', 'cym', 'dan', 'deu', 'ell', 'eng', 'est', 'fin', 'fra', 'gle', 'hin', 'hun', 'ind', 'ita', 'jpn', 'kan', 'kat', 'khk', 'kir', 'kor', 'lit', 'lug', 'mar', 'mlt', 'nld', 'pan', 'pes', 'pol', 'por', 'ron', 'rus', 'slk', 'slv', 'spa', 'swe', 'swh', 'tam', 'tel', 'tgl', 'tha', 'tur', 'urd', 'uzn', 'vie', 'yue', 'zul'}

for lang in LANGISO:
    rename_gender(f'/private/home/benjaminmuller/dev/biases/ResponsibleNLP/multilingual_gendered_nouns_dist/dataset/v1.0/{lang}_nouns.json')
