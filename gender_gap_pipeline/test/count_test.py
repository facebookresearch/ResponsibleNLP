#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import pathlib

import sys
sys.path.append('.')

from gender_gap_pipeline.src.gender_counts import MultilingualGenderDistribution
from gender_gap_pipeline.src.util import clean_sample


def count_test_eng():
    LANG = 'eng'
    
    DATA = ["I am a boy!", #1
            'ABU Abu aBU', # testing case:  3 female and 3 male
            ] 
    
    # data includes: 
    ## 36 words with white-space tokenization
    ## 41 with stanza

    file_dir = pathlib.Path('./tmp')
    file_dir.mkdir(exist_ok=True)
    
    # writing test data
    with open(file_dir/'test.txt', 'w') as f:
        for line in DATA:
            f.write(line+'\n')
    
    hb_counter = MultilingualGenderDistribution(store_hb_dir='./tmp', langs=[LANG], ft_model_path=None, dataset_version='v1.0')
    with open(file_dir/'test.txt') as file:
        hb_counter.process_txt(file=file, clean_sample=clean_sample, max_samples=None, expected_langs=[LANG])
    report = hb_counter.gender_dist()
    
    print(report['total'])
    assert report['total'][0] == 8, report
    print(f"TEST:Total number of words is correct {report['total'][0]}")
    assert report['feminine'][0] == 0, report['feminine']
    print(f"TEST:Total number of feminine words is correct {report['feminine'][0]}")
    assert report['masculine'][0] == 1
    print(f"TEST:Total number of masculine words is correct {report['masculine'][0]}")
    assert report['unspecified'][0] == 0, report['unspecified'][0]
    print(f"TEST:Total number of unspecified words is correct {report['unspecified'][0]}")

    print('All test counting done (v1.0 nouns list eng)')




def count_test_spa():
    LANG = 'spa'
    
    DATA = ['Este desconcierto surge de los planes de cambiar el nombre de la Asamblea a Parlamento de Gales.', # 0 
            'empty sentence ',  # 0 
            'esposa esposa esposas et tu che va', # 3 female
            'abuelos abuelos abuelo', # abuelos is both unspecified and make: 3 male, 2 unspecified
            'esposa . esposa! ?esposa esposa', # testing tokenizer: 4 female 
            'ABU Abu aBU', # testing case:  3 female and 3 male
            ] 
    
    # data includes: 
    ## 36 words with white-space tokenization
    ## 41 with stanza

    file_dir = pathlib.Path('./tmp')
    file_dir.mkdir(exist_ok=True)
    
    # writing test data
    with open(file_dir/'test.txt', 'w') as f:
        for line in DATA:
            f.write(line+'\n')
    
    hb_counter = MultilingualGenderDistribution(store_hb_dir='./tmp', langs=[LANG], ft_model_path=None, dataset_version='v1.0')
    with open(file_dir/'test.txt') as file:
        hb_counter.process_txt(file=file, clean_sample=clean_sample, max_samples=None, expected_langs=[LANG])
    report = hb_counter.gender_dist()
    
    print(report['total'])
    assert report['total'][0] == 40
    print(f"TEST:Total number of words is correct {report['total'][0]}")
    assert report['feminine'][0] == 7+3, report['feminine']
    print(f"TEST:Total number of feminine words is correct {report['feminine'][0]}")
    assert report['masculine'][0] == 3+3
    print(f"TEST:Total number of masculine words is correct {report['masculine'][0]}")
    assert report['unspecified'][0] == 2, report['unspecified'][0]
    print(f"TEST:Total number of unspecified words is correct {report['unspecified'][0]}")

    print('All test counting done (v1.0 nouns list spa)')





def count_test_eng_2():
    LANG = 'eng'
    
    DATA = ['my dad is a great guy', # 2
            'no no yes !  ',  # 0 
            'esposa esposa esposas et tu che va', # 0
            'parents parent ', # 2 neutral
            'girls ',
            
            ] 
    
    # data includes: 
    ## 36 words with white-space tokenization
    ## 41 with stanza

    file_dir = pathlib.Path('./tmp')
    file_dir.mkdir(exist_ok=True)
    
    # writing test data
    with open(file_dir/'test.txt', 'w') as f:
        for line in DATA:
            f.write(line+'\n')
    
    hb_counter = MultilingualGenderDistribution(store_hb_dir='./tmp', langs=[LANG], ft_model_path=None, dataset_version='v1.0')
    with open(file_dir/'test.txt') as file:
        hb_counter.process_txt(file=file, clean_sample=clean_sample, max_samples=None, expected_langs=[LANG])
    report = hb_counter.gender_dist()
    
    print(report['total'])
    assert report['total'][0] == 20
    print(f"TEST:Total number of words is correct {report['total'][0]}")
    assert report['feminine'][0] == 1, report['feminine']
    print(f"TEST:Total number of feminine words is correct {report['feminine'][0]}")
    assert report['masculine'][0] == 2
    print(f"TEST:Total number of masculine words is correct {report['masculine'][0]}")
    assert report['unspecified'][0] == 2, report['unspecified'][0]
    print(f"TEST:Total number of unspecified words is correct {report['unspecified'][0]}")

    print('All test counting done (v1.0 nouns list spa)')




def ratio_test():
    LANG = 'spa'
    
    DATA = ['Este desconcierto surge de los planes de cambiar el nombre de la Asamblea a Parlamento de Gales.', # 0 
            'empty sentence ',  # 0 
            'esposa esposa esposas et tu che va', # 3 female
            'abuelos abuelos abuelo', # abuelos is both unspecified and make: 3 male, 2 underspecified
            'esposa. esposa! ?esposa esposa', # testing tokenizer: 1 female 
            'ABU Abu aBU', # testing case:  3 female and 3 male
            ] 
    
    # data includes: 
    ## 36 words with white-space tokenization
    ## 41 with stanza

    file_dir = pathlib.Path('./tmp')
    file_dir.mkdir(exist_ok=True)
    
    # writing test data
    with open(file_dir/'test.txt', 'w') as f:
        for line in DATA:
            f.write(line+'\n')
    
    hb_counter = MultilingualGenderDistribution(store_hb_dir='./tmp', langs=[LANG], ft_model_path=None,  dataset_version='v1.0')
    with open(file_dir/'test.txt') as file:
        hb_counter.process_txt(file=file, clean_sample=clean_sample, max_samples=None, expected_langs=[LANG])
    report = hb_counter.gender_dist()
    
    count = report.iloc[0]
    ratio = report.iloc[1]
    
    n_words = count['total']
    for col in report.columns:
        if col == 'total':
            continue
        assert ratio[col] == count[col]/n_words*100, f"{col}: {ratio[col]} <> {count[col]/n_words}"
        print(f'TEST: {col} ratio correct {ratio[col]}')
    print('All test ratio done')
        

if __name__ == '__main__':
    count_test_spa()
    count_test_eng()
    count_test_eng_2()
    ratio_test()
    



