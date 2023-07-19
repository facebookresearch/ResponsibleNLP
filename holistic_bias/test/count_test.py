#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import pathlib

import sys
sys.path.append('.')

from holistic_bias.src.hb_counts import CountHolisticBias
from holistic_bias.src.util import clean_sample




def count_test_v11():
    LANG = 'spa'
    
    DATA = ['Este desconcierto surge de los planes de cambiar el nombre de la Asamblea a Parlamento de Gales.', # 0 
            'empty sentence ',  # 0 
            'esposa esposa esposas et tu che va', # 3 female
            'abuelos abuelos abuelo', # abuelos is both unspecified and make: 3 male, 2 unspecified
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
    
    hb_counter = CountHolisticBias(store_hb_dir='./tmp', langs=[LANG], ft_model_path=None, only_gender=True, dataset_version='v1.1')
    with open(file_dir/'test.txt') as file:
        hb_counter.process_txt(file=file, clean_sample=clean_sample, max_samples=None, expected_langs=[LANG])
    report = hb_counter.gender_dist()
    
    

    assert report['total'][0] == 36
    print(f"TEST:Total number of words is correct {report['total'][0]}")
    assert report['female'][0] == 7
    print(f"TEST:Total number of female words is correct {report['female'][0]}")
    assert report['male'][0] == 3+3
    print(f"TEST:Total number of male words is correct {report['male'][0]}")
    assert report['neutral'][0] == 2, report['unspecified'][0]
    print(f"TEST:Total number of neutral words is correct {report['neutral'][0]}")

    print('All test counting done (v1.1 nouns list)')



def count_test_v12():
    LANG = 'spa'
    
    DATA = ['Este desconcierto surge de los planes de cambiar el nombre de la Asamblea a Parlamento de Gales.', # 0 
            'empty sentence ',  # 0 
            'esposa esposa esposas et tu che va', # 3 female
            'abuelos abuelos abuelo', # abuelos is both unspecified and make: 3 male, 2 unspecified
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
    
    hb_counter = CountHolisticBias(store_hb_dir='./tmp', langs=[LANG], ft_model_path=None, only_gender=True, dataset_version='v1.2')
    with open(file_dir/'test.txt') as file:
        hb_counter.process_txt(file=file, clean_sample=clean_sample, max_samples=None, expected_langs=[LANG])
    report = hb_counter.gender_dist()
    
    assert report['total'][0] == 36
    print(f"TEST:Total number of words is correct {report['total'][0]}")
    assert report['female'][0] == 7
    print(f"TEST:Total number of female words is correct {report['female'][0]}")
    assert report['male'][0] == 3+3
    print(f"TEST:Total number of male words is correct {report['male'][0]}")
    assert report['unspecified'][0] == 2, report['unspecified'][0]
    print(f"TEST:Total number of unspecified words is correct {report['unspecified'][0]}")

    print('All test counting done (v1.2 nouns list)')


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
    
    hb_counter = CountHolisticBias(store_hb_dir='./tmp', langs=[LANG], ft_model_path=None, only_gender=True, dataset_version='v1.1')
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
    count_test_v11()
    count_test_v12()
    ratio_test()



