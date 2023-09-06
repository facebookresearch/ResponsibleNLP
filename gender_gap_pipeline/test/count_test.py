#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import pathlib

import sys

sys.path.append(".")

from gender_gap_pipeline.src.gender_counts import GenderGAP
from gender_gap_pipeline.src.util import clean_sample


def count_test_eng():
    LANG = "eng"

    DATA = [
        "I am a boy!",  # 1
        "ABU Abu aBU",  # testing case:  3 female and 3 male
    ]

    # data includes:
    ## 36 words with white-space tokenization
    ## 41 with stanza

    file_dir = pathlib.Path("./tmp")
    file_dir.mkdir(exist_ok=True)

    # writing test data
    with open(file_dir / "test.txt", "w") as f:
        for line in DATA:
            f.write(line + "\n")

    hb_counter = GenderGAP(
        store_hb_dir="./tmp", lang=LANG, ft_model_path=None, dataset_version="v1.0"
    )
    hb_counter.process_txt_file(file_dir=file_dir / "test.txt", clean_sample=clean_sample,  
                                max_samples=None,  return_vec=True)
    hb_counter.gender_dist(info_file='test')
    
    assert hb_counter.stat["total"] == 8, hb_counter.stat
    print(f"TEST:Total number of words is correct {hb_counter.stat['total']}")
    assert hb_counter.stat["feminine"] == 0,  hb_counter.stat["feminine"]
    print(f"TEST:Total number of feminine words is correct {hb_counter.stat['feminine']}")
    assert hb_counter.stat["masculine"]*hb_counter.stat["total"] == 100, hb_counter.stat
    print(f"TEST:Total number of masculine words is correct {hb_counter.stat['masculine']}")
    assert hb_counter.stat["unspecified"] == 0, hb_counter.stat["unspecified"]
    print(
        f"TEST:Total number of unspecified words is correct {hb_counter.stat['unspecified']}"
    )

    print(f"All test counting done (v1.0 nouns list {LANG})")


def count_test_spa():
    LANG = "spa"

    DATA = [
        "Este desconcierto surge de los planes de cambiar el nombre de la Asamblea a Parlamento de Gales.",  # 0
        "empty sentence ",  # 0
        "esposa esposa esposas et tu che va",  # 3 female
        "abuelos abuelos abuelo",  # abuelos is both unspecified and make: 3 male, 2 unspecified
        "esposa . esposa! ?esposa esposa",  # testing tokenizer: 4 female
        "ABU Abu aBU",  # testing case:  3 female and 3 male
    ]

    # data includes:
    ## 36 words with white-space tokenization
    ## 41 with stanza

    file_dir = pathlib.Path("./tmp")
    file_dir.mkdir(exist_ok=True)

    # writing test data
    with open(file_dir / "test.txt", "w") as f:
        for line in DATA:
            f.write(line + "\n")

    hb_counter = GenderGAP(
        store_hb_dir="./tmp", lang=LANG, ft_model_path=None, dataset_version="v1.0"
    )
    
    hb_counter.process_txt_file(file_dir=file_dir / "test.txt", clean_sample=clean_sample,  
                                    max_samples=None,  return_vec=True)
        
    hb_counter.gender_dist(info_file='test')

    print(hb_counter.stat)
    
    assert hb_counter.stat["total"] == 40, hb_counter.stat
    print(f"TEST:Total number of words is correct {hb_counter.stat['total']}")
    assert hb_counter.stat['feminine'] == (7 + 3)/hb_counter.stat["total"]*100, hb_counter.stat
    print(f"TEST:Total number of feminine words is correct {hb_counter.stat['feminine']}")
    assert hb_counter.stat["masculine"] == (3 + 3)/hb_counter.stat["total"]*100, hb_counter.stat
    print(f"TEST:Total number of masculine words is correct {hb_counter.stat['masculine']}")
    assert hb_counter.stat["unspecified"] == 2/hb_counter.stat["total"]*100, hb_counter.stat
    print(
        f"TEST:Total number of unspecified words is correct {hb_counter.stat['unspecified']}"
    )

    print(f"All test counting done (v1.0 nouns list {LANG})")


def count_test_eng_2():
    LANG = "eng"

    DATA = [
        "my dad is a great guy",  # 2
        "no no yes !  ",  # 0
        "esposa esposa esposas et tu che va",  # 0
        "parents parent ",  # 2 neutral
        "girls ",
    ]

    # data includes:
    ## 36 words with white-space tokenization
    ## 41 with stanza

    file_dir = pathlib.Path("./tmp")
    file_dir.mkdir(exist_ok=True)

    # writing test data
    with open(file_dir / "test.txt", "w") as f:
        for line in DATA:
            f.write(line + "\n")

    hb_counter = GenderGAP(
        store_hb_dir="./tmp", lang=LANG, ft_model_path=None, dataset_version="v1.0"
    )
    
    hb_counter.process_txt_file(
            file_dir=file_dir / "test.txt",
            clean_sample=clean_sample,
            max_samples=None,
            return_vec=True,
        )
    hb_counter.gender_dist(info_file='test')

    print(hb_counter.stat)
    assert hb_counter.stat["total"] == 20
    print(f"TEST:Total number of words is correct {hb_counter.stat['total']}")
    assert hb_counter.stat["feminine"] == 1/hb_counter.stat["total"]*100, hb_counter.stat["feminine"]
    print(f"TEST:Total number of feminine words is correct {hb_counter.stat['feminine']}")
    assert hb_counter.stat["masculine"] == 2/hb_counter.stat["total"]*100, hb_counter.stat
    print(f"TEST:Total number of masculine words is correct {hb_counter.stat['masculine']}")
    assert hb_counter.stat["unspecified"] == 2/hb_counter.stat["total"]*100, hb_counter.stat["unspecified"]
    print(
        f"TEST:Total number of unspecified words is correct {hb_counter.stat['unspecified']}"
    )

    print(f"All test counting done (v1.0 nouns list {LANG})")


if __name__ == "__main__":
    count_test_eng()
    count_test_spa()
    count_test_eng_2()
    
