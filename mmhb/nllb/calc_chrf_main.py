# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pandas as pd
import numpy as np
import json
from calc_chrf_utils import obtain_metric_scores

languages = ['spa', 'fra', 'ind', 'por', 'hin', 'ita', 'vie']

for lang in languages:
    # gender bias analysis -- only for eval_nouns
    df_all = pd.read_csv(f"mmhb_dataset/{lang}/devtest.csv", delimiter='\t')
    print(lang, "df_all.shape", df_all.shape)

    # limit to eval_nouns    
    eval_nouns = ["N051", "N052", "N063", "N064"]
    df_all = df_all[df_all["noun_id_main"].isin(eval_nouns)].reset_index(drop=True).copy()
    print(lang, "df_eval_nouns.shape", df_all.shape)

    ## EN-to-XX
    gender_groups = ["both", "feminine", "masculine"]
    for gender in gender_groups:
        obtain_metric_scores(df_all, "en-to-xx", lang, gender)

    ## XX-to-EN
    gender_groups = ["feminine", "masculine"]
    for gender in gender_groups:
        obtain_metric_scores(df_all, "xx-to-en", lang, gender)
