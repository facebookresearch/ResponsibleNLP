# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pandas as pd
import numpy as np
import json
from sacrebleu.metrics import METRICS as sacreBLEU_metrics

def obtain_metric_scores(df_all, task, lang, gender):
    if task == "en-to-xx":
        assert gender in ["both", "feminine", "masculine"], "gender not supported"
        ref_col = gender
        pred_col = f"translate_{lang}"
        out_col = f"chrf_eng_to_{lang}_{gender}"
    elif task == "xx-to-en":
        assert gender in ["feminine", "masculine"], "gender not supported"
        ref_col = "sentence_eng"
        pred_col = "translate_eng"
        out_col = f"chrf_{lang}_to_eng_{gender}"
    else:
        assert False, "task not supported"
    
    # only for non-nan rows
    df = df_all[~df_all[ref_col].isnull()].reset_index(drop=True).copy()
    
    if task == "xx-to-en":
        df = df[df['gender_group'] == gender].reset_index(drop=True).copy()
    
    print(lang, task, gender, "df.shape", df.shape)
    
    metric = "chrf"
    reference_sentences = [l.strip() for l in df[ref_col].to_list()]
    prediction_sentences = [l.strip() for l in df[pred_col].to_list()]

    scorer = sacreBLEU_metrics["CHRF"]()

    scores = []
    for pred, ref in zip(prediction_sentences, reference_sentences):
        try:
            ref = eval(ref)
            assert isinstance(ref, list), breakpoint()
        except:
            ref = [ref]
            assert task == "xx-to-en", breakpoint()
            
        score = scorer.sentence_score(pred, ref)
        score_in_json = json.loads(
            score.format(signature=str(scorer.get_signature()), is_json=True)
        )
        scores.append(score_in_json["score"])

    assert len(scores) == len(reference_sentences)
    
    df[out_col] = scores
    df.to_csv(f"nllb/chrf_results/raw_results/{out_col}_eval.csv", sep='\t', index=False)
