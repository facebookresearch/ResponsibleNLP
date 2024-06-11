# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pandas as pd
import glob
from utils_plot_etox import (
    plot_butterfly,
    plot_axis_distribution,
)

languages = ['eng', 'spa', 'fra', 'ind', 'por', 'hin', 'ita', 'vie']

dat_path_align = "etox/etox_results/raw_results/{lang}_alignTrue_etox_raw_results.csv"
dat_path_trans = "etox/etox_results/raw_results/{lang}_alignFalse_etox_raw_results.csv"
out_folder = "etox/etox_results"

## descriptor table
df_desc = pd.read_csv(f"eng_descriptors_full_id.tsv", delimiter='\t')
df_desc.columns = ["pattern", "axis", "bucket", "type", "descriptor"]
df_desc = df_desc[["pattern", "axis"]]

## load etox prediction results
d_pred = dict()
for lang in languages:
    df_align = pd.read_csv(dat_path_align.format(lang=lang), sep='\t')
    df_trans = pd.read_csv(dat_path_trans.format(lang=lang), sep='\t')

    df_align.rename(columns={"etox_result": 'etox_result_align', "tgt_etox_qty": "tgt_etox_qty_align"}, inplace=True)
    df_trans.rename(columns={"etox_result": 'etox_result_trans', "tgt_etox_qty": "tgt_etox_qty_trans"}, inplace=True)
    
    df = pd.merge(df_align, df_trans, on=['Unnamed: 0', 'index', 'sentence_eng', 'pattern_id_main',
       'noun_id_main', 'desc_id_main', 'split', 'both', 'feminine',
       'masculine', 'both_count', 'feminine_count', 'masculine_count', 'lang',
       f'sentence_{lang}', f'translate_{lang}', 'translate_eng'], how='inner')
    
    etox_col = "etox_en" if lang == "eng" else "etox_xx"
    df[etox_col+"_align"] = df["tgt_etox_qty_align"]
    df[etox_col+"_trans"] = df["tgt_etox_qty_trans"]
    
    d_pred[lang] = df

## eng
df_eng = d_pred["eng"]
df_eng.drop(['lang'], axis=1, inplace=True)
df_eng = df_eng.sort_values(by=['pattern_id_main','noun_id_main','desc_id_main']).reset_index(drop=True).copy()
print("df_eng.shape:", df_eng.shape)

## other langs
for lang in languages:
    if lang == "eng":
        continue
    print(lang)
    
    df = d_pred[lang].copy()
    df = df.sort_values(by=['pattern_id_main','noun_id_main','desc_id_main']).reset_index(drop=True).copy()
    print(f"df_{lang}.shape:", df.shape)
    
    # merge with english, to create XX-to-EN & EN-to-XX
    df_merge = pd.merge(df, df_eng, on=['pattern_id_main','noun_id_main','desc_id_main','sentence_eng','split'], how='inner')
    df_merge = df_merge.sort_values(by=['pattern_id_main','noun_id_main','desc_id_main']).reset_index(drop=True).copy()
    print(f"df_merge_{lang}.shape:", df_merge.shape)
    
    # translation has more toxic words than both source and reference
    df_merge["en_to_xx_added_toxicity"] = np.where(
        (df_merge["etox_xx_trans"]>df_merge["etox_xx_align"]) & (df_merge["etox_xx_trans"]>df_merge["etox_en_align"]),
        True, False
        )
    df_merge["xx_to_en_added_toxicity"] = np.where(
        (df_merge["etox_en_trans"]>df_merge["etox_xx_align"]) & (df_merge["etox_en_trans"]>df_merge["etox_en_align"]),
        True, False
        )
    
    df_merge[df_merge["xx_to_en_added_toxicity"]==True]['etox_result_trans_y']
    df_merge[df_merge["xx_to_en_added_toxicity"]==True]['translate_eng_y']
    df_eng[~df_eng['etox_result_trans'].isnull()]['etox_result_trans']
    df_eng[~df_eng['etox_result_trans'].isnull()]['translate_eng']

    df_merge = pd.merge(df_merge, df_desc, left_on="desc_id_main", right_on="pattern", how='left')
    assert df_merge['axis'].nunique() == df_desc['axis'].nunique()
    assert df_merge["en_to_xx_added_toxicity"].isnull().sum() == 0
    assert df_merge["xx_to_en_added_toxicity"].isnull().sum() == 0
    assert df_merge["desc_id_main"].isnull().sum() == 0
    assert (df_merge["desc_id_main"] != df_merge["pattern"]).sum() == 0
    
    df_merge.to_csv(f"{out_folder}/{lang}_etox_raw_results.csv", sep='\t', index=False)
    
    df_stats = df_merge.groupby(['axis']).agg({
        'en_to_xx_added_toxicity': "sum", 
        'xx_to_en_added_toxicity': "sum",
    }).reset_index()
    df_stats['lang'] = lang
    df_stats['en_to_xx_added_toxicity_prob'] = df_stats['en_to_xx_added_toxicity'] / df_merge["en_to_xx_added_toxicity"].count()
    df_stats['xx_to_en_added_toxicity_prob'] = df_stats['xx_to_en_added_toxicity'] / df_merge["xx_to_en_added_toxicity"].count()
    
    print("en_to_xx added toxicity:", 
          format(df_stats['en_to_xx_added_toxicity_prob'].sum() * 100, ".2f")
    )
    print("xx_to_en added toxicity:", 
          format(df_stats['xx_to_en_added_toxicity_prob'].sum() * 100, ".2f")
    )
    
    df_stats.to_csv(f"{out_folder}/{lang}_etox_agg_results.csv", sep='\t', index=False)


## agg all languages
df_pivots = []
for lang in languages:
    if lang == "eng":
        continue
    
    df = pd.read_csv(f"{out_folder}/{lang}_etox_agg_results.csv", delimiter='\t') #.fillna('')
    
    # Pivot en-xx and xx-en separately
    pivoted_df1 = df.pivot(index='lang', columns='axis', values='en_to_xx_added_toxicity_prob').reset_index()
    pivoted_df1['translation_type'] = "en_to_xx"
    pivoted_df2 = df.pivot(index='lang', columns='axis', values='xx_to_en_added_toxicity_prob').reset_index()
    pivoted_df2['translation_type'] = "xx_to_en"
    
    df_pivot = pd.concat([pivoted_df1, pivoted_df2], ignore_index=True)
    df_pivot.columns.name = None
    df_pivots.append(df_pivot)

df_all = pd.concat(df_pivots, ignore_index=True, sort=False)
df_all.fillna(0, inplace=True)

# total toxic percent
float_cols = df_all.select_dtypes(include=['float64']).columns
df_all['sum_toxic'] = df_all[float_cols].sum(axis=1)

df_all.to_csv(f"{out_folder}/etox_agg_results.csv", sep='\t', index=False)


## plot butterfly
df_butterfly = df_all.pivot(index='lang', columns='translation_type', values='sum_toxic').reset_index()
df_butterfly.columns.name = None
df_butterfly.set_index('lang', inplace=True)

plot_butterfly(df_butterfly, out_folder)


## plot hb axis distribution
df_en_to_xx = df_all[df_all["translation_type"] == "en_to_xx"].reset_index(drop=True).copy()
df_xx_to_en = df_all[df_all["translation_type"] == "xx_to_en"].reset_index(drop=True).copy()

plot_axis_distribution(df_en_to_xx, df_desc, "en_to_xx", out_folder)
plot_axis_distribution(df_xx_to_en, df_desc, "xx_to_en", out_folder)
