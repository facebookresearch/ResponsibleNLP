# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import glob
import pandas as pd
from utils_plot_chrf import plot_chrf, plot_chrf_demo

filenames = glob.glob(f"nllb/chrf_results/raw_results/chrf_*_to_*_*_eval.csv")

languages = ['eng', 'spa', 'fra', 'por', 'ita']
gender_groups = ["both", "feminine", "masculine", "None"]

## descriptor table
df_desc = pd.read_csv(f"mhb/eng_descriptors_full_id.tsv", delimiter='\t')
df_desc.columns = ["pattern", "axis", "bucket", "type", "descriptor"]
df_desc = df_desc[["pattern", "axis"]]

df_stats_chrf = []
df_stats_desc = []

for filename in filenames:
    lang1 = filename.split("/")[-1].split("_")[1]
    lang2 = filename.split("/")[-1].split("_")[3]
    gender = filename.split("/")[-1].split("_")[-2]
    
    if lang1 not in languages or lang2 not in languages:
        continue
    assert gender in gender_groups
    
    df = pd.read_csv(filename, delimiter='\t')
    # analysis limit to devtest set
    df = df[df['split'] == 'devtest'].reset_index(drop=True).copy()
    
    out_col = f"chrf_{lang1}_to_{lang2}_{gender}"
    
    count = df.shape[0]
    avg_score = df[out_col].sum() / count
    
    df_stats_chrf.append({
        "task": "en-to-xx" if lang1 == "eng" else "xx-to-en",
        "gender": gender,
        "lang": lang1 if lang1 != "eng" else lang2,
        "count": count,
        "avg_chrf": avg_score,
    })
    
    df_merge = pd.merge(df, df_desc, left_on="desc_id_main", right_on="pattern", how='left')

    assert df_merge['axis'].nunique() == df_desc['axis'].nunique()
    assert df_merge["desc_id_main"].isnull().sum() == 0
    assert (df_merge["desc_id_main"] != df_merge["pattern"]).sum() == 0
    
    df_stats = df_merge.groupby(['axis']).agg({
        out_col: ['count', 'mean'],
    }).reset_index()
    
    df_stats.columns = ['axis', 'count', 'avg_chrf']
    
    df_stats['task'] = "en-to-xx" if lang1 == "eng" else "xx-to-en"
    df_stats['gender'] = gender
    df_stats['lang'] = lang1 if lang1 != "eng" else lang2
    
    df_stats_desc.append(df_stats)
    
df_stats_chrf = pd.DataFrame(df_stats_chrf)
df_stats_desc = pd.concat(df_stats_desc, axis=0)

out_folder = "nllb/chrf_results"
df_stats_chrf.to_csv(f"{out_folder}/df_stats_chrf.csv")
df_stats_desc.to_csv(f"{out_folder}/df_stats_desc.csv")


### 1) plot df_stats_desc
df_stats_desc_en_to_xx = df_stats_desc[df_stats_desc['task'] == "en-to-xx"].reset_index(drop=True).copy()
df_stats_desc_xx_to_en = df_stats_desc[df_stats_desc['task'] == "xx-to-en"].reset_index(drop=True).copy()

df = df_stats_desc_en_to_xx.groupby(['axis','gender']).agg({
        "avg_chrf": ['mean'],
    }).reset_index()
df.columns = ['axis','gender','avg_chrf']
df_pivot = df.pivot(index='axis', columns='gender', values='avg_chrf').reset_index()
df_pivot.columns.name = None  # Remove the columns name
df_stats_desc_en_to_xx_pivot = df_pivot.copy()

df = df_stats_desc_xx_to_en.groupby(['axis','gender']).agg({
        "avg_chrf": ['mean'],
    }).reset_index()
df.columns = ['axis','gender','avg_chrf']
df_pivot = df.pivot(index='axis', columns='gender', values='avg_chrf').reset_index()
df_pivot.columns.name = None  # Remove the columns name
df_stats_desc_xx_to_en_pivot = df_pivot.copy()

out_folder = "nllb/chrf_results"
plot_chrf_demo(df_stats_desc_en_to_xx_pivot, out_folder, "en-to-xx")
plot_chrf_demo(df_stats_desc_xx_to_en_pivot, out_folder, "xx-to-en")
df_stats_desc_en_to_xx_pivot.to_csv(f"{out_folder}/df_stats_desc_en_to_xx_pivot.csv")
df_stats_desc_xx_to_en_pivot.to_csv(f"{out_folder}/df_stats_desc_xx_to_en_pivot.csv")


### 2) plot df_stats_chrf
df_stats_chrf_en_to_xx = df_stats_chrf[df_stats_chrf['task'] == "en-to-xx"].reset_index(drop=True).copy()
df_stats_chrf_xx_to_en = df_stats_chrf[df_stats_chrf['task'] == "xx-to-en"].reset_index(drop=True).copy()

LANG_CODES = {
    'ind': 'ind_Latn',    
    'hin': 'hin_Deva',
    'deu': 'deu_Latn',
    'eng': 'eng_Latn',
    'fra': 'fra_Latn',
    'tha': 'tha_Thai',
    'spa': 'spa_Latn',
    'ita': 'ita_Latn',
    'por': 'por_Latn',
    'vie': 'vie_Latn',
}

df = df_stats_chrf_en_to_xx.copy()
df_pivot = df.pivot(index='lang', columns='gender', values='avg_chrf').reset_index()
df_pivot.columns.name = None  # Remove the columns name
df_pivot['lang_code'] = df_pivot['lang'].apply(lambda x: LANG_CODES[x])
df_stats_chrf_en_to_xx = df_pivot.copy()

df = df_stats_chrf_xx_to_en.copy()
df_pivot = df.pivot(index='lang', columns='gender', values='avg_chrf').reset_index()
df_pivot.columns.name = None  # Remove the columns name
df_pivot['lang_code'] = df_pivot['lang'].apply(lambda x: LANG_CODES[x])
df_stats_chrf_xx_to_en = df_pivot.copy()

out_folder = "nllb/chrf_results"
plot_chrf(df_stats_chrf_en_to_xx, out_folder, "en-to-xx")
plot_chrf(df_stats_chrf_xx_to_en, out_folder, "xx-to-en")
df_stats_chrf_en_to_xx.to_csv(f"{out_folder}/df_stats_chrf_en_to_xx.csv")
df_stats_chrf_xx_to_en.to_csv(f"{out_folder}/df_stats_chrf_xx_to_en.csv")
