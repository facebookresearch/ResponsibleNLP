# The Gender-GAP (Gender-Aware Polyglot) Pipeline

You will find below instructions to compute the run the Gender-GAP pipeline. 
In a nutshell, given a dataset, you can use the script presented below to compute the distributions of genders. 

The pipeline is showcased in the paper *The Gender-GAP Pipeline: A Gender-Aware Polyglot Pipeline for Gender Characterisation in 55 Languages*. 

## Setup 

The code was tested with python 3.9. 

```
pip install -r requirements.txt
git clone https://github.com/facebookresearch/ResponsibleNLP.git
cd multilingual_gendered_nouns_dist
```

## Run Gender-GAP

Compute gender distribution of a single file in arb bel vie por eng spa:     

```
python gender_gap_pipeline/get_multilingual_gender_dist.py \
 --file_dir $DATA_DIR/ \
--file_names arb_Arab.devtest bel_Cyrl.devtest vie_Latn.devtest por_Latn.devtest eng_Latn.devtest spa_Latn.devtest \
--langs arb bel vie por eng spa
```

The final distribution will be written to ```./reports/report.csv``` by default.

## Extra Flags:

Add ```--printout_latex``` to printout the distribution per language in a latex-ready format    
Add ```--max_samples 100``` flag to limit the distribution to the top 100 samples.  
Add ```--write_dir $TARGET_DIR``` to specify the directory to write the report.csv to.  

 
## Languages supported and language code

| Language    | Code |
|-------------|------|
| Arabic      | arb  |
| Czech       | ces  |
| Greek       | ell  |
| Hindi       | hin  |
| Georgian    | kat  |
| Marathi     | mar  |
| Portuguese  | por  |
| Swedish     | swe  |
| Turkish     | tur  |
| Assamese    | asm  |
| Kurdish     | ckb  |
| English     | eng  |
| Hungarian   | hun  |
| Khakas      | khk  |
| Maltese     | mlt  |
| Romanian    | ron  |
| Swahili     | swh  |
| Urdu        | urd  |
| Belarusian  | bel  |
| Mandarin    | cmn  |
| Estonian    | est  |
| Indonesian  | ind  |
| Kirghiz     | kir  |
| Dutch       | nld  |
| Russian     | rus  |
| Tamil       | tam  |
| Uzbek       | uzn  |
| Bengali     | ben  |
| Welsh       | cym  |
| Finnish     | fin  |
| Italian     | ita  |
| Korean      | kor  |
| Punjabi     | pan  |
| Slovak      | slk  |
| Telugu      | tel  |
| Vietnamese  | vie  |
| Bulgarian   | bul  |
| Danish      | dan  |
| Irish       | gle  |
| Kannada     | kan  |
| Luganda     | lug  |
| Polish      | pol  |
| Spanish     | spa  |
| Thai        | tha  |
| Zulu        | zul  |

