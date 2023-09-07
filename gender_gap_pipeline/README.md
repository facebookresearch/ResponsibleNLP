# The Gender-GAP (Gender-Aware Polyglot) Pipeline

You will find below instructions to run the Gender-GAP pipeline. 
In a nutshell, given a dataset, you can use `python gender_gap_pipeline/get_multilingual_gender_dist.py` to compute the distributions of genders across the three classes: ***feminine, masculine and unspecified*** on text files, .gzip files or Hugging-Face hosted [datasets](https://huggingface.co/datasets).

The pipeline is described and showcased in the paper [*The Gender-GAP Pipeline: A Gender-Aware Polyglot Pipeline for Gender Characterisation in 55 Languages*](https://arxiv.org/pdf/2308.16871.pdf) and illustrated here:

![The Gender-GAP Pipeline Illustrated](https://github.com/facebookresearch/ResponsibleNLP/blob/main/gender_gap_pipeline/GenderGAP_img.png)


## Setup 

The code was tested with python 3.9. 

```
git clone https://github.com/facebookresearch/ResponsibleNLP.git
cd ResponsibleNLP/gender_gap_pipeline/
pip install -r requirements.txt
cd ../
```

NBs:
- Gender-GAP uses [Stanza](https://stanfordnlp.github.io/stanza/available_models.html) for word segmentation. 
- Stanza relies on pytorch for deep-learning-based word segmentation. The pipeline will therefore be significantly faster running on GPU. However, CPU is supported.     
- If you face any issues with importing pytorch, first install pytorch independently (e.g. with `conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia`
) then run `pip install -r requirements.txt` to install stanza.


If the language of your data is unknown, language detection is supported in the pipeline. To use language detection, download the fastext language detection model (only needed when --langs is not provided, or --lang_detect is triggered):   
```
cd ./ResponsibleNLP
mkdir fasttext_models
cd fasttext_models
wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
```


## Run the Gender-GAP Pipeline

### Gender-GAP on Text files or .gzip

Compute gender distribution of txt files. It also supports gzip text files ending with .gzip . 

```
python gender_gap_pipeline/get_multilingual_gender_dist.py 
--file_dir $DATA_DIR_1 $DATA_DIR_2  # Single or space-seperated list of directories where files_names are located. If a single directory is provided, it will assume that all file names are in it. If several directory are provided, one directory per file names is assumed. 
--file_names $file_1 $file_2   # Space-seperated list of file names
--langs $lang1 $lang2  # [OPTIONAL] single or space-seperated list of languages. If a single language is provideed, it will assume that all files are in this language. if --langs is not provided: language detection is triggered. Cf.  'Languages supported and language code' section for supported languages
```

For instance, you can run Gender-GAP on [FLORES-200 devtest data](https://github.com/facebookresearch/flores) on the Arabic, Belarusian, Vietnamese, Porthughese, English and Spanish split located in `$DATA_DIR` with:

```
python gender_gap_pipeline/get_multilingual_gender_dist.py \
--file_dir $DATA_DIR/ \
--file_names arb_Arab.devtest bel_Cyrl.devtest vie_Latn.devtest por_Latn.devtest eng_Latn.devtest spa_Latn.devtest \
--langs arb bel vie por eng spa
```

This will output:

```
>> 1012 sentences were processed
>> reports/report.csv
```

with reports/report.csv: 
|dataset|lang|masculine          |feminine           |unspecified        |total|n_doc_w_match     |ste_diff_fem_masc     |
|-------|----|-------------------|-------------------|-------------------|-----|------------------|----------------------|
|flores |arb |0.04696857019844221|0.05088261771497906|0.09393714039688442|25549|4.051383399209486 |0.00019570605268052815|
|flores |bel |0.4344951355435912 |0.16057428922263153|0.4439406819684519 |21174|12.747035573122531|0.000529808161936923  |
|flores |vie |0.300835213553418  |0.13854253255749516|1.4408423385979494 |25263|30.632411067193676|0.0004016076337256879 |
|flores |por |0.07828917549136759|0.10301207301495735|0.33787959948906016|24269|10.869565217391305|0.0002733229399289237 |
|flores |eng |0.06462453147214682|0.1206324587480074 |0.3791305846365947 |23211|11.16600790513834 |0.00028249642743351656|
|flores |spa |0.2007733491969066 |0.10410469958358119|0.26026174895895304|26896|12.25296442687747 |0.00033663613796724475|


### Gender-GAP on HuggingFace Datasets

Compute gender distribution of any datasets hosted in the HuggingFace [dataset hub](https://huggingface.co/datasets).

```
python gender_gap_pipeline/get_multilingual_gender_dist.py \
--hf_datasets  "$HUGGING_FACE_DATASET" \ # Single or space-seperated list of hugging-face datasets as found in https://huggingface.co/datasets 
--first_level_key $KEY \  #  First-level key (or list of first-level keys) of the hugging face datasets provided in --hf_datasets
--second_level_key $KEY \  # [OPTIONAL] For  datasets with a two-level dictionary structure: provide the second-level key (or list of second-level keys) to run Gender-GAP on. 
--split $SPLIT  \ # split of the dataset (or list of splits) to run the pipeline on (e.g. 'valid' 'test')
--langs $LANG \ #  # [OPTIONAL] single or space-seperated list of languages. If a single language is provideed, it will assume that all files are in this language. if --langs is not provided: language detection is triggered.  Cf. 'Languages supported and language code' section for supported languages
```


For instance, you can run Gender-GAP on the [dell-research-harvard/AmericanStories](https://huggingface.co/datasets/dell-research-harvard/AmericanStories) dataset on the 'article' text field on the '1804' .


```
python gender_gap_pipeline/get_multilingual_gender_dist.py \
--hf_datasets  "dell-research-harvard/AmericanStories" \
--first_level_key 'article' \
--split '1804'  \
--langs eng 
```

This will output:
```
>> 103 samples were counted
>> REPORT on dell-research-harvard/AmericanStories
>> reports/report.csv
```
with report.csv:

|dataset                              |lang|masculine          |feminine          |unspecified        |total|n_doc_w_match     |ste_diff_fem_masc   |
|-------------------------------------|----|-------------------|------------------|-------------------|-----|------------------|--------------------|
|dell-research-harvard/AmericanStories-1804-article|eng |0.25665704202759065|0.0962463907603465|0.12832852101379533|3117 |11.650485436893204|0.001063826559203403|


The final distribution will be written to ```./reports/report.csv``` by default.

## Extra Options

Add ```--printout_latex``` to printout the distribution per language in a latex-ready format    
Add ```--max_samples 100``` flag to limit the distribution to the top 100 samples.  
Add ```--write_dir $TARGET_DIR``` to specify the directory to write the report.csv to.  
Add ```--skip_failed_files``` to skip failed files processing and not to raise an error when processing sequentially multiple files. A Warning will be logged if a file fails to be processed. 

 
## Languages supported and language code

The Gender-GAP pipeline supports the 55 languages listed here:

|code|Language                     |
|----|-----------------------------|
|arb |Modern Standard Arabic              |
|asm |Assamese                     |
|bel |Belarusian                   |
|ben |Bengali                      |
|bul |Bulgarian                    |
|cat |Catalan                      |
|ces |Czech                        |
|ckb |Central Kurdish              |
|cmn |Mandarin Chinese             |
|cym |Welsh                        |
|dan |Danish                       |
|deu |German                       |
|ell |Modern Greek                 |
|eng |English                      |
|est |Estonian                     |
|fin |Finnish                      |
|fra |French                       |
|gle |Irish                        |
|hin |Hindi                        |
|hun |Hungarian                    |
|ind |Indonesian                   |
|ita |Italian                      |
|jpn |Japanese                     |
|kan |Kannada                      |
|kat |Georgian                     |
|khk |Halh Mongolian               |
|kir |Kirghiz                      |
|kor |Korean                       |
|lit |Lithuanian                   |
|lug |Ganda                        |
|lvs |Standard Latvian             |
|mar |Marathi                      |
|mlt |Maltese                      |
|nld |Dutch                        |
|pan |Panjabi                      |
|pes |Iranian Persian              |
|pol |Polish                       |
|por |Portuguese                   |
|ron |Romanian                     |
|rus |Russian                      |
|slk |Slovak                       |
|slv |Slovenian                    |
|spa |Spanish                      |
|swe |Swedish                      |
|swh |Swahili                      |
|tam |Tamil                        |
|tel |Telugu                       |
|tgl |Tagalog                      |
|tha |Thai                         |
|tur |Turkish                      |
|urd |Urdu                         |
|uzn |Northern Uzbek               |
|vie |Vietnamese                   |
|yue |Yue Chinese                  |
|zul |Zulu                         |




## Citation

If you use the Gender-GAP pipeline or the underlying lexicon in your work, please cite:

```bibtex
@article{muller2023gender,
  title={The Gender-GAP Pipeline: A Gender-Aware Polyglot Pipeline for Gender Characterisation in 55 Languages},
  author={Muller, Benjamin and Alastruey, Belen and Hansanti, Prangthip and Kalbassi, Elahe and Ropers, Christophe and Smith, Eric Michael and Williams, Adina and Zettlemoyer, Luke and Andrews, Pierre and Costa-juss{\`a}, Marta R},
  journal={arXiv preprint arXiv:2308.16871},
  year={2023}
}
```