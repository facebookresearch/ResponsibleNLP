# Compute Multilingual Gender Distrubution

You will find below instructions to compute the holistic bias distribution of HuggingFace datasets. 
In a nutshell, given a dataset from the [huggingface hub](https://huggingface.co/datasets), you can use the script presented below to compute the distributions of gender and demographic groups based on the [Holistic Bias](https://arxiv.org/abs/2205.09209) dataset. 
In addition, you can call the function CountHolisticBias to use in a custom data processing pipeline. 

## Setup 

```
pip install fasttext
pip install datasets  
pip install stanza   
pip install nltk 
pip install pythainlp
```

`git clone git@github.com:facebookresearch/ResponsibleNLP.git` 

```
git checkout count_hb
cd ./ResponsibleNLP  
```

Download fastext language detection model:   
```
mkdir fasttext_models
wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
```
 
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



## Compute Biases of Hugging Face Datasets   

For instance: to compute the biases distribution of the dataset `Anthropic/hh-rlhf` of the field `chosen`, split `test`, processing 100 samples at most. 

```
python multilingual_gendered_nouns_dist/get_multilingual_gender_dist.py --dataset "Anthropic/hh-rlhf" --first_level_key 'chosen' --split test --max_samples 100 --langs en
```       

## Compute on text file


Compute gender distribution of a single file in  arb bel vie por eng spa :     

```
python multilingual_gendered_nouns_dist/get_multilingual_gender_dist.py  --file_dir ./data/flores200_dataset/devtest/ \
    --file_names arb_Arab.devtest bel_Cyrl.devtest vie_Latn.devtest por_Latn.devtest eng_Latn.devtest spa_Latn.devtest  \
    --langs arb bel vie por eng spa \
    --max_samples 100
```

## Call count function

To integrate the count function into an existing data processing pipeline (e.g. in Spark), 
the count_demographics method of MultilingualGenderDistribution can be called as showed in the example below:

```
from multilingual_gendered_nouns_dist.src.gender_counts import MultilingualGenderDistribution

# Load Holistic Bias list
counter = MultilingualGenderDistribution(store_hb_dir='./tmp', langs=['eng'], ft_model_path='./fasttext_models/lid.176.bin')

sample = "Authorities have not identified the man, but local CNN affiliates have named him as Simon Baccanello, a 46-year-old teacher at nearby Elliston Area School."

# detect language
lang_detected = counter.detect_language(text=sample)

# update counters for sample
counter.count_demographics(sample, lang_detected)

# printout final counts
hb_counter.printout_summary()

# Expected output:
#Out of 24 words:
#neutral words amounts for 0 (0.0%), masculine words amounts for 1 (4.2%), feminine words amounts for 0 (0.0%),
#Out of 1 samples:
#age-old samples amounts for 1 (100.0%),
#null-(none) samples amounts for 1 (100.0%),
```