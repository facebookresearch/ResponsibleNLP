# Multilingual Holistic Biases Counts

## Setup 

`pip install fasttext`
`pip install datasets`

`git clone git@github.com:facebookresearch/ResponsibleNLP.git`
see https://github.com/facebookresearch/ResponsibleNLP/tree/main/holistic_bias for setting it up.

`git checkout count_hb`
`cd ./ResponsibleNLP`

Download fastext language detection model: 
`mkdir fasttext_models`
`wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin`

## Compute Biases of Hugging Face Datasets

For instance: to compute the biases distribution of the dataset `Anthropic/hh-rlhf` of the field `chosen`, split `test`, max_samples 100

`python holistic_bias/get_stats_hb.py --dataset "Anthropic/hh-rlhf" --first_level_key 'chosen' --split test`

## Call count function

To integrate the count function into an existing data processing pipeline (e.g. in Spark), 
the count_demographics method of CountHolisticBias should be called as showed in the example:

```
from holistic_bias.src.hb_counts import CountHolisticBias

# Load Holistic Bias list
hb_counter = CountHolisticBias(store_hb_dir='./tmp', langs=['en'], ft_model_path='./fasttext_models/lid.176.bin')

    sample = "Authorities have not identified the man, but local CNN affiliates have named him as Simon Baccanello, a 46-year-old teacher at nearby Elliston Area School."

# detect language
lang_detected = hb_counter.detect_language(text=sample)
# update counters for sample
hb_counter.count_demographics(sample, lang_detected)
# printout final counts
hb_counter.printout_summary()
# Expected output:
#Out of 24 words:
#neutral words amounts for 0 (0.0%), male words amounts for 1 (4.2%), female words amounts for 0 (0.0%),

#Out of 1 samples:
#age-old samples amounts for 1 (100.0%),
#null-(none) samples amounts for 1 (100.0%),
```