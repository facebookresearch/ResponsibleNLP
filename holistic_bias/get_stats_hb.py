"""
CountHolisticBias loads the holistic biases templates and gender nouns collections and compute gender and demographics distribution of given datasets.

The dataset can be loaded from the HF datasets library or from a text file. 



e.g.  python holistic_bias/get_stats_hb.py --dataset "Anthropic/hh-rlhf" --first_level_key 'chosen' --split test --max_samples 10
"""
import argparse
import re
import sys

import typing as tp
from collections import Counter
from pathlib import Path
from tqdm import tqdm

sys.path.append('.')

from datasets import load_dataset
from holistic_bias.src.hb_counts import CountHolisticBias
def clean_sample(text: str):
    text = text.replace('\n', ' ')
    return text.strip()

# Define a function for language detection

if __name__ == '__main__':
    
    # Load the pre-trained language identification model
    

    parser = argparse.ArgumentParser(description='Example of using argparse')

    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--first_level_key', type=str, required=True)
    parser.add_argument('--second_level_key', type=str, default=None)
    parser.add_argument('--max_samples', default=None)

    # FIX MESSAGEING
    args = parser.parse_args()
    
    dataset = load_dataset(args.dataset) # e.g. "HuggingFaceH4/stack-exchange-preferences"
    hb_counter = CountHolisticBias(store_hb_dir='./tmp', langs=['en'], ft_model_path='./fasttext_models/lid.176.bin')
    
    hb_counter.process_dataset(dataset, split=args.split, 
                               first_level_key=args.first_level_key, # 'answers'
                               second_level_key=args.second_level_key,  # 'text'
                              clean_sample=clean_sample, 
                              max_samples=args.max_samples)
    
    hb_counter.printout_summary()
    
    # call on a text line: 
    
    
    
    
