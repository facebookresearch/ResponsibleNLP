import csv 
import sys, pdb, os
sys.path.append('../../../filtering/')
from utils import *
import random
import argparse
import shutil

parser = argparse.ArgumentParser()

parser.add_argument('--dataset_name', default='ai2_arc', help='name of the dataset')
parser.add_argument('--out_dir', default='dataset_no_ques', help='path after removing questions')

args=parser.parse_args()

dataset_folder = "../../../datasets"

data_dir = os.path.join(dataset_folder, args.dataset_name)
out_dir = args.out_dir

if not os.path.exists(out_dir) :
    os.mkdir(out_dir)


out_folder = os.path.join(out_dir, 'test')
if not os.path.exists(out_folder) :
    os.mkdir(out_folder)

for category in os.listdir(data_dir) :

    ## Removing questions from all test files and saving them again
    if category == 'test':
        out_folder = os.path.join(out_dir, category)
        if not os.path.exists(out_folder) :
            os.mkdir(out_folder)

        for file in os.listdir(os.path.join(data_dir, category)):
            dataset = read_csv(os.path.join(data_dir, category, file))

            new_dataset = []
            for entry in dataset :

                options = entry[1:5]
                mapping = dict(zip('ABCD', options))

                # Get the answer from the original list
                answer = mapping[entry[-1]]
                answer_option = entry[-1]

                ## Using empty string for question
                new_entry = [' '] + options + [answer_option]

                new_dataset.append(new_entry)

            write_csv(new_dataset, os.path.join(out_folder, file))

    else:
        ## Saving non-test folders such as dev, val as is
        src_path = os.path.join(data_dir, category)
        dst_path = os.path.join(out_dir, category)
        shutil.copytree(src_path, dst_path)