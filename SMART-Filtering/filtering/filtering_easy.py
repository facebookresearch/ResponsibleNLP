# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the CC BY NC license found in the
# LICENSE-CC-BY-NC file in the project directory.

import os, pdb
from utils import *
import random

class FilterEasyClass:
    def __init__(self, CONFIG):
        self.CONFIG = CONFIG
        self.data_dir = CONFIG['DATA_DIR']
        self.results_dir = CONFIG['MODEL_RESULTS_DIR']
        self.model_list_for_filter = CONFIG['MODELS_LIST_FOR_FILTER']
        self.true_false_idx = CONFIG['TRUE_FALSE_BOOL_IDX_IN_RESULT']
        self.mcq_start_idx, self.mcq_end_idx = CONFIG['MCQ_PROB_START_IDX'], CONFIG['MCQ_PROB_END_IDX']
        self.easy_filter_threshold, self.hard_filter_threshold = CONFIG['EASY_FILTER_THRESHOLD'], CONFIG['HARD_FILTER_THRESHOLD'] 
        random.seed(10)

    def filter_easy(self, dataset_map):

        subset_removal_easy, subset_removal_hard = {}, {}
        global_total, global_correct_high_probs, global_incorrect_high_probs = 0,0,0

        for category in sorted(os.listdir(self.data_dir)) :
            ## Reading each results file
            results_file = category.replace('_test','')

            len_category = len(dataset_map[category]) 
            ## Storing number of correct and incorrect samples for all tested models
            count_correct = [0] * len_category
            count_correct_high_probs = [0] * len_category
            count_incorrect = [0] * len_category
            count_incorrect_high_probs = [0] * len_category

            for model in self.model_list_for_filter: 
                results_model = read_csv(os.path.join(self.results_dir, model, results_file))[1:]

                for i in range(len(results_model)):
                    if results_model[i][self.true_false_idx].upper() == 'TRUE' : 
                        #checking probs for true cases
                        count_correct[i] += 1

                        prob_correct = get_answer_prob(results_model[i], self.CONFIG) 

                        if float(prob_correct) > self.easy_filter_threshold : 
                            count_correct_high_probs[i] += 1
                    else :
                        ##Checking all false cases for hard sample filtering 
                        count_incorrect[i] += 1
                        probs = results_model[i][self.mcq_start_idx:self.mcq_end_idx+1]
                    
                        prob_incorrect = max([float(num) for num in probs])

                        #if float(prob_incorrect) > hard_filter_threshold : 
                        #    count_incorrect_high_probs[i] += 1

            ## Indexes for examples with all models correct/incorrect high prob answers
            subset_removal_easy[category] = [index for index, value in enumerate(count_correct_high_probs) if value == len(self.model_list_for_filter)] 
            #subset_removal_hard[category] =  [index for index, value in enumerate(count_incorrect_high_probs) if value == len(self.model_list_for_filter)]
            
        #print ('Length of original dataset: ', sum(map(len, dataset_map.values())))
        for category in dataset_map: 
            ## Removing easy questions from mmlu
            ## Keeping 10% of easy questions
            dataset_map[category] = [idx for idx in dataset_map[category] if idx not in subset_removal_easy[category] or (idx in subset_removal_easy[category] and random.random() < 0.1)]
            #dataset_map[category] = [idx for idx in dataset_map[category] if idx not in subset_removal_easy[category]]

        ##TODO Mark hard examples with a seperate tag. Not removing hard examples as some of them might be correct labels with model getting them wrong
        #for category in dataset_map : 
        #    ## Removing hard questions from mmlu
        #    dataset_map[category] = [idx for idx in dataset_map[category] if idx not in subset_removal_hard[category]]
        #print ('Length of dataset after hard filtering: ', sum(map(len, dataset_map.values())))

        return dataset_map
