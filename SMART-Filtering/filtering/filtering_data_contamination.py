import os, pdb
from utils import *


class FilterDataContClass:
    def __init__(self, CONFIG):
        self.CONFIG = CONFIG
        self.data_dir = CONFIG['DATA_DIR']
        self.no_ques_dir = CONFIG['NO_QUES_RESULTS_DIR']
        self.no_ques_models_for_filter = CONFIG['NO_QUES_MODELS_FOR_FILTER'] 
        self.true_false_idx = CONFIG['TRUE_FALSE_BOOL_IDX_IN_RESULT']
        self.threshold_nota = CONFIG['NOTA_THRESHOLD']


    def filter_data_contamination(self,dataset_map):
        no_ques_removal = {}
        global_no_ques_high_probs = 0
        for category in sorted(os.listdir(self.data_dir)) :
            ## Reading each results file
            results_file = category.replace('_test','')
            ## Reading original dataset to get dataset size. This step can be replaced by reading one of the results category for faster inference
            #dataset = read_csv(os.path.join(data_dir, category))
            len_category = len(dataset_map[category]) 
            prob_correct_high_probs = [0] * len_category

            for model in self.no_ques_models_for_filter : 
                no_ques_rows = read_csv(os.path.join(self.no_ques_dir, model, results_file))[1:]

                for i in range(len(no_ques_rows)) :
                    if no_ques_rows[i][self.true_false_idx].upper() == 'TRUE' : 
                        probs_nota = get_answer_prob(no_ques_rows[i], self.CONFIG)

                        if probs_nota > self.threshold_nota:
                            prob_correct_high_probs[i] += 1
            no_ques_removal[category] = [index for index, value in enumerate(prob_correct_high_probs) if value == len(self.no_ques_models_for_filter)]
            global_no_ques_high_probs += len(no_ques_removal[category])

        for category in dataset_map : 
            ## Removing no ques high probs from mmlu
            dataset_map[category] = [idx for idx in dataset_map[category] if idx not in no_ques_removal[category]]

        return dataset_map