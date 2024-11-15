import sys, pdb, os
import random
from utils import *

class PreFilteringClass:
    def __init__(self, CONFIG):
        self.CONFIG = CONFIG
        self.anomolous_categories = CONFIG['ANOMOLOUS_DATASETS']
        self.data_dir = CONFIG['DATA_DIR']
        self.question_idx = CONFIG['QUESTION_IDX']
        random.seed(10)

    ## removing anomolous subsets from dataset based on list given in config.py
    def filter_anomolous(self, dataset_map):
        for anomolous_category in self.anomolous_categories:
            for category in dataset_map:
                if (anomolous_category in category) or (category in anomolous_category):
                    ## Removing all examples from anomolous categories
                    ## Modify this part if only some examples from a category needs to be removed
                    dataset_map[category] = []
        return dataset_map


    def find_exact_matches(self, dataset_map):
        ##Dict to save question count
        questions_count = {}

        for category in sorted(os.listdir(self.data_dir)) :
            dataset = read_csv(os.path.join(self.data_dir, category))
            
            ## Reading dataset 
            for i in range(len(dataset)) : 
                ## Get category and idx mapping for each question
                temp = {}
                temp['category'] = category
                temp['idx'] = i

                ##Getting question string from dataset
                question = dataset[i][self.question_idx].lower()

                #Checking if question already exists
                if question not in questions_count :  
                    questions_count[question] = []
                
                questions_count[question].append(temp)
                

        for question in questions_count:
            if len(questions_count[question]) > 1 :
                ##Randomly selecting only one question to keep from the same questions
                random_element = random.choice(questions_count[question])
                # Create a list of elements to remove from dataset_map
                elements_to_remove = [(element['category'], element['idx']) for element in questions_count[question] if element != random_element]
                # Remove the elements from dataset_map
                for category, idx in elements_to_remove:
                    dataset_map[category].remove(idx)

                #print(f"Question: {question} \n Idx: {questions_count[question]}")
            
        return dataset_map