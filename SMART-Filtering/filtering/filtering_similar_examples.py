import os, pdb
from utils import *
import argparse
import random, h5py
from scipy.stats import gaussian_kde
import numpy as np
from difflib import SequenceMatcher
import matplotlib.pyplot as plt


class FilterSimilarClass:
    def __init__(self, CONFIG):
        self.data_dir = CONFIG['DATA_DIR']
        self.question_idx = CONFIG['QUESTION_IDX']
        self.cluster_dir = CONFIG['CLUSTER_DIR']
        self.cosine_distances_list = CONFIG['COSINE_DISTANCES']
        self.knns = CONFIG['COSINE_KNNS']
        random.seed(10)


    def calculate_similarity(self, a, b):
        """Calculate the similarity between two strings"""
        return SequenceMatcher(None, a, b).ratio()

    def remove_symbolic_differences(self, combine_list, questions_cluster, threshold=0.8):
        """Remove similar questions from combine_list above 90% exact similarity"""
        # Get the first question
        first_question = questions_cluster[0]
        
        # Initialize an empty list to store the indices to be removed
        indices_to_remove = []
        
        # Iterate over the questions starting from the second question
        for i in range(1, len(questions_cluster)):
            # Calculate the similarity between the current question and the first question
            similarity = self.calculate_similarity(first_question, questions_cluster[i])
            
            # If the similarity is greater than the threshold, add the index to the list
            if similarity > threshold:
                indices_to_remove.append(i)

        # Remove the indices from combine_list in reverse order to avoid index shifting
        for i in sorted(indices_to_remove, reverse=True):
            del combine_list[i]
        
        return combine_list



    ##Examples based on cosine based similarity
    def find_cosine_based_similar(self, dataset_map):
        #pdb.set_trace()
        mapping_dataset_cosine_idx = {}
        idx=0

        for category in sorted(os.listdir(self.data_dir)) :
            dataset = read_csv(os.path.join(self.data_dir, category))
            
            ## Reading dataset 
            for i in range(len(dataset)) : 
                ## Get category and idx mapping for each question
                temp = {}
                temp['category'] = category
                temp['idx_in_category'] = i
                temp['question'] = dataset[i][self.question_idx]

                mapping_dataset_cosine_idx[idx] = temp
                idx+=1

        cluster1 = self.get_clusters(os.path.join(self.cluster_dir, self.cosine_distances_list[0]))

        
        seen_data_points=[]
        
        for key in cluster1:
            if len(cluster1[key]) > 0:
                combine_list = [key] + cluster1[key]
                questions_cluster = []

                ## Checking if the clustered questions differ only in one or two positions. For ex, a difference of an operator in mathematical question changes meaning completely
                for idx in combine_list:
                    questions_cluster += [mapping_dataset_cosine_idx[idx]['question']]

                combine_list = self.remove_symbolic_differences(combine_list, questions_cluster)

                ## Removing elements from cluster which are already seen
                combine_list = [element for element in combine_list if element not in seen_data_points]

                if len(combine_list) <2 : 
                    continue

                #pdb.set_trace()

                ##Randomly select half elements from the cluster to remove from the dataset
                random_elements = random.sample(combine_list, (len(combine_list) // 2))

                elements_to_remove = [element for element in combine_list if element in random_elements]
                for element in elements_to_remove:
                    category, idx = mapping_dataset_cosine_idx[element]['category'], mapping_dataset_cosine_idx[element]['idx_in_category'] 
                    if idx in dataset_map[category] : 
                        dataset_map[category].remove(idx)
                seen_data_points += combine_list

        return dataset_map


    def get_clusters(self, dist_file): 
        with h5py.File(dist_file, 'r') as f:
            # Access the data in the file
            group = f['group']
            list_data = group['dataset_seq'][:]
            if self.knns == all: 
                array1_data = group['closest_indices'][:,:]
                array2_data = group['closest_values'][:,:]
            else:
                array1_data = group['closest_indices'][:, :self.knns]
                array2_data = group['closest_values'][:, :self.knns]

        mapping_dataset_cosine_idx = {}
        idx=0

        for category in sorted(os.listdir(self.data_dir)) :
            dataset = read_csv(os.path.join(self.data_dir, category))
            
            ## Reading dataset 
            for i in range(len(dataset)) : 
                ## Get category and idx mapping for each question
                temp = {}
                temp['category'] = category
                temp['idx_in_category'] = i
                temp['question'] = dataset[i][self.question_idx]

                mapping_dataset_cosine_idx[idx] = temp
                idx+=1


        new_cosine_data = []
        question_tuples = []
        for q_idx in range(len(array1_data)):
            for match_idx in range(len(array1_data[q_idx])):
                score = self.calculate_similarity(mapping_dataset_cosine_idx[q_idx]['question'], mapping_dataset_cosine_idx[array1_data[q_idx][match_idx]]['question'])
                if score > 0.8:
                    continue
                else:
                    new_cosine_data.append(array2_data[q_idx][match_idx:])
                    break
                

        new_cosine_data = np.concatenate(new_cosine_data)

        ## Finding the right threshold

        # Estimate the kernel density of the data
        kde = gaussian_kde(new_cosine_data, bw_method=0.15)
        # Evaluate the KDE on a grid of points
        x_grid = np.linspace(min(new_cosine_data), max(new_cosine_data), 1000)
        y_grid = kde.evaluate(x_grid)
        # Find the local maxima in the KDE (these correspond to the thresholds between Gaussians)
        local_maxima_idx = np.where(np.diff(np.sign(np.diff(y_grid))) < 0)[0]
        thresholds = x_grid[local_maxima_idx]

        # Check the number of detected clusters based on the number of local maxima
        if len(thresholds) == 1:
            # If there is only one cluster, half the threshold value
            # This adjustment is for the datasets for which a single peak represents a central cluster
            threshold = thresholds[0] / 2
        else:
            # If multiple clusters are detected, use the first threshold as is
            threshold = thresholds[0]

        ##Code to plot data distribution for the entire dataset
        #out_dir = 'figures/'
        #if not os.path.exists(out_dir):
        #    os.mkdir(out_dir)

        #print ('Saving plots in: ', out_dir)

        #plt.hist(new_cosine_data, bins=400, alpha=0.7, color='#6495ED', edgecolor='#6495ED')

        #plt.axvline(x=threshold, color='black', linestyle='--', linewidth=1)
        #plt.text(threshold + (max(new_cosine_data) - min(new_cosine_data)) * 0.01, plt.ylim()[1] * 0.8, 'Threshold', fontsize=6)

        ##plt.title(f'Data Distribution for {self.knns} nearest neighbors for MMLU')
        #plt.title(f'Data Distribution for entire MMLU dataset')
        #plt.xlabel('Cosine Distance')
        #plt.ylabel('Frequency')

        #plt.savefig(os.path.join(out_dir, f'llm2vec_arc_knn={self.knns}.jpg'), bbox_inches='tight', dpi=300)

        similar_points = {}

        for i, row in enumerate(array2_data):
            similar_points[i] = []
            for j, val in enumerate(array2_data[i]):
                if val > threshold:
                    break
                point = array1_data[i][j]
                similar_points[i].append(point)

        return similar_points