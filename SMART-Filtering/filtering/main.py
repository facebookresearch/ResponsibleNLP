
import os, pdb
import importlib
from utils import *
import argparse, sys
from filtering_easy import FilterEasyClass 
from filtering_data_contamination import FilterDataContClass
from pre_filtering import PreFilteringClass
from filtering_similar_examples import *
import copy
from scipy import stats
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='arc', help='name of the dataset which has corresponding config_{dataset}.py file')
parser.add_argument('--category_wise', type=bool, default=False, help='get results for each category for each step of the filtering')
parser.add_argument('--accuracy', type=bool, default=False, help='get accuracy on original and smart filtering dataset for all models')


def main():
    args = parser.parse_args()

    config_module_name = f'configs.config_{args.dataset}'
    config_module = importlib.import_module(config_module_name)
    CONFIG = config_module.CONFIG

    #Initializing class for each step
    filter_easy_class = FilterEasyClass(CONFIG)
    filter_data_cont_class = FilterDataContClass(CONFIG)
    pre_filtering_class = PreFilteringClass(CONFIG)
    filter_similar_class = FilterSimilarClass(CONFIG)
    data_dir = CONFIG['DATA_DIR']

    if not os.path.exists(data_dir) :
        sys.exit('Path issue in data folder. Change the pointer to desired dataset')

    def create_dataset_mapping(): 
        dataset_mapping = {}
        for category in sorted(os.listdir(data_dir)) :
            dataset = read_csv(os.path.join(data_dir, category))

            ## Creating indexing for subset filtering
            dataset_mapping[category] = [idx for idx in range(len(dataset))]
        return dataset_mapping


    #Creating entire dataset mapping
    dataset_mapping = create_dataset_mapping()
    print ('Length of original dataset: ', sum(map(len, dataset_mapping.values())))
    if args.category_wise:
        for key in dataset_mapping:
            print (key, len(dataset_mapping[key])) 

    #performing easy filtering
    dataset_after_easy = filter_easy_class.filter_easy(copy.deepcopy(dataset_mapping))
    print ('Length of dataset after easy hard filtering: ', sum(map(len, dataset_after_easy.values())), float((sum(map(len, dataset_mapping.values())) - sum(map(len, dataset_after_easy.values())))/sum(map(len, dataset_mapping.values()))))
    if args.category_wise:
        for key in dataset_after_easy:
            print (key, len(dataset_after_easy[key]), float((len(dataset_mapping[key]) - len(dataset_after_easy[key]))/len(dataset_mapping[key])))

    #performing data contamination filtering
    dataset_after_data_cont = filter_data_cont_class.filter_data_contamination(copy.deepcopy(dataset_mapping))
    print ('Length of dataset after data cont filtering: ', sum(map(len, dataset_after_data_cont.values())), float((sum(map(len, dataset_mapping.values())) - sum(map(len, dataset_after_data_cont.values())))/sum(map(len, dataset_mapping.values()))))
    if args.category_wise:
        for key in dataset_after_data_cont:
            print (key, len(dataset_after_data_cont[key]), float((len(dataset_mapping[key]) - len(dataset_after_data_cont[key]))/len(dataset_mapping[key]))) 

    #removing anomolous subsets
    dataset_after_anomolous = pre_filtering_class.filter_anomolous(copy.deepcopy(dataset_mapping))
    print ('Length of dataset after anomolous subset: ', sum(map(len, dataset_after_anomolous.values())), float((sum(map(len, dataset_mapping.values())) - sum(map(len, dataset_after_anomolous.values())))/sum(map(len, dataset_mapping.values()))))
    if args.category_wise:
        for key in dataset_after_anomolous:
            print (key, len(dataset_after_anomolous[key]), float((len(dataset_mapping[key]) - len(dataset_after_anomolous[key]))/len(dataset_mapping[key])))

    #removing duplicate questions
    dataset_after_exact_matches = pre_filtering_class.find_exact_matches(copy.deepcopy(dataset_mapping))
    print ('Length of dataset after exact matches filtering: ', sum(map(len, dataset_after_exact_matches.values())), float((sum(map(len, dataset_mapping.values())) - sum(map(len, dataset_after_exact_matches.values())))/sum(map(len, dataset_mapping.values()))))
    if args.category_wise:
        for key in dataset_after_exact_matches:
            print (key, len(dataset_after_exact_matches[key]), float((len(dataset_mapping[key]) - len(dataset_after_exact_matches[key]))/len(dataset_mapping[key])))

    #removing similar examples
    dataset_after_cosine_similarity = filter_similar_class.find_cosine_based_similar(copy.deepcopy(dataset_mapping))
    print ('Length of dataset after cosine based filtering: ', sum(map(len, dataset_after_cosine_similarity.values())), float((sum(map(len, dataset_mapping.values())) - sum(map(len, dataset_after_cosine_similarity.values())))/sum(map(len, dataset_mapping.values()))))

    ##Summing exact matches and cosine similarity
    similar_questions_filtered = 2 * sum(map(len, dataset_mapping.values())) - sum(map(len, dataset_after_exact_matches.values())) - sum(map(len, dataset_after_cosine_similarity.values()))

    print ('Length of dataset after filtering similar questions: ', sum(map(len, dataset_mapping.values())) - similar_questions_filtered, float((sum(map(len, dataset_mapping.values())) - similar_questions_filtered)/sum(map(len, dataset_mapping.values()))))
    if args.category_wise:
        for key in dataset_after_cosine_similarity:
            print (key, len(dataset_after_cosine_similarity[key]), float((len(dataset_mapping[key]) - len(dataset_after_cosine_similarity[key]))/len(dataset_mapping[key])))

    # combining all filtering steps
    filtered_dataset = {}
    for key in dataset_mapping:
        filtered_dataset[key] = list(set(dataset_after_easy[key]) & 
                                    set(dataset_after_data_cont[key]) & 
                                    set(dataset_after_anomolous[key]) & 
                                    set(dataset_after_exact_matches[key]) & 
                                    set(dataset_after_cosine_similarity[key]))

    print ('Length of dataset after all filtering: ', sum(map(len, filtered_dataset.values())), float((sum(map(len, dataset_mapping.values())) - sum(map(len, filtered_dataset.values())))/sum(map(len, dataset_mapping.values()))))
    if args.category_wise:
        for key in filtered_dataset:
            print (key, len(dataset_mapping[key]), len(filtered_dataset[key]), float((len(dataset_mapping[key]) - len(filtered_dataset[key]))/len(dataset_mapping[key])))


    if args.accuracy: 
        results_dir = CONFIG['MODEL_RESULTS_DIR']
        model_list = CONFIG['MODEL_LIST_FOR_TESTING']

        ans_idx, true_false_idx = CONFIG['ANS_IDX'], CONFIG['TRUE_FALSE_BOOL_IDX_IN_RESULT']

        ##TODO Fix this to make it dependent on config

        accuracies_org = {} 
        accuracies_smart = {}
        print ('accuracy before and after SMART filtering : ')
        for model in model_list : 
            global_correct, global_correct_subset, global_count, global_count_subset = 0, 0, 0, 0 
            for category in os.listdir(data_dir) :
                results_file = category.replace('_test','')
                org_rows = read_csv(os.path.join(results_dir, model, results_file))[1:]
                filter_indices = filtered_dataset[category]
                correct, correct_subset, total_count, total_count_subset = 0, 0, 0, 0
                for i in range(len(org_rows)) :
                    total_count += 1 
                    if org_rows[i][true_false_idx].upper() == 'TRUE' : 
                        correct += 1
                    if i in filter_indices : 
                        total_count_subset += 1 
                        if org_rows[i][true_false_idx].upper() == 'TRUE' : 
                            correct_subset += 1

                global_correct += correct
                global_count += total_count
                global_correct_subset += correct_subset
                global_count_subset += total_count_subset

                #print (float(round(correct_org / total_count, 3)), float(round(correct_mmlu_exp / total_count, 3)), float(round(correct_shuffled_v2 / total_count, 3)), float(round(correct_v1 / total_count, 3)), float(round(correct_v2 / total_count, 3)), float(round(correct / total_count, 3)))

            print (f'{model} ' + str(float(round(global_correct / global_count, 3))) + ' ' + str(float(round(global_correct_subset / global_count_subset, 3))))

            accuracies_org[model] = float(round(global_correct / global_count, 3))
            accuracies_smart[model] = float(round(global_correct_subset / global_count_subset, 3))

        print (global_count)
        print (global_count_subset)

        print ('---Calculating Kendall Tau correlation between model ranking before and after applying SMART---')

        model_ranking_org = [model_list[i] for i in np.argsort(list(accuracies_org.values()))[::-1][:len(model_list)]]
        model_ranking_smart = [model_list[i] for i in np.argsort(list(accuracies_smart.values()))[::-1][:len(model_list)]]

        ranking_org = []
        ranking_smart = []
        for i in range(len(model_ranking_org)):
            ranking_org.append(i)
            ranking_smart.append(model_ranking_smart.index(model_ranking_org[i]))
        
        res = stats.kendalltau(ranking_org, ranking_smart)
        print ('Tau correlation: ', str(res.statistic))


        print ('---Calculating pearson correlation between ChatBot Arena Scores and model accuracies, before and after applying SMART---')
        ## ELO scores on LM Arena as on Sept 23 2024
        lm_arena_elo_scores = {'results_Llama-2-70b-hf': 1093, 
                            'results_Meta-Llama-3-8B-Instruct': 1152, 
                            'results_Mistral-7B-v0.3': 1072, 
                            'results_Qwen1.5-32B-Chat': 1126, 
                            'results_Yi-1.5-34B-Chat': 1157, 
                            'results_Yi-34B-Chat': 1111, 
                            'results_Meta-Llama-3.1-70B-Instruct': 1248, 
                            'results_Mixtral-8x22B-Instruct-v0.1': 1147, 
                            'results_Qwen2-72B-Instruct': 1187, 
                            'results_dbrx-instruct': 1103, 
                            'results_gemma-2-27b-it': 1217, 
                            'results_gemma-7b-it': 1038, 
                            'results_Meta-Llama-3-70B-Instruct': 1206, 
                            'results_Mixtral-8x7B-Instruct-v0.1': 1114, 
                            'results_Phi-3-medium-4k-instruct': 1123, 
                            'results_gemma-2-9b-it': 1188}

        
        lm_arena_scores = []
        org_scores = []
        smart_scores = []
        models = []
        for model in model_ranking_org:
            if model in lm_arena_elo_scores:
                models.append(model)
                lm_arena_scores.append(lm_arena_elo_scores[model])
                org_scores.append(accuracies_org[model])
                smart_scores.append(accuracies_smart[model])
        
        pearson_correlation_org, p_value = stats.pearsonr(lm_arena_scores,org_scores)
        print('Pearson Correlation between ChatBot Arena and model scores on original dataset: ', str(pearson_correlation_org))
        pearson_correlation_smart, p_value = stats.pearsonr(lm_arena_scores,smart_scores)
        print('Pearson Correlation between ChatBot Arena and model scores on SMART dataset: ', str(pearson_correlation_smart))


if __name__ == '__main__':
    main()
