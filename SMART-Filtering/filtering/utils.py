import csv
import pandas as pd
import json

def save_json(out_path, save_dict, indent=4, encoding='utf-8') :
    with open(out_path, 'w+') as f :
        json.dump(save_dict, f, indent=indent, ensure_ascii=False)


def read_csv(inp_file, encoding='utf-8') :
    rows = []
    with open(inp_file, newline='', encoding=encoding) as csvfile:
        #csvreader = csv.reader(csvfile, quotechar='|', skipinitialspace=True)
        csvreader = csv.reader(csvfile, skipinitialspace=True)
        for row in csvreader:
            rows.append(row)
    return rows

def write_csv(dataset, out_file) :
    with open(out_file, 'w+', encoding='UTF8', newline='') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerows(dataset)

def get_answer_prob(row, CONFIG) : 
    ans_idx = CONFIG['ANS_IDX']
    mcq_start_idx, mcq_end_idx = CONFIG['MCQ_PROB_START_IDX'], CONFIG['MCQ_PROB_END_IDX']

    probs = row[mcq_start_idx:mcq_end_idx+1]
    answer = row[ans_idx]
    #mapping = dict(zip('ABCD', probs))
    mapping = dict(zip('ABCDE', probs))

    prob = mapping[answer]

    return float(prob)

