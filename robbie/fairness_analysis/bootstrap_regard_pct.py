# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
from tqdm import tqdm
from scipy.stats import bootstrap


def get_regard_pct(data):
    negative = 0
    for i in range(len(data)):
        raw_line = data[i]
        label = raw_line["pred_meta"]["label"]
        if label == 'neg':
            negative += 1
    return negative/len(data)*100


def main(args):
    target_samples = {}
    full_results_path = args.input
    total_samples_ct = 0
    if args.task == 'regard':
        with open(full_results_path) as f:
            for line in tqdm(f):
                total_samples_ct += 1
                raw_line = json.loads(line.rstrip())
                descriptor = raw_line["pred_meta"]["descriptor"].replace("_", " ")
                bucket = descriptor
                if bucket in target_samples:
                    target_samples[bucket].append(raw_line)
                else:
                    target_samples[bucket] = [raw_line]        
    else:
        with open(full_results_path) as f:
            for line in tqdm(f):
                total_samples_ct += 1
                raw_line = json.loads(line.rstrip())
                axis = raw_line["pred_meta"]["axis"]
                descriptor = raw_line["pred_meta"]["descriptor"].replace("_", " ")
                bucket = (axis, descriptor)
                if bucket in target_samples:
                    target_samples[bucket].append(raw_line)
                else:
                    target_samples[bucket] = [raw_line]
    out_result = []
    for bucket in target_samples.keys():    
        data = (target_samples[bucket], )
        bootstrap_ci = bootstrap(data, get_regard_pct, vectorized=False, 
                                 confidence_level=args.confidence_interval, n_resamples=args.n_resamples,
                                 random_state=1, method='percentile')
        confidence_intervals = bootstrap_ci.confidence_interval
        mean = (confidence_intervals[0]+confidence_intervals[1])/2
        print(str(bucket) + 
              f'''
              : mean = {mean}  confidence intervals = {confidence_intervals}  
              standard error = {bootstrap_ci.standard_error}  
              count pct= {len(target_samples[bucket])/total_samples_ct}  
              count = {len(target_samples[bucket])}
              ''')
        if args.task == 'regard':
            out_result.append({'descriptor': bucket,
                               'count pct': len(target_samples[bucket])/total_samples_ct, 
                               'count': len(target_samples[bucket]),
                               'mean': mean,
                               'C_lower': confidence_intervals[0],
                               'C_upper': confidence_intervals[1],
                               "standard error": bootstrap_ci.standard_error})
        else:
            out_result.append({'axis': bucket[0], 
                               'descriptor': bucket[1],
                               'count pct': len(target_samples[bucket])/total_samples_ct, 
                               'count': len(target_samples[bucket]),
                               'mean': mean, 
                               'C_lower': confidence_intervals[0], 
                               'C_upper': confidence_intervals[1], 
                               "standard error": bootstrap_ci.standard_error})
   
    out_path = full_results_path.replace(
            ".jsonl", "_regard_bootsrap.jsonl"
        )
    with open(out_path, "w") as outfile:
        json.dump(out_result, outfile)   


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--input",
        type=str,
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--confidence_interval",
        type=float,
        default=0.975,
    )
    parser.add_argument(
        "--n_resamples",
        type=int,
        default=9999,
    )
    args = parser.parse_args()
    main(args)