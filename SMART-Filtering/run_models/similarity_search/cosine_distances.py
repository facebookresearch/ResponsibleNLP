# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the CC BY NC license found in the
# LICENSE-CC-BY-NC file in the project directory.

import sys, pdb, os
import h5py

sys.path.append('../../filtering')
from utils import *
import argparse 
from manifold_metrics import *
parser = argparse.ArgumentParser()
parser.add_argument('--knns', default=100, help='no of knns')
parser.add_argument('--embedding_dir', type=str)
parser.add_argument('--out_dir', type=str)

args = parser.parse_args()

embeddings_dir = args.embedding_dir

out_dir = args.out_dir
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

out_file = os.path.join(out_dir, 'cosine_' + embeddings_dir.split('/')[-1] + '_knn=' + str(args.knns) + '.h5')


dataset_seq = []
all_embeds = []

for file in sorted(os.listdir(embeddings_dir)) : 
    category = file.split('.npy')[0]
    dataset_seq.append(category)
    if 'sbert' in embeddings_dir  or 'LLM2Vec' in embeddings_dir : 
        #sbert embedings are different from LLM embeddings. Have 2 dimensions in place of 3
        if not isinstance(all_embeds, np.ndarray):
            all_embeds = np.load(os.path.join(embeddings_dir,'{}.npy'.format(category)))
        else :
            all_embeds = np.concatenate((all_embeds, np.load(os.path.join(embeddings_dir,'{}.npy'.format(category)))), axis=0)
    else :
        if not isinstance(all_embeds, np.ndarray):
            all_embeds = np.load(os.path.join(embeddings_dir,'{}.npy'.format(category)))[:,0,:]
        else :
            all_embeds = np.concatenate((all_embeds, np.load(os.path.join(embeddings_dir,'{}.npy'.format(category)))[:,0,:]), axis=0)

cosine_similarity = compute_pairwise_distance(all_embeds)

if args.knns == 'all':
    ##This is for cosine distances on entire dataset -- computationally expensive as O(n^2)
    closest_indices = np.argsort(cosine_similarity, axis=-1)[:,1:]
else : 
    closest_indices = np.argsort(cosine_similarity, axis=-1)[:,1:int(args.knns)+1]
closest_values = cosine_similarity[np.arange(cosine_similarity.shape[0])[:, None], closest_indices]

save_dict = {'dataset_seq': dataset_seq, 'closest_indices': closest_indices,'closest_values': closest_values}

with h5py.File(out_file, 'w') as f:
    f.create_group('group')
    f['group'].create_dataset('dataset_seq', data=save_dict['dataset_seq'])
    f['group'].create_dataset('closest_indices', data=save_dict['closest_indices'])
    f['group'].create_dataset('closest_values', data=save_dict['closest_values'])

