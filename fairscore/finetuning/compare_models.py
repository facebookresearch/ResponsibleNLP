#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
# from fairseq.models.roberta import RobertaModel
from transformers import AutoModel

# PATH_1 = "/checkpoint/rebeccaqian/fairscore/fairberta/huggingface/roberta_base_16gb_ep105_masked_lm_pretrained_from_fairseq/"
# PATH_2 = "/checkpoint/ccross/fairscore/roberta_bookwiki_masked_lm_pretrained_rebecca_5-12"

PATH_1 = "/checkpoint/rebeccaqian/fairscore/fairberta/huggingface/fairberta_base_16gb_ep80_masked_lm_pretrained_from_fairseq/"
PATH_2 = "/checkpoint/ccross/fairscore/fairberta_16GB_last_chkpt_masked_lm_pretrained_from_fairseq"
model_1 = AutoModel.from_pretrained(PATH_1)
model_2 = AutoModel.from_pretrained(PATH_2)

# model = AutoModel.from_pretrained("/checkpoint/ccross/fairscore/roberta_16GB_last_chkpt_masked_lm_pretrained_from_fairseq/")

# model.pooler.dense.requires_grad_(False)
# print(model.pooler.dense.weight)
# model.embeddings.token_type_embeddings.requires_grad_(False)
# print(model.embeddings.token_type_embeddings.weight)

sd1 = model_1.state_dict()
sd2 = model_2.state_dict()

for key in sd1:
    w1 = sd1[key]
    w2 = sd2[key]
    if False in torch.eq(w1, w2):
        print(key)
        # print(w1)
        # print(w2)

import ipdb; ipdb.set_trace()