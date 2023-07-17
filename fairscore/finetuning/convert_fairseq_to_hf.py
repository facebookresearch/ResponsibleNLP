#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# see /checkpoint/rebeccaqian/fairscore/roberta/2022-04-25/expt_256_concat.model.pt for a usable state dict
# roberta from scratch: /checkpoint/rebeccaqian/fairscore/roberta/2022-04-25/512_orig.model.pt
# fairberta: /checkpoint/rebeccaqian/fairscore/roberta/2022-04-25/expt_256_concat.model.pt

# -- for RoBERTa mini -- 
# python convert_fairseq_to_hf.py  \
#     --fairseq_model_path /checkpoint/rebeccaqian/fairscore/roberta/2022-04-25/512_orig.model.pt \
#     --hf_model_path_out_masked_lm /checkpoint/ccross/fairscore/roberta_mini_masked_lm_pretrained_from_fairseq \
#     --hf_model_path_out_seq_class /checkpoint/ccross/fairscore/roberta_mini_seq_class_pretrained_from_fairseq

# -- for FairBERTa mini --
# python convert_fairseq_to_hf.py  \
#     --fairseq_model_path /checkpoint/rebeccaqian/fairscore/roberta/2022-04-25/expt_256_concat.model.pt \
#     --hf_model_path_out_masked_lm /checkpoint/ccross/fairscore/fairberta_mini_masked_lm_pretrained_from_fairseq \
#     --hf_model_path_out_seq_class /checkpoint/ccross/fairscore/fairberta_mini_seq_class_pretrained_from_fairseq

# -- for FairBERTa bookwiki from koustuv --
# python convert_fairseq_to_hf.py  \
#     --fairseq_model_path /checkpoint/rebeccaqian/fairscore/fairberta/2022-05-06/fairberta_bookwiki.model.pt \
#     --hf_model_path_out_masked_lm /checkpoint/ccross/fairscore/fairberta_bookwiki_masked_lm_pretrained_from_fairseq \
#     --hf_model_path_out_seq_class /checkpoint/ccross/fairscore/fairberta_bookwiki_seq_classifier_pretrained_from_fairseq

# -- for FairBERTa bookwiki from Rebecca --
# python convert_fairseq_to_hf.py  \
#     --fairseq_model_path /checkpoint/rebeccaqian/fairscore/roberta/2022-05-12/roberta_base_bookwiki.model.pt \
#     --hf_model_path_out_masked_lm /checkpoint/ccross/fairscore/fairberta_bookwiki_masked_lm_pretrained_rebecca_5-12 \
#     --hf_model_path_out_seq_class /checkpoint/ccross/fairscore/fairberta_bookwiki_seq_classifier_pretrained_rebecca_5-12


# -- for RoBERTa bookwiki --
# python convert_fairseq_to_hf.py  \
#     --fairseq_model_path /checkpoint/ccross/fairscore/roberta_bookwiki_koustuv/model.pt \
#     --hf_model_path_out_masked_lm /checkpoint/ccross/fairscore/roberta_bookwiki_masked_lm_pretrained_from_fairseq \
#     --hf_model_path_out_seq_class /checkpoint/ccross/fairscore/roberta_bookwiki_seq_classifier_pretrained_from_fairseq

# -- for fairBERTa 160GB (last checkpoint) --
# python convert_fairseq_to_hf.py \
#     --fairseq_model_path /checkpoint/adinawilliams/2022-05-26/test_fairberta.faststatsync.me_fp16.roberta_base.cmpltdoc.tps512.adam.b2_0.98.eps1e-06.cl0.0.lr0.0006.wu24000.dr0.1.atdr0.1.wd0.01.ms16.uf2.mu500000.s1.zero2.fsdp.ngpu256/checkpoint_last-shard0.pt \
#     --hf_model_path_out_masked_lm /checkpoint/ccross/fairscore/fairberta_160GB_last_chkpt_masked_lm_pretrained_from_fairseq \
#     --hf_model_path_out_seq_class /checkpoint/ccross/fairscore/fairberta_160GB_last_chkpt_seq_classifier_pretrained_from_fairseq

# python convert_fairseq_to_hf.py \
#     --fairseq_model_path /checkpoint/adinawilliams/2022-04-30/bookwiki_rob_base.faststatsync.me_fp16.roberta_base.cmpltdoc.tps512.adam.b2_0.98.eps1e-06.cl0.0.lr0.0006.wu24000.dr0.1.atdr0.1.wd0.01.ms16.uf8.mu100000.s1.zero2.fsdp.ngpu64/checkpoint_last-shard0.pt \
#     --hf_model_path_out_masked_lm /checkpoint/ccross/fairscore/fairberta_16GB_last_chkpt_masked_lm_pretrained_from_fairseq \
#     --hf_model_path_out_seq_class /checkpoint/ccross/fairscore/fairberta_16GB_last_chkpt_seq_classifier_pretrained_from_fairseq

# python convert_fairseq_to_hf.py \
#     --fairseq_model_path /checkpoint/adinawilliams/2022-05-26/test_fairberta.faststatsync.me_fp16.roberta_base.cmpltdoc.tps512.adam.b2_0.98.eps1e-06.cl0.0.lr0.0006.wu24000.dr0.1.atdr0.1.wd0.01.ms16.uf2.mu500000.s1.zero2.fsdp.ngpu256/checkpoint_best-shard0.pt \
#     --hf_model_path_out_masked_lm /checkpoint/ccross/fairscore/fairberta_16GB_best_chkpt_masked_lm_pretrained_from_fairseq \
#     --hf_model_path_out_seq_class /checkpoint/ccross/fairscore/fairberta_16GB_best_chkpt_seq_classifier_pretrained_from_fairseq

# python convert_fairseq_to_hf.py \
#     --fairseq_model_path /checkpoint/mkambadur/2022-05-10/roberta_bookwiki.faststatsync.me_fp16.roberta_base.cmpltdoc.tps512.adam.b2_0.98.eps1e-06.cl0.0.lr0.0006.wu24000.dr0.1.atdr0.1.wd0.01.ms16.uf4.mu100000.s1.zero2.fsdp.ngpu128/checkpoint_last-shard0.pt \
#     --hf_model_path_out_masked_lm /checkpoint/ccross/fairscore/roberta_16GB_last_chkpt_masked_lm_pretrained_from_fairseq \
#     --hf_model_path_out_seq_class /checkpoint/ccross/fairscore/roberta_16GB_last_chkpt_seq_classifier_pretrained_from_fairseq



import argparse
import re
import torch
from transformers import (
    RobertaConfig, RobertaTokenizer, RobertaForMaskedLM, RobertaForSequenceClassification
)

# ignored the classifier/lm heads since we're finetuning from pretrained checkpoints only
FAIRSEQ2HF_KEYS_SEQ_CLASSIFICATION = [
    ("model.encoder.sentence_encoder", "roberta.encoder"),
    ("encoder.sentence_encoder", "roberta.encoder"), # may have a fairseq version shift that uses this instead
    (".layers.", ".layer."),
    ("embed_tokens", "embeddings.word_embeddings"),
    ("embed_positions", "embeddings.position_embeddings"),
    ("emb_layer_norm", "embeddings.LayerNorm"),
    ("layernorm_embedding", "embeddings.LayerNorm"),
    ("roberta.encoder.embeddings", "roberta.embeddings"),
    ("self_attn.k_proj", "attention.self.key"),
    ("self_attn.q_proj", "attention.self.query"),
    ("self_attn.v_proj", "attention.self.value"),
    ("self_attn_layer_norm", "attention.output.LayerNorm"),
    ("self_attn.out_proj", "attention.output.dense"),
    ("fc1", "intermediate.dense"),
    ("fc2", "output.dense"),
    ("final_layer_norm", "output.LayerNorm"),
]

FAIRSEQ2HF_KEYS_MASKED_LM = FAIRSEQ2HF_KEYS_SEQ_CLASSIFICATION.copy()
FAIRSEQ2HF_KEYS_MASKED_LM.extend([
    ("model.encoder.lm_head.weight", "lm_head.decoder.weight"),
    ("model.encoder.lm_head.bias", "lm_head.decoder.bias"),
    ("model.encoder.lm_head.dense", "lm_head.dense"),
    ("encoder.lm_head.weight", "lm_head.decoder.weight"),
    ("encoder.lm_head.bias", "lm_head.decoder.bias"),
    ("encoder.lm_head.dense", "lm_head.dense"),
    # ("model.encoder.lm_head.dense", "lm_head.dense"),
    # ("model.encoder.sentence_encoder.lm_head.layer_norm", "lm_head.layer_norm"),
    # ("model.encoder.sentence_encoder.lm_head.weight", "lm_head.decoder.weight"),
    # ("model.encoder.sentence_encoder.lm_head.bias", "lm_head.decoder.bias"),
    # ("model.encoder.sentence_encoder.lm_head.dense", "lm_head.dense"),
    # ("model.encoder.sentence_encoder.lm_head.layer_norm", "lm_head.layer_norm"),
])

def save_huggingface_model(model_name_or_path, state_dict_from_fairseq, output_path, is_sequence_classifier):
    print(model_name_or_path, len(state_dict_from_fairseq), output_path, is_sequence_classifier)
    config = RobertaConfig.from_pretrained(model_name_or_path)
    
    if is_sequence_classifier:
        model = RobertaForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=None,
            config=config,
            state_dict=state_dict_from_fairseq
        )

        # skip classifier keys in saving model
        classifier_keys = ["classifier.dense.weight", "classifier.dense.bias", "classifier.out_proj.weight", "classifier.out_proj.bias"]
        if model._keys_to_ignore_on_save is not None:
            model._keys_to_ignore_on_save.extend(classifier_keys)
        else:
            model._keys_to_ignore_on_save = classifier_keys
    else:
        model = RobertaForMaskedLM.from_pretrained(
            pretrained_model_name_or_path=None,
            config=config,
            state_dict=state_dict_from_fairseq
        )
    
    tokenizer = RobertaTokenizer.from_pretrained(model_name_or_path)
    tokenizer.save_pretrained(output_path)
    model.save_pretrained(output_path)
    print(f"Saved model to {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fairseq_model_path", default="roberta.base/model.pt", type=str)
    parser.add_argument("--hf_model_name_or_path_seq_class", default="roberta-base", type=str)
    parser.add_argument("--hf_model_name_or_path_masked_lm", default="roberta-base", type=str)
    parser.add_argument("--hf_model_path_out_masked_lm", type=str, required=True, help="path to save HuggingFace masked LM model")
    parser.add_argument("--hf_model_path_out_seq_class", type=str, required=True, help="path to save HuggingFace sequence classification model")
    args = parser.parse_args()

    # 1. load fairseq state dict
    state_dict_fairseq = torch.load(args.fairseq_model_path)
    if "model" in state_dict_fairseq and "cfg" in state_dict_fairseq:
        state_dict_fairseq = state_dict_fairseq["model"]

    # 2. map for masked LM model
    print("Mapping for masked LM model")
    state_dict_huggingface_masked_lm = RobertaForMaskedLM.from_pretrained(args.hf_model_name_or_path_seq_class).state_dict()
    
    for key_fairseq, key_hf in FAIRSEQ2HF_KEYS_MASKED_LM:
        state_dict_fairseq_remapped_masked_lm = {}
        for k,v in state_dict_fairseq.items():
            remapped_key = re.sub(key_fairseq, key_hf, k)
            state_dict_fairseq_remapped_masked_lm[remapped_key] = v
        state_dict_fairseq = state_dict_fairseq_remapped_masked_lm
        
    # these are keys in the fairseq state dict that are not used by the HF model
    unused_keys_from_fairseq = [k for k in state_dict_fairseq if k not in state_dict_huggingface_masked_lm]
    print(f"\n\tUnused keys from fairseq: {unused_keys_from_fairseq}")
    
    # these are keys in the HF state dict that were not found in the fairseq state dict
    missing_keys_for_hf = [k for k in state_dict_huggingface_masked_lm if k not in state_dict_fairseq]
    print(f"\n\tMissing {len(missing_keys_for_hf)} keys for HuggingFace model: {missing_keys_for_hf}")

    save_huggingface_model(
        model_name_or_path=args.hf_model_name_or_path_masked_lm,
        state_dict_from_fairseq=state_dict_fairseq_remapped_masked_lm,
        output_path=args.hf_model_path_out_masked_lm,
        is_sequence_classifier=False
        )

    # 3. next map for sequence classification model
    print("\n\n\nMapping for seq classifier model")
    state_dict_huggingface_seq_class = RobertaForSequenceClassification.from_pretrained(args.hf_model_name_or_path_seq_class).state_dict()
    
    for key_fairseq, key_hf in FAIRSEQ2HF_KEYS_SEQ_CLASSIFICATION:
        state_dict_fairseq_remapped_seq_classification = {}
        for k,v in state_dict_fairseq.items():
            remapped_key = re.sub(key_fairseq, key_hf, k)
            state_dict_fairseq_remapped_seq_classification[remapped_key] = v
        state_dict_fairseq = state_dict_fairseq_remapped_seq_classification
        
    # these are keys in the fairseq state dict that are not used by the HF model
    unused_keys_from_fairseq = [k for k in state_dict_fairseq if k not in state_dict_huggingface_seq_class]
    print(f"\n\tUnused keys from fairseq: {unused_keys_from_fairseq}")
    
    # these are keys in the HF state dict that were not found in the fairseq state dict
    missing_keys_for_hf = [k for k in state_dict_huggingface_seq_class if k not in state_dict_fairseq]
    print(f"\n\tMissing {len(missing_keys_for_hf)} keys for HuggingFace model: {missing_keys_for_hf}")
    
    save_huggingface_model(
        model_name_or_path=args.hf_model_name_or_path_seq_class,
        state_dict_from_fairseq=state_dict_fairseq_remapped_seq_classification,
        output_path=args.hf_model_path_out_seq_class,
        is_sequence_classifier=True
        )
    
if __name__ == "__main__":
    main()