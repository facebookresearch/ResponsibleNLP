#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import numpy as np
import pandas as pd
from nltk.tokenize import RegexpTokenizer


def process_jigsaw_1(dat_folder):
    """
    Preprocess the Jigsaw Toxic Comment Classification Challenge dataset.
    """

    df_name = "jigsaw-toxic-comment-classification-challenge"

    df_train = pd.read_csv(os.path.join(dat_folder, df_name, "train.csv"))

    df_test = pd.read_csv(os.path.join(dat_folder, df_name, "test.csv"))
    df_test_labels = pd.read_csv(os.path.join(dat_folder, df_name, "test_labels.csv"))
    df_test = df_test.merge(df_test_labels, how='left', on='id')

    # reject the jigsaw items we have no scores for (neither toxic nor beneign)
    df_test = df_test[df_test['toxic'] != -1]

    df_test = df_test.assign(test_set=1)
    df_train = df_train.assign(test_set=0)

    df_jigsaw1 = pd.concat([df_test, df_train], axis=0).reset_index(drop=True).copy()

    # rename df_jigsaw to be consistent with unintended-bias datasets
    df_jigsaw1 = df_jigsaw1.rename(columns={'toxic': 'toxicity',
                                            'severe_toxic': 'severe_toxicity',
                                            'identity_hate': 'identity_attack'})
    df_jigsaw1 = df_jigsaw1.assign(jigsaw_dat=1)
    df_jigsaw1 = df_jigsaw1.assign(identity_grp="predicted")

    return df_jigsaw1


def process_jigsaw_2(dat_folder):
    """
    Preprocess the Jigsaw Unintended Bias in Toxicity Classification dataset.
    """

    df_name = "jigsaw-unintended-bias-in-toxicity-classification"

    df_train = pd.read_csv(os.path.join(dat_folder, df_name, "train.csv"))
    df_train = df_train.rename(columns={'target': 'toxicity'})  # to make train has the same format as test

    toxic_cols = ['toxicity', 'severe_toxicity', 'obscene', 'threat',
                  'insult', 'identity_attack', 'sexual_explicit']
    group_cols = ['male', 'female', 'transgender', 'other_gender', 'heterosexual',
                  'homosexual_gay_or_lesbian', 'bisexual', 'other_sexual_orientation',
                  'christian', 'jewish', 'muslim', 'hindu', 'buddhist', 'atheist',
                  'other_religion', 'black', 'white', 'asian', 'latino',
                  'other_race_or_ethnicity', 'physical_disability',
                  'intellectual_or_learning_disability',
                  'psychiatric_or_mental_illness', 'other_disability']

    df_test_public = pd.read_csv(os.path.join(dat_folder, df_name, "test_public_expanded.csv"))
    df_test_private = pd.read_csv(os.path.join(dat_folder, df_name, "test_private_expanded.csv"))

    df_train = df_train[['id', 'comment_text']+toxic_cols+group_cols]
    df_test_public = df_test_public[['id', 'comment_text']+toxic_cols+group_cols]
    df_test_private = df_test_private[['id', 'comment_text']+toxic_cols+group_cols]

    df_test = pd.concat([df_test_public, df_test_private], axis=0).reset_index(drop=True).copy()

    df_test = df_test.assign(test_set=1)
    df_train = df_train.assign(test_set=0)

    df_jigsaw = pd.concat([df_test, df_train], axis=0).reset_index(drop=True).copy()

    # look at rows with sensitive group labels
    df_has_group = df_jigsaw.dropna(axis=0, subset=group_cols, how="all").reset_index(drop=True).copy()

    # look at rows with/o sensitive group labels (only toxicity labels)
    df_toxic_only = df_jigsaw[df_jigsaw.isnull().any(axis=1)]
    df_toxic_only = df_toxic_only.drop(group_cols, axis=1).reset_index(drop=True).copy()

    assert df_toxic_only.shape[0] + df_has_group.shape[0] == df_jigsaw.shape[0]
    assert df_toxic_only.isna().sum().sum() == 0  # no missing in df_toxic_only
    assert df_has_group.isna().sum().sum() == 0  # no missing in df_has_group

    def binarize(x):
        return np.where(x >= 0.5, 1, 0)

    # axis=0 means apply to columns; axis=1 to rows
    df_toxic_only[toxic_cols] = df_toxic_only[toxic_cols].apply(binarize, axis=0)
    df_has_group[toxic_cols+group_cols] = df_has_group[toxic_cols+group_cols].apply(binarize, axis=0)

    df_toxic_only = df_toxic_only.assign(jigsaw_dat=2)
    df_has_group = df_has_group.assign(jigsaw_dat=2)

    df_toxic_only = df_toxic_only.assign(identity_grp="predicted")
    df_has_group = df_has_group.assign(identity_grp="annotated")

    return df_toxic_only, df_has_group


def process_jigsaw_all(df_jigsaw1, df_toxic_only, df_has_group):
    """
    Preprocess the two Jigsaw datasets.
    """

    df_jigsaw_lst = [df_jigsaw1, df_toxic_only, df_has_group]

    df_lst_processed = list()
    for df in df_jigsaw_lst:
        df = remove_long_rows(df)
        df = augment_row(df)
        df = df[["id", "comment_text", "toxicity", "jigsaw_dat", "test_set", "identity_grp"]].copy()
        df_lst_processed.append(df)

    df_jigsaw = pd.concat(df_lst_processed, axis=0).reset_index(drop=True).copy()
    return df_jigsaw


def tokenized_words(sentence):
    """
    Comment tokenization.
    """

    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(sentence)
    return tokens


def remove_long_rows(df_use):
    """
    Removing comments that are too short, too long, or don't contain any letters.
    """

    df = df_use.copy()

    # limit to only single line items (remove rows with '\n')
    df.drop(df.index[df['comment_text'].str.contains("\n")], axis=0, inplace=True)

    # calculate tokens
    df['length'] = df['comment_text'].apply(lambda x: len(x))  # num of letters
    df['tokens'] = df['comment_text'].apply(tokenized_words)  # get words in list
    df['len_tokens'] = df['tokens'].apply(lambda x: len(x))   # num of words

    # remove rows that contain too many words/sentences
    df = df[df['len_tokens'] > 1]
    df = df[df['len_tokens'] <= 100]
    df = df[df['length'] <= 600]
    # row must contain at least a letter
    df = df.reset_index(drop=True).copy()
    df = df[df['comment_text'].str.contains(r"(?:[A-Za-z]+)")]

    # reset index
    df = df.reset_index(drop=True).copy()
    return df


def augment_row(df_use):
    """
    Augment the comments by breaking them into sentences.
    Each sentence is associated with the original label of the comment.
    Remove sentences that are too short, too long, or don't contain any letters.
    """

    df = df_use.copy()

    # Getting s as pandas series which has split on full stop and new sentence a new line
    s = df["comment_text"].str.split(r'\n|\.|\!|\?|\;').apply(pd.Series, 1).stack()
    s.index = s.index.droplevel(-1)  # to line up with df's index
    s.name = 'comment_text'  # needs a name to join

    # There are blank or emplty cell values after above process. Removing them
    s.replace('', np.nan, inplace=True)
    s.dropna(inplace=True)

    # Joining should ideally get me proper output
    del df['comment_text']
    df = df.join(s)
    df['comment_text'] = df['comment_text'].str.strip()

    # redefine length/tokens/len_tokens
    df['length'] = df['comment_text'].apply(lambda x: len(x))  # num of letters
    df['tokens'] = df['comment_text'].apply(tokenized_words)  # words
    df['len_tokens'] = df['tokens'].apply(lambda x: len(x))   # num of words
    # remove short sentences again
    df = df[df['len_tokens'] > 1]
    df = df[df['len_tokens'] <= 100]
    df = df[df['length'] <= 600]
    # line must contain a letter
    df = df[df['comment_text'].str.contains(r"(?:[A-Za-z]+)", regex=True)]

    # join tokens as new comment_text (all punctuations are removed)
    df['comment_text'] = df['tokens'].str.join(" ").str.lower()

    # remove duplicate comment_text (e.g. "you go girl" appear multiple times)
    df.drop_duplicates(subset=['comment_text'], keep='first', inplace=True)

    # reset index
    df = df.reset_index(drop=True).copy()
    return df
