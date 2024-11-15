# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# prdc
# Copyright (c) 2020-present NAVER Corp.
# MIT license


import numpy as np
import pandas as pd
import sklearn.metrics, pdb


def compute_pairwise_distance(data_x, data_y=None):
    """
    From: https://github.com/clovaai/generative-evaluation-prdc/blob/master/prdc/prdc.py
    Args:
        data_x: numpy.ndarray([N, feature_dim], dtype=np.float32)
        data_y: numpy.ndarray([N, feature_dim], dtype=np.float32)
    Returns:
        numpy.ndarray([N, N], dtype=np.float32) of pairwise distances.
    """
    if data_y is None:
        data_y = data_x
    dists = sklearn.metrics.pairwise_distances(
        data_x, data_y, metric="cosine", n_jobs=8
    )
    return dists


def get_kth_value(unsorted, k, axis=-1):
    """
    From: https://github.com/clovaai/generative-evaluation-prdc/blob/master/prdc/prdc.py
    Args:
        unsorted: numpy.ndarray of any dimensionality.
        k: int
    Returns:
        kth values along the designated axis.
    """
    indices = np.argpartition(unsorted, k, axis=axis)[..., :k]
    k_smallests = np.take_along_axis(unsorted, indices, axis=axis)
    kth_values = k_smallests.max(axis=axis)
    pdb.set_trace()
    return kth_values


def compute_nearest_neighbour_distances(input_features, nearest_k):
    """
    From: https://github.com/clovaai/generative-evaluation-prdc/blob/master/prdc/prdc.py
    Args:
        input_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        nearest_k: int
    Returns:
        Distances to kth nearest neighbours.
    """
    distances = compute_pairwise_distance(input_features)
    radii = get_kth_value(distances, k=nearest_k + 1, axis=-1)
    return radii
