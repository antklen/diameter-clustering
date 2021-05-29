"""
Computation of distance matrix.
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import RadiusNeighborsTransformer


def compute_dist_matrix(X, metric='inner_product', fill_diagonal=False):
    """
    Compute distance matrix between points and optionally fill diagonal elements
    with np.inf (may be convenient in some situation).

    Args:
        X (np.array): 2D array with data points.
        metric (str): Distance metric for scipy.spatial.distance.pdist or 'inner_product'.
            If 'inner_product' then use np.inner instead of pdist which is much faster.
            np.inner could be used instead of cosine distance for normalized vectors.
        fill_diagonal (bool): If True then fill diagonal with np.inf.
    """

    if X.ndim == 1:
        X = X[None, :]  # for correct work of distance computation

    if metric == 'inner_product':
        dist = 1 - np.inner(X, X)
    else:
        dist = pdist(X, metric=metric)
        # squareform converts emmpty dist array([]) to array([[0.]])
        # this behavior could break the code later
        dist = squareform(dist) if len(dist) > 0 else np.empty((0, 0))

    if fill_diagonal:
        np.fill_diagonal(dist, np.inf)

    return dist


def compute_sparse_dist_matrix(X, metric='cosine', max_distance=0.2):
    """
    Compute distance matrix in sparse csr format using sklearn RadiusNeighborsTransformer.
    Non-zero elements of matrix are elements for which distance is less than max_distance.

    Args:
        X (np.array): 2D array with data points.
        metric (str): Distance metric.
        max_distance (float): Maximum distance threshold.
    """

    transformer = RadiusNeighborsTransformer(mode='distance', algorithm='brute',
                                             metric=metric, radius=max_distance)

    return transformer.fit_transform(X)
