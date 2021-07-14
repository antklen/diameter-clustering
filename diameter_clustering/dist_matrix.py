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
        metric (str): Distance metric. Possible options are 'inner_product' or one of metrics
            available in scipy.spatial.distance.pdist. If 'inner_product' then use np.inner
            which is much faster than pdist. 'inner_product' could be used instead of cosine
            distance for normalized vectors.
        fill_diagonal (bool): If True then fill diagonal with np.inf.

    Returns:
        np.array with shape (len(X), len(X)).
    """

    if X.ndim == 1:
        X = X[None, :]  # for correct work of distance computation

    if metric == 'inner_product':
        dist_matrix = 1 - np.inner(X, X)
    else:
        dist_matrix = pdist(X, metric=metric)
        # squareform converts emmpty dist_matrix array([]) to array([[0.]])
        # this behavior could break the code later
        dist_matrix = squareform(dist_matrix) if len(dist_matrix) > 0 else np.empty((0, 0))

    if fill_diagonal:
        np.fill_diagonal(dist_matrix, np.inf)

    return dist_matrix


def compute_sparse_dist_matrix(X, metric='cosine', max_distance=0.2):
    """
    Compute distance matrix in sparse csr format using sklearn RadiusNeighborsTransformer.
    Zero elements of matrix are elements for which distance is greater than max_distance.

    Args:
        X (np.array): 2D array with data points.
        metric (str): Distance metric
            (possible options in sklearn.neighbors.VALID_METRICS['brute']).
        max_distance (float): Maximum distance threshold.

    Returns:
        scipy.sparse.csr_matrix with shape (len(X), len(X)).
    """

    transformer = RadiusNeighborsTransformer(mode='distance', algorithm='brute',
                                             metric=metric, radius=max_distance)

    return transformer.fit_transform(X)
