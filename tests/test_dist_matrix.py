"""Tests for distance matrix computation."""

import numpy as np
import scipy
from sklearn.datasets import make_blobs

from diameter.dist_matrix import compute_dist_matrix, compute_sparse_dist_matrix


X, y = make_blobs(n_samples=100, n_features=50, random_state=42)


def test_dist_matrix():

    dist_matrix = compute_dist_matrix(X)
    assert np.all(np.isfinite(dist_matrix))

    dist_matrix = compute_dist_matrix(X, metric='inner_product')
    assert np.all(np.isfinite(dist_matrix))

    dist_matrix = compute_dist_matrix(X, fill_diagonal=True)
    assert np.all(np.diagonal(dist_matrix) == np.inf)

    dist_matrix = compute_dist_matrix(X[0])


def test_sparse_dist_matrix():

    dist_matrix = compute_sparse_dist_matrix(X, metric='cosine', max_distance=0.5)
    assert isinstance(dist_matrix, scipy.sparse.csr_matrix)
