"""Tests for MaxDiameterClustering."""

import numpy as np
from scipy.spatial.distance import pdist
from sklearn.datasets import make_blobs

from diameter_clustering import MaxDiameterClustering
from diameter_clustering.dist_matrix import compute_dist_matrix, compute_sparse_dist_matrix


MAX_DISTANCE = 0.5

X, y = make_blobs(n_samples=100, n_features=50, centers=3,
                  cluster_std=5, random_state=42)


def compute_max_dist(X, labels, metric='cosine'):
    """Compute maximum distance between points inside clusters."""

    max_dist = []

    for cluster in np.unique(labels):
        x_cluster = X[labels == cluster]
        dist = pdist(x_cluster, metric=metric)
        if len(dist) == 0:
            max_dist.append(0)
        else:
            max_dist.append(dist.max())

    return np.max(max_dist)


def test_max_diameter():

    model = MaxDiameterClustering(max_distance=MAX_DISTANCE, criterion='distance',
                                  metric='cosine', sparse_dist=False, use_timer=True)
    labels = model.fit_predict(X)
    assert compute_max_dist(X, labels) < MAX_DISTANCE

    model = MaxDiameterClustering(max_distance=MAX_DISTANCE, criterion='size',
                                  metric='cosine', sparse_dist=False, use_timer=True)
    labels = model.fit_predict(X)
    assert compute_max_dist(X, labels) < MAX_DISTANCE

def test_inner_product():

    x_normalized = X/(np.linalg.norm(X, axis=-1, keepdims=True) + 1e-16)
    model = MaxDiameterClustering(max_distance=MAX_DISTANCE, metric='inner_product',
                                  sparse_dist=False, deterministic=True)
    labels = model.fit_predict(x_normalized)
    assert compute_max_dist(x_normalized, labels) < MAX_DISTANCE

    model2 = MaxDiameterClustering(max_distance=MAX_DISTANCE, metric='cosine',
                                   sparse_dist=False, deterministic=True)
    labels2 = model2.fit_predict(X)
    assert np.array_equal(labels, labels2)


def test_precomputed():

    model = MaxDiameterClustering(max_distance=MAX_DISTANCE, precomputed_dist=True,
                                  sparse_dist=False)
    dist_matrix = compute_dist_matrix(X, metric='cosine')
    labels = model.fit_predict(dist_matrix)
    assert compute_max_dist(X, labels) < MAX_DISTANCE


def test_sparse():

    model = MaxDiameterClustering(max_distance=MAX_DISTANCE, metric='cosine',
                                  sparse_dist=True)
    labels = model.fit_predict(X)
    assert compute_max_dist(X, labels) < MAX_DISTANCE

    model = MaxDiameterClustering(max_distance=MAX_DISTANCE,
                                  sparse_dist=True, precomputed_dist=True)
    dist_matrix = compute_sparse_dist_matrix(X, max_distance=MAX_DISTANCE)
    labels = model.fit_predict(dist_matrix)
    assert compute_max_dist(X, labels) < MAX_DISTANCE


def test_deterministic():

    model = MaxDiameterClustering(max_distance=MAX_DISTANCE, metric='cosine',
                                  deterministic=True)
    labels1 = model.fit_predict(X)
    labels2 = model.fit_predict(X)
    assert np.array_equal(labels1, labels2)


def test_sparse_dense_equivalence():

    model1 = MaxDiameterClustering(max_distance=MAX_DISTANCE, metric='cosine',
                                   sparse_dist=False, deterministic=True)
    labels1 = model1.fit_predict(X)

    model2 = MaxDiameterClustering(max_distance=MAX_DISTANCE, metric='cosine',
                                   sparse_dist=True, deterministic=True)
    labels2 = model2.fit_predict(X)

    assert np.array_equal(labels1, labels2)
