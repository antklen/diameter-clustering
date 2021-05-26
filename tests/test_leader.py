"""Tests for LeaderClustering."""

import numpy as np
from scipy.spatial.distance import pdist
from sklearn.datasets import make_blobs

from diameter import LeaderClustering
from diameter.dist_matrix import compute_dist_matrix, compute_sparse_dist_matrix


MAX_RADIUS = 0.25
MAX_RADIUS_EUCLIDEAN = 30

X, y = make_blobs(n_samples=100, n_features=50, centers=3,
                  cluster_std=3, random_state=42)


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


def test_leader():

    model = LeaderClustering(max_radius=MAX_RADIUS_EUCLIDEAN, metric='euclidean')
    labels = model.fit_predict(X)
    assert compute_max_dist(X, labels, metric='euclidean') < MAX_RADIUS_EUCLIDEAN * 2

    model = LeaderClustering(max_radius=MAX_RADIUS_EUCLIDEAN, metric='euclidean',
                             change_leaders=True)
    labels = model.fit_predict(X)
    assert compute_max_dist(X, labels, metric='euclidean') < MAX_RADIUS_EUCLIDEAN * 2

    model = LeaderClustering(max_radius=MAX_RADIUS, metric='cosine')
    labels = model.fit_predict(X)
    assert len(labels) == len(X)


def test_inner_product():

    x_normalized = X/(np.linalg.norm(X, axis=-1, keepdims=True) + 1e-16)
    model = LeaderClustering(max_radius=MAX_RADIUS, metric='inner_product',
                             deterministic=True)
    labels = model.fit_predict(x_normalized)
    assert len(labels) == len(X)

    model2 = LeaderClustering(max_radius=MAX_RADIUS, metric='cosine',
                              deterministic=True)
    labels2 = model2.fit_predict(X)
    assert np.array_equal(labels, labels2)

def test_precomputed():

    model = LeaderClustering(max_radius=MAX_RADIUS_EUCLIDEAN, metric='euclidean',
                             precomputed_dist=True)
    dist_matrix = compute_dist_matrix(X, metric='euclidean')
    labels = model.fit_predict(dist_matrix)
    assert compute_max_dist(X, labels, metric='euclidean') < MAX_RADIUS_EUCLIDEAN * 2


def test_sparse():

    model = LeaderClustering(max_radius=MAX_RADIUS_EUCLIDEAN, metric='euclidean',
                             sparse_dist=True)
    labels = model.fit_predict(X)
    assert compute_max_dist(X, labels, metric='euclidean') < MAX_RADIUS_EUCLIDEAN * 2

    model = LeaderClustering(max_radius=MAX_RADIUS_EUCLIDEAN,
                             sparse_dist=True, precomputed_dist=True)
    dist_matrix = compute_sparse_dist_matrix(X, metric='euclidean',
                                             max_distance=MAX_RADIUS_EUCLIDEAN)
    labels = model.fit_predict(dist_matrix)
    assert compute_max_dist(X, labels, metric='euclidean') < MAX_RADIUS_EUCLIDEAN * 2


def test_deterministic():

    model = LeaderClustering(max_radius=MAX_RADIUS, metric='cosine',
                             deterministic=True)
    labels1 = model.fit_predict(X)
    labels2 = model.fit_predict(X)
    assert np.array_equal(labels1, labels2)


def test_sparse_dense_equivalence():

    model1 = LeaderClustering(max_radius=MAX_RADIUS, metric='cosine',
                              sparse_dist=False, deterministic=True)
    labels1 = model1.fit_predict(X)

    model2 = LeaderClustering(max_radius=MAX_RADIUS, metric='cosine',
                              sparse_dist=True, deterministic=True)
    labels2 = model2.fit_predict(X)

    assert np.array_equal(labels1, labels2)
