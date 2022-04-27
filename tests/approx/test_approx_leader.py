"""Tests for ApproxLeaderClustering."""

import numpy as np
from scipy.spatial.distance import pdist
from sklearn.datasets import make_blobs

from diameter_clustering.approx import ApproxLeaderClustering, HNSWIndex
from diameter_clustering.dist_matrix import compute_dist_matrix, compute_sparse_dist_matrix


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


def test_approx_leader():

    hnsw_index = HNSWIndex(max_elements=len(X), space='l2', dim=50,
                           ef=100, ef_construction=200, M=16)
    model = ApproxLeaderClustering(hnsw_index, max_radius=MAX_RADIUS_EUCLIDEAN)
    labels = model.fit_predict(X)
    assert compute_max_dist(X, labels, metric='euclidean') < MAX_RADIUS_EUCLIDEAN * 2

    hnsw_index = HNSWIndex(max_elements=len(X), space='cosine', dim=50,
                           ef=100, ef_construction=200, M=16)
    model = ApproxLeaderClustering(hnsw_index, max_radius=MAX_RADIUS)
    labels = model.fit_predict(X)
    assert len(labels) == len(X)


def test_deterministic():

    hnsw_index1 = HNSWIndex(max_elements=len(X), space='cosine', dim=50,
                        ef=100, ef_construction=200, M=16)
    model1 = ApproxLeaderClustering(hnsw_index1, max_radius=0.2, deterministic=True)
    labels1 = model1.fit_predict(X)

    hnsw_index2 = HNSWIndex(max_elements=len(X), space='cosine', dim=50,
                            ef=100, ef_construction=200, M=16)
    model2 = ApproxLeaderClustering(hnsw_index2, max_radius=0.2, deterministic=True)
    labels2 = model2.fit_predict(X)
    assert np.array_equal(labels1, labels2)
