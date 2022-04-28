"""
Implementation of Leader clustering.
"""

from typing import Union

import numpy as np
from scipy.sparse import csr_matrix
from tqdm import tqdm

from .base import FitPredictMixin, DistanceMatrixMixin


class LeaderClustering(FitPredictMixin, DistanceMatrixMixin):
    """Leader clustering algorithm.

    Args:
        max_radius (float): Maximum radius of cluster
            (maximum distance between leader and all other points in cluster).
        change_leaders (bool): if True then change cluster leader if there is a point with smaller
            average distance to all points in cluster.
        metric (str): Distance metric.
            For sparse_dist=True possible options are in sklearn.neighbors.VALID_METRICS['brute'].
            For sparse_dist=False possible options are 'inner_product' or one of metrics
            available in scipy.spatial.distance.pdist. If 'inner_product' then use np.inner
            which is much faster than pdist. 'inner_product' could be used instead
            of cosine distance for normalized vectors.
        precomputed_dist (bool): If True, then input should be precomputed distance matrix,
            if False then input is array with features.
        sparse_dist (bool): If True, then use distance matrix in sparse format (zero elements
            are elements for which distance between points is greater than max_distance).
            If False, then consider distance matrix as ordinary numpy array.
        deterministic (bool): If True then take points one by one to get deterministic behavior.
            If False then select points at random, so results would be different for each run.

    Attributes:
        labels_ (np.array): Array with cluster labels after fitting model.
        n_clusters_ (int): Number of clusters after fitting model.
        leaders_ (np.array): Array with 1 for cluster leaders and with 0 for all other points.
    """

    def __init__(self, max_radius: float = 0.1, change_leaders: bool = False,
                 metric: str = 'cosine', precomputed_dist: bool = False,
                 sparse_dist: bool = True, deterministic: bool = False):

        self.max_radius = max_radius
        self.change_leaders = change_leaders
        self.metric = metric
        self.precomputed_dist = precomputed_dist
        self.sparse_dist = sparse_dist
        self.deterministic = deterministic

        self.max_distance = max_radius  # is needed for computation of sparse distance matrix
        self.labels_ = None
        self.leaders_ = None
        self.n_clusters_ = None

    def fit(self, X: Union[np.ndarray, csr_matrix]):
        """Fit clustering from features or distance matrix.

        Args:
            X (np.ndarray or scipy.sparse.csr_matrix): Array with features or
                precomputed distance matrix, could be in sparse matrix format.
        """

        dist_matrix = self._prepare_distance_matrix(X)

        # create arrays for labels and leaders
        labels = np.empty(dist_matrix.shape[0])
        labels.fill(np.nan)
        leaders = np.zeros(dist_matrix.shape[0])

        # choose first point and assign label to it
        idx = 0 if self.deterministic else np.random.choice(range(len(labels)))
        labels[idx] = 0
        next_cluster = 1
        leaders[idx] = 1

        for _ in tqdm(range(len(labels)-1), desc='LeaderClustering fit'):

            # choose next point
            indexes = np.where(np.isnan(labels))[0]
            idx = indexes[0] if self.deterministic else np.random.choice(indexes)
            # find indices of current leaders
            current_leaders_idx = np.where(leaders == 1)[0]
            current_leaders_labels = labels[current_leaders_idx]

            # find distances to current leaders
            leaders_dist = self._slice_distance_matrix(dist_matrix, idx, current_leaders_idx)

            if np.min(leaders_dist) <= self.max_radius:
                # assign cluster with nearest leader as label
                labels[idx] = current_leaders_labels[leaders_dist.argmin()]

                # change leader in cluster if there is better candidate for it
                if self.change_leaders:
                    cluster_idx = np.where(labels == labels[idx])[0]
                    dist_inside = dist_matrix[cluster_idx][:, cluster_idx].mean(axis=1)
                    min_idx = cluster_idx[dist_inside.argmin()]
                    nearest_leader_idx = current_leaders_idx[leaders_dist.argmin()]
                    if min_idx != nearest_leader_idx:
                        leaders[nearest_leader_idx] = 0
                        leaders[min_idx] = 1

            else:
                # assign new cluster label
                labels[idx] = next_cluster
                leaders[idx] = 1
                next_cluster += 1

        self.labels_ = labels.astype(int)
        self.leaders_ = leaders
        self.n_clusters_ = labels.max() + 1
