"""
Simple greedy algorithm for clustering with maximum distance between points inside clusters.
"""

from typing import Union

import numpy as np
import numpy_groupies as npg
from scipy.sparse import csr_matrix
from tqdm import tqdm

from .base import FitPredictMixin, DistanceMatrixMixin
from .timer import TimerWithHistory


class MaxDiameterClustering(FitPredictMixin, DistanceMatrixMixin):
    """Clustering with maximum diameter (maximum distance between points) inside clusters.

    Args:
        max_distance (float): Maximum distance between points in clusters.
        criterion (str): Criterion for choosing cluster from several candidates.
            If 'distance' then choose cluster with minimum average distance to given point.
            If 'size' then choose cluster with maximum current size.
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
            If False, then distance matrix is ordinary numpy array.
        deterministic (bool): If True then take points one by one to get determenistic behavior.
            If False then select points at random, so results would be different for each run.
        use_timer (bool): If True then use TimerWithHistory in fit method, which can be accessed
            via self.timer. Can be useful for debugging.

    Attributes:
        labels_ (np.array): Array with cluster labels after fitting model.
        n_clusters_ (int): Number of clusters after fitting model.
        timer: Timer with history of execution time (access history via self.timer.history).
    """

    def __init__(self, max_distance: float = 0.2, criterion: str = 'distance',
                 metric: str = 'cosine', precomputed_dist: bool = False,
                 sparse_dist: bool = True, deterministic: bool = False,
                 use_timer: bool = False):

        if criterion not in ['size', 'distance']:
            raise ValueError('Wrong criterion value, should be "size" or "distance".')

        self.max_distance = max_distance
        self.criterion = criterion
        self.metric = metric
        self.precomputed_dist = precomputed_dist
        self.sparse_dist = sparse_dist
        self.deterministic = deterministic
        self.use_timer = use_timer

        self.labels_ = None
        self.n_clusters_ = None
        self.timer = None

    def fit(self, X: Union[np.ndarray, csr_matrix]):
        """Fit clustering from features or distance matrix.

        Args:
            X (np.ndarray or scipy.sparse.csr_matrix): Array with features or
                precomputed distance matrix, could be in sparse format.
        """

        dist_matrix = self._prepare_distance_matrix(X)

        # create array for labels
        labels = np.empty(dist_matrix.shape[0])
        labels.fill(np.nan)

        # handle case when empty input data is passed
        if len(labels) == 0:
            self.labels_ = labels
            self.n_clusters_ = 0
            return

        # choose first point and assign label to it
        idx = 0 if self.deterministic else np.random.choice(range(len(labels)))
        labels[idx] = 0
        next_cluster = 1

        self.timer = TimerWithHistory(disable=not self.use_timer)

        for _ in tqdm(range(len(labels)-1), desc='MaxDiameterClustering fit'):

            # choose next point
            with self.timer(name='choose_next_point'):
                indexes = np.where(np.isnan(labels))[0]
                idx = indexes[0] if self.deterministic else np.random.choice(indexes)
            # find indices of already labeled points
            with self.timer(name='find_labeled_points'):
                current_cluster_idx = np.where(~np.isnan(labels))[0]
                current_cluster_labels = labels[current_cluster_idx].astype(int)
            # find distances to already labeled points
            with self.timer(name='get_distances'):
                current_dist = self._slice_distance_matrix(dist_matrix, idx, current_cluster_idx)

            # find max distance to each existent cluster
            with self.timer(name='max_distance_to_clusters'):
                cluster_dist_max = npg.aggregate(current_cluster_labels, current_dist,
                                                 func='max', fill_value=np.inf)

            if np.min(cluster_dist_max) <= self.max_distance:
                # find existent clusters with max dist < threshold
                with self.timer(name='candidate_clusters'):
                    candidate_clusters = np.where(cluster_dist_max <= self.max_distance)[0]
                    # directly get label if there is only one such cluster
                    if len(candidate_clusters) == 1:
                        labels[idx] = candidate_clusters[0]
                        continue
                    # otherwise we need to choose between candidate clusters
                    candidate_clusters_idx = np.isin(current_cluster_labels, candidate_clusters)
                    candidate_clusters_labels = current_cluster_labels[candidate_clusters_idx]

                if self.criterion == 'distance':
                    candidate_clusters_dist = current_dist[candidate_clusters_idx]
                    labels[idx] = self._best_candidate_distance(candidate_clusters_labels,
                                                                candidate_clusters_dist)
                elif self.criterion == 'size':
                    labels[idx] = self._best_candidate_size(candidate_clusters_labels)
            else:
                # assign new cluster label
                with self.timer(name='assign_new_label'):
                    labels[idx] = next_cluster
                    next_cluster += 1

        self.labels_ = labels.astype(int)
        self.n_clusters_ = labels.max() + 1

    def _best_candidate_distance(self, candidate_clusters_labels: np.ndarray,
                                 candidate_clusters_dist: np.ndarray) -> int:
        """Find best candidate cluster based on average distance to clusters."""

        # find average distance to clusters
        with self.timer(name='average_distance_to_clusters'):
            cluster_dist_mean = npg.aggregate(candidate_clusters_labels,
                                              candidate_clusters_dist,
                                              func='mean', fill_value=np.inf)

        # assign cluster with min average distance as label
        with self.timer(name='distance_argmin'):
            label = cluster_dist_mean.argmin()

        return label

    def _best_candidate_size(self, candidate_clusters_labels: np.ndarray) -> int:
        """Find best candidate cluster based on size of clusters."""

        # find size of clusters
        with self.timer(name='average_size_of_clusters'):
            cluster_size = npg.aggregate(candidate_clusters_labels,
                                         candidate_clusters_labels,
                                         func='count')
        # assign cluster with max size as label
        with self.timer(name='size_argmax'):
            label = cluster_size.argmax()

        return label
