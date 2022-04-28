"""
Implementation of Quality threshold clustering.
"""

from typing import Union

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from tqdm import tqdm

from .mixins import FitPredictMixin, DistanceMatrixMixin


class QTClustering(FitPredictMixin, DistanceMatrixMixin):
    """Quality threshold clustering.

    Args:
        max_radius (float): Maximum radius of cluster
            (maximum distance between center of cluster and all other points).
        min_cluster_size (int): Minimum size of clusters, stop iterations at this cluster size.
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
        verbose (bool): If True then output progress info, otherwise be silent.

    Attributes:
        labels_ (np.array): Array with cluster labels after fitting model.
        n_clusters_ (int): Number of clusters after fitting model.
        centers_ (np.array): Array with 1 for cluster centers and with 0 for all other points.
    """

    def __init__(self, max_radius: float = 0.1, min_cluster_size: int = 2,
                 metric: str = 'cosine', precomputed_dist: bool = False,
                 sparse_dist: bool = True, verbose: bool = True):

        self.max_radius = max_radius
        self.min_cluster_size = min_cluster_size
        self.metric = metric
        self.precomputed_dist = precomputed_dist
        self.sparse_dist = sparse_dist
        self.verbose = verbose

        self.max_distance = max_radius  # is needed for computation of sparse distance matrix
        self.labels_ = None
        self.centers_ = None
        self.n_clusters_ = None

    def fit(self, X: Union[np.ndarray, csr_matrix]):
        """Fit clustering from features or distance matrix.

        Args:
            X (np.ndarray or scipy.sparse.csr_matrix): Array with features or
                precomputed distance matrix, could be in sparse matrix format.
        """

        dist_matrix = self._prepare_distance_matrix(X)

        if self.sparse_dist:
            dist_mask = lil_matrix(dist_matrix)
            dist_mask[dist_mask > 0] = 1
            dist_mask.setdiag(1)
            labels, centers = self.fit_sparse(dist_mask)
        else:
            dist_mask = dist_matrix < self.max_radius
            np.fill_diagonal(dist_mask, True)
            labels, centers = self.fit_dense(dist_mask)

        self.labels_ = labels
        self.centers_ = centers
        self.n_clusters_ = labels.max() + 1

    def fit_dense(self, dist_mask: np.ndarray):
        """Fit clustering from distance matrix mask when it is dense matrix."""

        labels = np.empty(dist_mask.shape[0])
        labels.fill(np.nan)
        centers = []
        cluster_number = 0
        total_count = 0

        with tqdm(total=dist_mask.shape[0], disable=not self.verbose) as pbar:
            while dist_mask.any():

                # find size of candidate clusters for each point
                candidate_size = dist_mask.sum(axis=1)

                if np.max(candidate_size) < self.min_cluster_size:
                    labels[np.where(np.isnan(labels))] = -1
                    break

                # pick the biggest possible cluster from candidates
                center_idx = np.argmax(candidate_size)
                cluster_points_idx = np.where(dist_mask[center_idx])[0]
                # assign labels
                labels[cluster_points_idx] = cluster_number
                centers.append(center_idx)
                # remove labeled data from further calculations
                dist_mask[cluster_points_idx, :] = False
                dist_mask[:, cluster_points_idx] = False

                # finalize iteration
                cluster_number += 1
                size = np.max(candidate_size)
                total_count += size

                pbar.update(size)
                pbar.set_description(
                    f"QTClustering fit. Current cluster size {size}, total count {total_count}")

        return labels, centers

    def fit_sparse(self, dist_mask: csr_matrix):
        """Fit clustering from distance matrix mask when it is sparse matrix."""

        labels = np.empty(dist_mask.shape[0])
        labels.fill(np.nan)
        centers = []
        cluster_number = 0
        total_count = 0

        with tqdm(total=dist_mask.shape[0], disable=not self.verbose) as pbar:
            while dist_mask.sum() > 0:

                # find size of candidate clusters for each point
                candidate_size = dist_mask.sum(axis=1)

                if np.max(candidate_size) < self.min_cluster_size:
                    labels[np.where(np.isnan(labels))] = -1
                    break

                # pick the biggest possible cluster from candidates
                center_idx = np.argmax(candidate_size)
                cluster_points_idx = dist_mask[center_idx].nonzero()[1]
                # assign labels
                labels[cluster_points_idx] = cluster_number
                centers.append(center_idx)

                # remove labeled data from further calculations
                dist_mask[cluster_points_idx, :] = 0
                dist_mask[:, cluster_points_idx] = 0

                # finalize iteration
                cluster_number += 1
                size = np.max(candidate_size)
                total_count += size

                pbar.update(size)
                pbar.set_description(
                    f"QTClustering fit. Current cluster size {size}, total count {total_count}")

        return labels, centers
