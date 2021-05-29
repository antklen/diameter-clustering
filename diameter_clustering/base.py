"""
Base classes for all clustering algorithms.
"""

import numpy as np

from .dist_matrix import compute_dist_matrix, compute_sparse_dist_matrix


class FitPredictMixin:
    """Mixin with fit_predict method."""

    def fit_predict(self, X):
        """Fit clustering from features or distance matrix and return cluster labels.

        Args:
            X (np.array or sparse matrix): Array with features or precomputed distance matrix,
                which could be in sparse matrix format.
        """

        self.fit(X)

        return self.labels_


class DistanceMatrixMixin:
    """Mixin with methods for working with distance matrix."""

    def _prepare_distance_matrix(self, X):
        """Prepare distance matrix.

        If self.precomputed_dist is True then do nothing, only check for square shape of X.
        Otherwise compute distance matrix regarding X as array of features. If self.sparse_dist
        is True then compute matrix in sparse format."""

        if not self.precomputed_dist:
            if self.sparse_dist:
                return compute_sparse_dist_matrix(X, metric=self.metric,
                                                  max_distance=self.max_distance)
            else:
                return compute_dist_matrix(X, metric=self.metric)

        if X.shape[0] != X.shape[1]:
            raise ValueError(f'Distance matrix should be square. Got matrix of shape {X.shape}.')

        return X

    def _slice_distance_matrix(self, dist, idx, indexes):
        """Get one row of distance matrix.
        Get distance between given point and several other points.

        Args:
            dist (np.array or csr_matrix): Distance matrix.
            idx (int): Index of given point.
            indexes (np.array): Indexes of other points.
        """

        if self.sparse_dist:
            current_dist = dist[idx, indexes].toarray()[0, :]
            current_dist[current_dist == 0] = np.inf
        else:
            current_dist = dist[idx, indexes]

        return current_dist
