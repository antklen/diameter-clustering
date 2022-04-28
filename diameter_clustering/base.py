"""
Base classes for all clustering algorithms.
"""

import logging
from typing import Union

import numpy as np
from scipy.sparse import csr_matrix

from .dist_matrix import compute_dist_matrix, compute_sparse_dist_matrix
from .timer import timer


class FitPredictMixin:
    """Mixin with fit_predict method."""

    def fit_predict(self, X: Union[np.ndarray, csr_matrix]) -> np.ndarray:
        """Fit clustering from features or distance matrix and return cluster labels.

        Args:
            X (np.ndarray or scipy.sparse.csr_matrix): Array with features
                or precomputed distance matrix, which could be in sparse matrix format.

        Returns:
            Array with cluster labels.
        """

        self.fit(X)

        return self.labels_


class DistanceMatrixMixin:
    """Mixin with methods for working with distance matrix."""

    def _prepare_distance_matrix(self, X: Union[np.ndarray, csr_matrix]):
        """Prepare distance matrix.

        If self.precomputed_dist is True then do nothing, only check for correctness of X.
        Otherwise compute distance matrix regarding X as array of features. If self.sparse_dist
        is True then compute matrix in sparse format."""

        if not self.precomputed_dist:
            if self.sparse_dist:
                logging.info('computing distance matrix in sparse format...')
                with timer('compute_sparse_dist_matrix'):
                    return compute_sparse_dist_matrix(X, metric=self.metric,
                                                      max_distance=self.max_distance)
            else:
                logging.info('computing distance matrix in dense format...')
                with timer('compute_dist_matrix'):
                    return compute_dist_matrix(X, metric=self.metric)

        if X.shape[0] != X.shape[1]:
            raise ValueError(f'Distance matrix should be square. Got matrix of shape {X.shape}.')

        if self.sparse_dist:
            if not isinstance(X, csr_matrix):
                raise TypeError('Sparse distance matrix should be in '
                                'scipy.sparse.csr_matrix format.')
        elif not isinstance(X, np.ndarray):
            raise TypeError('Dense distance matrix should be '
                            'an instance of np.ndarray.')

        return X

    def _slice_distance_matrix(self, dist_matrix: Union[np.ndarray, csr_matrix],
                               idx: int, indexes: np.ndarray):
        """Get one row of distance matrix.
        Get distance between given point and several other points.

        Args:
            dist (np.ndarray or scipy.sparse.csr_matrix): Distance matrix.
            idx (int): Index of given point.
            indexes (np.ndarray): Indexes of other points.
        """

        if isinstance(dist_matrix, csr_matrix):
            current_dist = dist_matrix[idx, indexes].toarray()[0, :]
            current_dist[current_dist == 0] = np.inf
        else:
            current_dist = dist_matrix[idx, indexes]

        return current_dist
