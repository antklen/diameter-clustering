"""
Version of Leader clustering using approximate nearest neighbors search.
"""

import numpy as np
from tqdm import tqdm

from .hnsw import HNSWIndex
from ..timer import timer


class ApproxLeaderClustering:
    """Leader clustering algorithm with approximate nearest neighbors search.

    Approximate nearest neighbors index is used to store leaders of clusters
    and to find nearest leader for new points.

    Args:
        ann_index: instance of HNSWIndex.
        max_radius: Maximum radius of each cluster
            (maximum distance between the leader and all other points in cluster).
        deterministic: If True then take points one by one to get deterministic behavior.
            If False then select points at random, so results would be different for each run.
        verbose: If True then output progress info, otherwise be silent.

    Attributes:
        labels_ (np.ndarray): Array with cluster labels after fitting model.
        n_clusters_ (int): Number of clusters after fitting model.
        leaders_ (np.ndarray): Array with 1 for cluster leaders and with 0 for all other points.

    Examples:
        import numpy as np
        from diameter_clustering.approx import HNSWIndex
        from diameter_clustering.approx import ApproxLeaderClustering

        # fit model
        data = np.random.rand(1000, 50)
        hnsw_index = HNSWIndex(max_elements=len(data), space='cosine', dim=50,
                               ef=100, ef_construction=200, M=16)
        model = ApproxLeaderClustering(hnsw_index, max_radius=0.2, deterministic=True)
        labels = model.fit_predict(data)

        # save index for later usage
        hnsw_index.save('hnsw_index.bin')

        # predict clusters for new data later
        new_data = np.random.rand(100, 50)
        hnsw_index = HNSWIndex(max_elements=len(new_data), path='hnsw_index.bin',
                               space='cosine', dim=50, ef=100)
        model = ApproxLeaderClustering(hnsw_index, max_radius=0.2, deterministic=True)
        new_labels = model.predict(new_data)
    """

    def __init__(self, ann_index: HNSWIndex, max_radius: float = 0.1,
                 deterministic: bool = True, verbose: bool = True):

        self.ann_index = ann_index
        self.max_radius = max_radius
        self.deterministic = deterministic
        self.verbose = verbose

        self.labels_ = None
        self.leaders_ = None
        self.n_clusters_ = None

    def fit(self, X: np.ndarray):
        """Fit clustering.

        Args:
            X: Array with features.
        """

        # create arrays for labels and leaders
        labels = np.empty(len(X))
        labels.fill(np.nan)
        leaders = np.zeros(len(X))

        # handle case when empty input data is passed
        if len(labels) == 0:
            self.labels_ = labels
            self.leaders_ = leaders
            self.n_clusters_ = 0
            return

        # choose first point and assign label to it
        idx = 0 if self.deterministic else np.random.choice(range(len(labels)))
        labels[idx] = 0
        next_cluster = 1
        leaders[idx] = 1
        self.ann_index.add_item(X[idx])

        for _ in tqdm(range(len(labels)-1), desc='ApproxLeaderClustering fit',
                      disable=not self.verbose):

            # choose next point
            indexes = np.where(np.isnan(labels))[0]
            idx = indexes[0] if self.deterministic else np.random.choice(indexes)

            # find nearest leader
            nearest_leader_idx, nearest_leader_dist = self.ann_index.find_nearest_point(X[idx])

            if nearest_leader_dist <= self.max_radius:
                # assign cluster with nearest leader as label
                labels[idx] = nearest_leader_idx
            else:
                # assign new cluster label
                labels[idx] = next_cluster
                leaders[idx] = 1
                next_cluster += 1
                self.ann_index.add_item(X[idx])

        self.labels_ = labels.astype(int)
        self.leaders_ = leaders
        self.n_clusters_ = labels.max() + 1


    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit clustering and return cluster labels.

        Args:
            X: Array with features.

        Returns:
            Numpy array with labels for data points in X.
        """

        self.fit(X)

        return self.labels_

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Assigning new points to existent clusters without making new clusters.

        Returning -1 for points which can't be assigned to any cluster.
        Finding nearest leaders for points one by one.

        Args:
            X: Array with features for new points.

        Returns:
            Numpy array with labels for data points in X.
        """

        # create array for new labels
        labels = np.empty(len(X))

        for idx in tqdm(range(len(X)), desc='ApproxLeaderClustering assign points to clusters',
                        disable=not self.verbose):

            # find nearest leader
            nearest_leader_idx, nearest_leader_dist = self.ann_index.find_nearest_point(X[idx])

            if nearest_leader_dist <= self.max_radius:
                # assign cluster with nearest leader as label
                labels[idx] = nearest_leader_idx
            else:
                # assign -1 for point which is not close enough to any leader
                labels[idx] = -1

        return labels


    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """Assigning new points to existent clusters without making new clusters.

        Returning -1 for points which can't be assigned to any cluster.
        Finding nearest leaders for all points at once.

        Args:
            X: Array with features for new points.

        Returns:
            Numpy array with labels for data points in X.
        """

        # create array for new labels
        labels = np.empty(len(X))

        # find nearest leaders
        with timer('find_nearest_point_batch', disable=not self.verbose):
            nearest_leaders_idx, nearest_leaders_dist = \
                self.ann_index.find_nearest_point_batch(X)

        for idx in range(len(X)):

            if nearest_leaders_dist[idx] <= self.max_radius:
                # assign cluster with nearest leader as label
                labels[idx] = nearest_leaders_idx[idx]
            else:
                # assign -1 for point which is not close enough to any leader
                labels[idx] = -1

        return labels
