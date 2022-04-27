"""
Wrapper for approximate nearest neighbors search using hnswlib library.
"""

from typing import Optional

import hnswlib
import numpy as np


class HNSWIndex:
    """
    Approximate nearest neighbors search using hnswlib library.

    Args:
        max_elements: Maximum number of elements that can be stored in index (hnswlib parameter).
        path: Path to previously saved index. If not None, load it. If None, initialize empty index.
        space: Distance metric (hnswlib parameter). Possible values:
            'l2', 'ip' (inner product), 'cosine.
        dim: Dimensionality of vectors in index (hnswlib parameter).
        ef: hnswlib parameter, defines query time accuracy/speed trade-off.
        ef_construction: hnswlib parameter, defines construction time/accuracy trade-off.
        M: hnswlib parameter, defines maximum number of outgoing connections in the graph.

    Attributes:
        index: Instance of hnswlib.Index.
    """

    def __init__(self, max_elements: int, path: Optional[str] = None,
                 space: str = 'ip', dim: int = 512, ef: int = 100,
                 ef_construction: int = 250, M: int = 16):

        self.index = hnswlib.Index(space=space, dim=dim)

        if path is not None:
            self.index.load_index(path, max_elements=max_elements)
        else:
            self.index.init_index(max_elements=max_elements, ef_construction=ef_construction, M=M)

        self.index.set_ef(ef)

    def add_item(self, vector: np.array, label: Optional[int] = None):
        """Add one element to index.

        Args:
            vector: Numpy array with vector for one element.
            label: Optional integer label for this element.
        """

        self.index.add_items(vector, ids=label)

    def add_items(self, vectors: np.array, labels: Optional[int] = None):
        """Add batch of elements to index.

        Args:
            vectors: Numpy array with vectors for given elements.
            label: Optional integer labels for this elements.
        """

        self.index.add_items(vectors, ids=labels)

    def find_nearest_point(self, vector: np.array):
        """Find nearest point from index for given vector.

        Args:
            vector: Numpy array.

        Returns:
            Label of nearest point and distance to it.
        """

        labels, distances = self.index.knn_query(vector, k=1)
        return labels[0, 0], distances[0, 0]

    def find_nearest_point_batch(self, vectors: np.array):
        """Find nearest point from index for batch of vectors.

        Args:
            vectors: Numpy array.

        Returns:
            Labels of nearest points and corresponding distances to it.
        """

        labels, distances = self.index.knn_query(vectors, k=1)
        return labels[:, 0], distances[:, 0]

    def save(self, path: str):
        """Save index to disk.

        Args:
            path: Save index to this path.
        """

        self.index.save_index(path)
