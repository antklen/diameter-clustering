# Clustering with maximum diameter

Clustering algorithms with maximum distance between points inside clusters.

When we have interpetable metric like cosine distance it could be nice to have clusters with maximum distance between points. Then we can find good threshold for maximum distance and be confident that points inside clusters are really similar. Unfortunately popular clustering algorithms don't have such behavior.

Main algorithm is MaxDiameterClustering. It is a simple greedy algorithm, in which we add points one by one. If there is a cluster with all points close enough to new points, then we add new point to this cluster. If there is no such cluster, this point starts new cluster.

Also two similar algorithms are added - Leader Clustering and Quality Threshold Clustering.


## Installation

Install from PyPI
```sh
pip install diameter-clustering
```

Install from source
```sh
pip install git+git://github.com/antklen/diameter-clustering.git
# or
git clone git@github.com:antklen/diameter-clustering.git
cd diameter-clustering
pip install .
```

## Usage

### MaxDiameterClustering

Basic usage of MaxDiameterClustering:
```python
from sklearn.datasets import make_blobs
from diameter_clustering import MaxDiameterClustering

X, y = make_blobs(n_samples=100, n_features=50)

model = MaxDiameterClustering(max_distance=0.3, metric='cosine')
labels = model.fit_predict(X)
```

Instead of using feature matrix `X` we can pass precomputed distance matrix:
```python
from diameter_clustering.dist_matrix import compute_sparse_dist_matrix

dist_matrix = compute_sparse_dist_matrix(X, metric='cosine')

model = MaxDiameterClustering(max_distance=0.3, precomputed_dist=True)
labels = model.fit_predict(dist_matrix)
```

By default computation of distance matrix in sparse format is used (`sparse_dist=True`), because calculation of distance matrix between all points in dense format is expensive. But when dataset is not so big (roughly less than 20k-30k points) `sparse_dist=False` mode can be used. It could be faster for small datasets or useful when you already have precomputed distance matrix in dense format.
```python
model = MaxDiameterClustering(max_distance=0.3, metric='cosine', sparse_dist=False)
labels = model.fit_predict(X)


from diameter_clustering.dist_matrix import compute_dist_matrix

dist_matrix = compute_dist_matrix(X, max_distance=0.3, metric='cosine')

model = MaxDiameterClustering(max_distance=0.3, sparse_dist=False, precomputed_dist=True)
labels = model.fit_predict(dist_matrix)
```

When we want to compute cosine distance in dense format and our vectors are normalized, it is better to use
`inner_product` as metric because it is much faster:
```python
X_normalized = X/(np.linalg.norm(X, axis=-1, keepdims=True) + 1e-16)

model = MaxDiameterClustering(max_distance=0.3, metric='inner_product', sparse_dist=False)
labels = model.fit_predict(X_normalized)
```

With `deterministic=True` we can get reproducible results:
```python
model = MaxDiameterClustering(max_distance=0.3, metric='cosine', deterministic=True)
labels = model.fit_predict(X)
```



### Leader Clustering

```python
from diameter_clustering import LeaderClustering

model = LeaderClustering(max_radius=0.15, metric='cosine')
labels = model.fit_predict(X)
```

`precomputed_dist`, `sparse_dist`, `deterministic` and `inner_product`
can be used as in MaxDiameterClustering.


### Quality Threshold Clustering

```python
from diameter_clustering import QTClustering

model = QTClustering(max_radius=0.15, metric='cosine', min_cluster_size=5)
labels = model.fit_predict(X)
```

`precomputed_dist`, `sparse_dist`, and `inner_product`
can be used as in MaxDiameterClustering. This algorithm is deterministic by design.





