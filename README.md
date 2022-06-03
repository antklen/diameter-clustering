# Clustering with maximum diameter

Collection of clustering algorithms with maximum distance between points inside clusters.

When we have interpretable metric like cosine distance it could be nice to have clusters with maximum distance between points. Then we can find good threshold for maximum distance and be confident that points inside clusters are really similar. Also we dont' need to specify number of clusters with such approach.

Unfortunately most of popular clustering algorithms don't have such behavior.

Possible applications:
- Embeddings of text data with cosine distance.
- Geo data with haversine distance.

## Algorithms

### MaxDiameterClustering

A simple greedy algorithm, in which we add points one by one. If there is a cluster with all points close enough to new points, then we add new point to this cluster. If there is no such cluster, this point starts new cluster.

### Quality Threshold Clustering

[Explanation](https://sites.google.com/site/dataclusteringalgorithms/quality-threshold-clustering-algorithm-1).

Inspired by this [repository](https://github.com/melvrl13/python-quality-threshold).
### Leader Clustering

[Explanation on stackoverflow](https://stackoverflow.com/questions/36928654/leader-clustering-algorithm-explanation)

[R package](https://cran.r-project.org/web/packages/leaderCluster/index.html)

### Approximate Leader Clustering

Use approximate nearest neighbors search (currently hnswlib) to speed up Leader Clustering.


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

### Quality Threshold Clustering

```python
from diameter_clustering import QTClustering

model = QTClustering(max_radius=0.15, metric='cosine', min_cluster_size=5)
labels = model.fit_predict(X)
```

`precomputed_dist`, `sparse_dist`, and `inner_product`
can be used as in MaxDiameterClustering. This algorithm is deterministic by design.

### Leader Clustering

```python
from diameter_clustering import LeaderClustering

model = LeaderClustering(max_radius=0.15, metric='cosine')
labels = model.fit_predict(X)
```

`precomputed_dist`, `sparse_dist`, `deterministic` and `inner_product`
can be used as in MaxDiameterClustering.

### Approximate Leader Clustering

```python
from diameter_clustering.approx import HNSWIndex
from diameter_clustering.approx import ApproxLeaderClustering

# fit model
hnsw_index = HNSWIndex(max_elements=len(X), space='cosine', dim=50,
                       ef=100, ef_construction=200, M=16)
model = ApproxLeaderClustering(hnsw_index, max_radius=0.15, deterministic=True)
labels = model.fit_predict(X)

# save index for later usage
hnsw_index.save('hnsw_index.bin')

# predict clusters for new data later
hnsw_index = HNSWIndex(max_elements=len(X_new), path='hnsw_index.bin',
                        space='cosine', dim=50, ef=100)
model = ApproxLeaderClustering(hnsw_index, max_radius=0.15, deterministic=True)
new_labels = model.predict(X_new)
```
