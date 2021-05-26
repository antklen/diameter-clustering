# Clustering with maximum diameter

Clustering algorithms with maximum distance between points inside clusters.

When we have interpetable metric like cosine distance it could be good to have clusters with maximum distance between points. Then we can find good threshold for maximum distance and be confident that points inside clusters are really similar. Unfortunately popular algorithms don't have such behavior.

Main algorithm is MaxDiameterClustering. It is a simple greedy algorithm, in which we add points one by one. If there is a cluster with all points close enough to new points, then we add new point to this cluster. If there is no such cluster, this point starts new cluster.

Also two similar algorithms are added - Leader Clustering and Quality Threshold Clustering.

## Usage

### MaxDiameterClustering

Basic usage of MaxDiameterClustering:
```python
from sklearn.datasets import make_blobs
from diameter import MaxDiameterClustering

X, y = make_blobs(n_samples=100, n_features=50)

model = MaxDiameterClustering(max_distance=0.3, metric='cosine')
labels = model.fit_predict(X)
```

When we want to compute cosine distance and our vectors are normalized, it is better to use
`inner_product` as metric because it is much faster:
```python
X_normalized = X/(np.linalg.norm(X, axis=-1, keepdims=True) + 1e-16)

model = MaxDiameterClustering(max_distance=0.3, metric='inner_product')
labels = model.fit_predict(X_normalized)
```

Instead of using feature matrix `X` we can pass precomputed distance matrix:
```python
from diameter.dist_matrix import compute_dist_matrix

dist_matrix = compute_dist_matrix(X, metric='cosine')

model = MaxDiameterClustering(max_distance=0.3, precomputed_dist=True)
labels = model.fit_predict(dist_matrix)
```

Calculation of full distance matrix between all points is expensive, so for big datasets
it is better to use distance matrix in sparse format:
```python
model = MaxDiameterClustering(max_distance=0.3, metric='cosine', sparse_dist=True)
labels = model.fit_predict(X)

model = MaxDiameterClustering(max_distance=0.3, sparse_dist=True, precomputed_dist=True)
dist_matrix = compute_sparse_dist_matrix(X, max_distance=0.3, metric='cosine')
labels = model.fit_predict(dist_matrix)
```

With `deterministic=True` we can get reproducible results:
```python
model = MaxDiameterClustering(max_distance=0.3, metric='cosine', deterministic=True)
labels = model.fit_predict(X)
```



### Leader Clustering

```python
from diameter import LeaderClustering

model = LeaderClustering(max_radius=0.15, metric='cosine')
labels = model.fit_predict(X)
```

Precomputed distance, sparse distance, deterministic behavior  and inner_product
could be used as in MaxDiameterClustering.


### Quality Threshold Clustering

```python
from diameter import QTClustering

model = QTClustering(max_radius=0.15, metric='cosine', min_cluster_size=5)
labels = model.fit_predict(X)
```

Precomputed distance, sparse distance  and inner_product
could be used as in MaxDiameterClustering. This algorithm is deterministic by design.





