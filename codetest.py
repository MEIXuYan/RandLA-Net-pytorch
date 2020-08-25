import numpy as np
from sklearn.neighbors import KDTree
# KDTree 
rng = np.random.RandomState(0)
X = rng.random_sample((10, 3))  # 10 points in 3 dimensions

tree = KDTree(X, leaf_size=2)              # doctest: +SKIP
dist, ind = tree.query(X[:1], k=3)                # doctest: +SKIP

inds = tree.query(X[:1], return_distance=False)

print(X)
print(X[:1])
print(ind)  # indices of 3 closest neighbors
print(X[ind])
print(dist)  # distances to 3 closest neighbors
print(inds)