import numpy as np
import pytest
from sklearn.neighbors import NearestNeighbors

from pdist_aggregation import parallel_knn


@pytest.mark.parametrize("n", [10 ** i for i in [2, 3, 4]])
@pytest.mark.parametrize("d", [2, 5, 10, 100])
@pytest.mark.parametrize("ratio_train_test", [10, 2, 1, 0.5])
@pytest.mark.parametrize("n_neighbors", [1, 10, 100])
@pytest.mark.parametrize("chunk_size", [2 ** i for i in range(12, 17)])
def test_correctness(n, d, ratio_train_test, n_neighbors, chunk_size, dtype=np.float64):
    np.random.seed(1)
    Y = np.random.rand(int(n * d)).astype(dtype).reshape((-1, d))
    X = np.random.rand(int(n * d / ratio_train_test)).astype(dtype).reshape((-1, d))

    neigh = NearestNeighbors(n_neighbors=n_neighbors, algorithm="brute")
    neigh.fit(Y)

    knn_sk = neigh.kneighbors(X, return_distance=False)
    knn, _ = parallel_knn(X, Y, k=n_neighbors, chunk_size=chunk_size)

    np.testing.assert_array_equal(knn, knn_sk)
