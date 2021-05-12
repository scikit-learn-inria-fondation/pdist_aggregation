import numpy as np
import pytest
from pdist_agregation import parallel_knn
from sklearn.neighbors import NearestNeighbors


@pytest.mark.parametrize("n_samples", [10 ** i for i in [2, 3, 4]])
@pytest.mark.parametrize("n_features", [2, 5, 10, 100])
@pytest.mark.parametrize("ratio_train_test", [10, 2, 1, 0.5])
@pytest.mark.parametrize("n_neighbors", [1, 10, 100])
@pytest.mark.parametrize("working_memory", [2 ** i for i in range(10, 20)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_correctness(
        n_samples,
        n_features,
        ratio_train_test,
        n_neighbors,
        working_memory,
        dtype,
):
    np.random.seed(1)
    Y = (
        np.random.rand(int(n_samples * n_features))
            .astype(dtype)
            .reshape((-1, n_features))
    )
    X = (
        np.random.rand(int(n_samples * n_features / ratio_train_test))
            .astype(dtype)
            .reshape((-1, n_features))
    )

    neigh = NearestNeighbors(n_neighbors=n_neighbors, algorithm='brute')
    neigh.fit(Y)

    knn_sk = neigh.kneighbors(X, return_distance=False)
    knn, _ = parallel_knn(X, Y, k=n_neighbors, working_memory=working_memory)

    np.testing.assert_array_equal(knn, knn_sk)
