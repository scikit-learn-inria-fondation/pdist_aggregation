import numpy as np
import pytest
from sklearn.neighbors import NearestNeighbors

from nn import KeOpsNearestNeighbors
from pdist_aggregation import parallel_knn


@pytest.mark.parametrize("n", [10 ** i for i in [2, 3, 4]])
@pytest.mark.parametrize("d", [2, 5, 10, 100])
@pytest.mark.parametrize("ratio_train_test", [10, 2, 1, 0.5])
@pytest.mark.parametrize("n_neighbors", [1, 10, 100])
@pytest.mark.parametrize("chunk_size", [2 ** i for i in range(9, 13)])
@pytest.mark.parametrize("use_chunk_on_train", [True, False])
def test_correctness(
    n,
    d,
    ratio_train_test,
    n_neighbors,
    chunk_size,
    use_chunk_on_train,
    dtype=np.float64,
):
    np.random.seed(1)
    X_train = np.random.rand(int(n * d)).astype(dtype).reshape((-1, d))
    X_test = (
        np.random.rand(int(n * d / ratio_train_test)).astype(dtype).reshape((-1, d))
    )

    neigh = NearestNeighbors(n_neighbors=n_neighbors, algorithm="brute")
    neigh.fit(X_train)

    knn_sk = neigh.kneighbors(X_test, return_distance=False)
    knn, _ = parallel_knn(
        X_train,
        X_test,
        k=n_neighbors,
        chunk_size=chunk_size,
        use_chunk_on_train=use_chunk_on_train,
    )

    np.testing.assert_array_equal(knn, knn_sk)


@pytest.mark.parametrize("n", [10 ** i for i in [2, 3, 4]])
@pytest.mark.parametrize("d", [2, 5, 10, 100])
@pytest.mark.parametrize("ratio_train_test", [10, 2, 1, 0.5])
@pytest.mark.parametrize("n_neighbors", [1, 10, 100])
def test_correctness_pykeops(
    n,
    d,
    ratio_train_test,
    n_neighbors,
    dtype=np.float64,
):
    np.random.seed(1)
    Y = np.random.rand(int(n * d)).astype(dtype).reshape((-1, d))
    X = np.random.rand(int(n * d // ratio_train_test)).astype(dtype).reshape((-1, d))

    neigh = NearestNeighbors(n_neighbors=n_neighbors, algorithm="brute")
    neigh.fit(Y)

    knn_sk = neigh.kneighbors(X, return_distance=False)
    knn = KeOpsNearestNeighbors(n_neighbors=n_neighbors).fit(Y).kneighbors(X)

    np.testing.assert_array_equal(knn, knn_sk)
