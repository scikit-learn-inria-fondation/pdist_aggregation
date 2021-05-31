import numpy as np
import pytest
from sklearn.neighbors import NearestNeighbors

from pdist_aggregation import parallel_knn


@pytest.mark.parametrize("n_samples", [10 ** i for i in [2, 3, 4]])
@pytest.mark.parametrize("n_features", [2, 5, 10, 100])
@pytest.mark.parametrize("ratio_train_test", [10, 2, 1, 0.5])
@pytest.mark.parametrize("n_neighbors", [1, 10, 100, 1000])
@pytest.mark.parametrize("chunk_size", [2 ** i for i in range(9, 13)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("strategy", ["auto", "chunk_on_train", "chunk_on_test"])
def test_correctness(
    n_samples,
    n_features,
    ratio_train_test,
    n_neighbors,
    chunk_size,
    dtype,
    strategy,
):
    if n_samples < n_neighbors:
        pytest.skip(
            f"Skipping as n_samples (={n_samples}) < n_neighbors (={n_neighbors})",
            allow_module_level=True,
        )

    np.random.seed(1)
    X_train = (
        np.random.rand(int(n_samples * n_features))
        .astype(dtype)
        .reshape((-1, n_features))
    )
    X_test = (
        np.random.rand(int(n_samples * n_features / ratio_train_test))
        .astype(dtype)
        .reshape((-1, n_features))
    )

    neigh = NearestNeighbors(n_neighbors=n_neighbors, algorithm="brute")
    neigh.fit(X_train)

    knn_sk = neigh.kneighbors(X_test, return_distance=False)
    knn, _ = parallel_knn(
        X_train,
        X_test,
        k=n_neighbors,
        chunk_size=chunk_size,
        strategy=strategy,
    )

    np.testing.assert_array_equal(knn, knn_sk)


@pytest.mark.parametrize("n_samples", [10 ** i for i in [2, 3, 4]])
@pytest.mark.parametrize("n_features", [2, 5, 10, 100])
@pytest.mark.parametrize("ratio_train_test", [10, 2, 1, 0.5])
@pytest.mark.parametrize("n_neighbors", [1, 10, 100, 1000])
@pytest.mark.parametrize("chunk_size", [2 ** i for i in range(9, 13)])
@pytest.mark.parametrize("strategy", ["chunk_on_train", "chunk_on_test"])
def test_float32_against_float64(
    n_samples, n_features, ratio_train_test, n_neighbors, chunk_size, strategy
):
    if n_samples < n_neighbors:
        pytest.skip(
            f"Skipping as n_samples (={n_samples}) < n_neighbors (={n_neighbors})",
            allow_module_level=True,
        )

    np.random.seed(1)
    X_train = np.random.rand(int(n_samples * n_features)).reshape((-1, n_features))
    X_test = np.random.rand(int(n_samples * n_features / ratio_train_test)).reshape(
        (-1, n_features)
    )

    knn_float64, _ = parallel_knn(
        X_train.astype(np.float64),
        X_test.astype(np.float64),
        k=n_neighbors,
        strategy=strategy,
    )

    knn_float32, _ = parallel_knn(
        X_train.astype(np.float32),
        X_test.astype(np.float32),
        k=n_neighbors,
        strategy=strategy,
    )

    np.testing.assert_array_equal(knn_float64, knn_float32)
