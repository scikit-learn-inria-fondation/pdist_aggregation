import numpy as np
import pytest
from sklearn.neighbors import NearestNeighbors

from pdist_aggregation import parallel_knn


@pytest.mark.parametrize("n", [10 ** i for i in [2, 3, 4]])
@pytest.mark.parametrize("d", [2, 5, 10, 100])
@pytest.mark.parametrize("ratio_train_test", [10, 2, 1, 0.5])
@pytest.mark.parametrize("n_neighbors", [1, 10, 100, 1000])
@pytest.mark.parametrize("chunk_size", [2 ** i for i in range(9, 13)])
@pytest.mark.parametrize("strategy", ["auto", "chunk_on_train", "chunk_on_test"])
def test_against_sklearn(
    n,
    d,
    ratio_train_test,
    n_neighbors,
    chunk_size,
    strategy,
    dtype=np.float64,
):
    if n < n_neighbors:
        pytest.skip(
            f"Skipping as n (={n}) < n_neighbors (={n_neighbors})",
            allow_module_level=True,
        )

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
        strategy=strategy,
    )

    np.testing.assert_array_equal(knn, knn_sk)


@pytest.mark.parametrize(
    "translation", [10 ** i for i in [2, 3, 4, 5, 6, 7, 8, 9, 10, 12]]
)
@pytest.mark.parametrize("n", [1000])
@pytest.mark.parametrize("d", [10, 100, 1000])
@pytest.mark.parametrize("n_neighbors", [100])
@pytest.mark.parametrize("strategy", ["chunk_on_train", "chunk_on_test"])
def test_far_from_origin(
    translation,
    n,
    d,
    n_neighbors,
    strategy,
    dtype=np.float64,
):
    """Testing mitigation for points far from the origin"""

    np.random.seed(1)
    X_train = np.random.rand(int(n * d)).astype(dtype).reshape((-1, d))
    X_test = np.random.rand(int(n * d)).astype(dtype).reshape((-1, d))

    # Computation should be precise here (X_train and X_test are closed
    # to the origin)
    knn, _ = parallel_knn(
        X_train,
        X_test,
        k=n_neighbors,
    )

    # Translating to make them far from the origin entering the regime
    # with potentially bad precision
    X_train += (translation,)
    X_test += (translation,)

    # As array get modified by the implementation, we use the sum as
    # a proxy for their identity test (see bellow)
    sum_original_X_train = X_train.sum()
    sum_original_X_test = X_test.sum()

    # No translation
    knn_translated, _ = parallel_knn(
        X_train,
        X_test,
        k=n_neighbors,
    )

    np.testing.assert_array_equal(knn, knn_translated)
    np.testing.assert_array_equal(sum_original_X_train, X_train.sum())
    np.testing.assert_array_equal(sum_original_X_test, X_test.sum())
