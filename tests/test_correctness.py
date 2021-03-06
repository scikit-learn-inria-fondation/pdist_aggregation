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
def test_correctness(
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

    sk_knn_distances, sk_knn_indices = neigh.kneighbors(X_test, return_distance=True)
    (new_knn_distances, new_knn_indices), _ = parallel_knn(
        X_train,
        X_test,
        k=n_neighbors,
        chunk_size=chunk_size,
        strategy=strategy,
        return_distance=True,
    )

    np.testing.assert_almost_equal(new_knn_distances, sk_knn_distances)
    np.testing.assert_array_equal(new_knn_indices, sk_knn_indices)
