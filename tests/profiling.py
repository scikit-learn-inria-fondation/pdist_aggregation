import numpy as np

from pdist_aggregation import parallel_knn


def main(args=None):
    d = 50
    n_neighbors = 100
    chunk_size = 256
    np.random.seed(1)
    X_train = np.random.rand(int(1000 * d)).reshape((-1, d))
    X_test = np.random.rand(int(1000000 * d // 2)).reshape((-1, d))
    parallel_knn(
        X_train, X_test,
        k=n_neighbors,
        chunk_size=chunk_size,
        use_chunk_on_train=True,
    )


if __name__ == "__main__":
    main()
