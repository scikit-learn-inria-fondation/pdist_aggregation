import numpy as np

from pdist_aggregation import parallel_knn


def main(args=None):
    n = 1e4
    d = 100
    n_neighbors = 100
    chunk_size = 1024
    np.random.seed(1)
    X_train = np.random.rand(int(n * d)).reshape((-1, d))
    X_test = np.random.rand(int(n * d // 2)).reshape((-1, d))

    parallel_knn(X_train, X_test, k=n_neighbors, chunk_size=chunk_size)


if __name__ == "__main__":
    main()
