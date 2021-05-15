import numpy as np
from pdist_aggregation import parallel_knn
from viztracer import VizTracer


def main(args=None):
    n = 1e4
    d = 100
    n_neighbors = 100
    chunk_size = 4096
    use_chunks_on_Y = True
    np.random.seed(1)
    Y = np.random.rand(int(n * d)).reshape((-1, d))
    X = np.random.rand(int(n * d // 2)).reshape((-1, d))

    parallel_knn(
        X, Y, k=n_neighbors, chunk_size=chunk_size, use_chunks_on_Y=use_chunks_on_Y
    )


if __name__ == "__main__":
    main()
