import numpy as np
from pdist_agregation import parallel_knn

def main(args=None):
    n = 1e4
    d = 100
    n_neighbors = 100
    working_memory = 4_000_000
    use_chunks_on_Y = True
    np.random.seed(1)
    Y = np.random.rand(int(n * d)).reshape((-1, d))
    X = np.random.rand(int(n * d // 2)).reshape((-1, d))

    parallel_knn(X, Y,
                 k=n_neighbors,
                 working_memory=working_memory,
                 use_chunks_on_Y=use_chunks_on_Y)

if __name__ == "__main__":
    main()
