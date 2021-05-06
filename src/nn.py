from pdist_agregation import parallel_knn


class NearestNeighbors:
    def __init__(self, n_neighbors=1, algorithm="brute"):
        self.n_neighbors = n_neighbors
        self.algorithm = algorithm
        self.n_neighbors = n_neighbors

    def fit(self, X):
        self.X_ = X
        return self

    def kneighbors(self, X,
                   chunk_size=4098,
                   use_chunks_on_Y=True,
                   return_distance=False):
        return parallel_knn(X, self.X_,
                            k=self.n_neighbors,
                            chunk_size=chunk_size,
                            use_chunks_on_Y=use_chunks_on_Y)
