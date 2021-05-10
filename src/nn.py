from pdist_agregation import parallel_knn
from sklearn.utils.validation import check_array

class NearestNeighbors:
    def __init__(self, n_neighbors=1, algorithm="brute"):
        self.n_neighbors = n_neighbors
        self.algorithm = algorithm
        self.n_neighbors = n_neighbors

    def fit(self, X):
        self.X_ = X
        return self

    @property
    def use_chunks_on_Y(self):
        raise NotImplementedError()

    def kneighbors(self, X,
                   working_memory=4_000_000,
                   return_distance=False):
        X = check_array(X, order="C")
        return parallel_knn(X, self.X_,
                            k=self.n_neighbors,
                            working_memory=working_memory,
                            use_chunks_on_Y=self.use_chunks_on_Y)

class NearestNeighborsSingleChunking(NearestNeighbors):
    @property
    def use_chunks_on_Y(self):
        return False

class NearestNeighborsDoubleChunking(NearestNeighbors):
    @property
    def use_chunks_on_Y(self):
        return True
