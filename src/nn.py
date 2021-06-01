import abc

from sklearn.utils.validation import check_array
from threadpoolctl import threadpool_limits

from pdist_aggregation import parallel_knn


class NearestNeighbors:
    def __init__(self, n_neighbors=1, algorithm="brute"):
        self.n_neighbors = n_neighbors
        self.algorithm = algorithm
        self.n_neighbors = n_neighbors

    def fit(self, X):
        self._X_train = X
        return self

    @abc.abstractmethod
    def _strategy(self):
        pass

    def kneighbors(self, X, chunk_size=4096, return_distance=False):
        X = check_array(X, order="C")
        # Avoid thread over-subscription by BLAS
        with threadpool_limits(limits=1, user_api="blas"):
            return parallel_knn(
                self._X_train,
                X,
                k=self.n_neighbors,
                chunk_size=chunk_size,
                strategy=self._strategy(),
                return_distance=return_distance,
            )


class NearestNeighborsParallelXtrain(NearestNeighbors):
    def _strategy(self):
        return "chunk_on_train"


class NearestNeighborsParallelXtest(NearestNeighbors):
    def _strategy(self):
        return "chunk_on_test"


class NearestNeighborsParallelAuto(NearestNeighbors):
    def _strategy(self):
        return "auto"
