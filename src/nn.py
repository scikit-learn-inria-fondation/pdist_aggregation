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
    def use_chunk_on_train(self):
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
                use_chunk_on_train=self._use_chunk_on_train,
            )


class NearestNeighborsParrallelXtrain(NearestNeighbors):
    def use_chunk_on_train(self):
        return True


class NearestNeighborsParrallelXtest(NearestNeighbors):
    def use_chunk_on_train(self):
        return False
