import abc

from sklearn.metrics._pairwise_distances_reduction import ArgKmin as _ArgKmin
from sklearn.utils.validation import check_array
from threadpoolctl import threadpool_limits

from pairwise_aggregation import _argkmin


class NearestNeighbors:
    def __init__(self, n_neighbors=1, algorithm="brute"):
        self.n_neighbors = n_neighbors
        self.algorithm = algorithm

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
            return _argkmin(
                self._X_train,
                X,
                k=self.n_neighbors,
                chunk_size=chunk_size,
                strategy=self._strategy(),
                return_distance=return_distance,
            )


class NearestNeighborsParallelY(NearestNeighbors):
    def _strategy(self):
        return "chunk_on_Y"


class NearestNeighborsParallelX(NearestNeighbors):
    def _strategy(self):
        return "chunk_on_X"


class NearestNeighborsParallelAuto(NearestNeighbors):
    def _strategy(self):
        return "auto"


class ArgKmin:
    def __init__(self, n_neighbors=1, algorithm="brute"):
        self.n_neighbors = n_neighbors
        self.algorithm = algorithm

    def fit(self, X):
        self._X_train = X
        return self

    @abc.abstractmethod
    def _strategy(self):
        pass

    def kneighbors(self, X, chunk_size=4096, return_distance=False):
        # Checks on arrays are done at the initialisation here
        # triggered by this call
        argkmin = _ArgKmin.get_for(X=X, Y=self._X_train,
                                   k=self.n_neighbors,
                                   chunk_size=chunk_size)

        # Avoid thread over-subscription by BLAS
        with threadpool_limits(limits=1, user_api="blas"):
            return argkmin.compute(strategy=self._strategy(),
                                   return_distance=return_distance)


class ArgKminParallelY(ArgKmin):
    def _strategy(self):
        return "parallel_on_Y"


class ArgKminParallelX(ArgKmin):
    def _strategy(self):
        return "parallel_on_X"


class ArgKminParallelAuto(ArgKmin):
    def _strategy(self):
        return "auto"
