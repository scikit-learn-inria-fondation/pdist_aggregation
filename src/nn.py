import abc

from sklearn.metrics._pairwise_distances_reduction import ArgKmin as _ArgKmin
from sklearn.utils.validation import check_array
from threadpoolctl import threadpool_limits

from pdist_aggregation import _argkmin


class ArgKmin:
    """Adaptor to benchmark scikit-learn#20254's implementations."""
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


class OldArgKmin:
    """Adaptor to benchmarks previous 'flattened' implementation implemented
    in this project."""
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


class OldArgKminParallelY(OldArgKmin):
    def _strategy(self):
        return "chunk_on_Y"


class OldArgKminParallelX(OldArgKmin):
    def _strategy(self):
        return "chunk_on_X"


class OldArgKminParallelAuto(OldArgKmin):
    def _strategy(self):
        return "auto"
