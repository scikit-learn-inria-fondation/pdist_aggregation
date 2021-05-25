import abc

import numpy as np
from pykeops.numpy import Genred
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
    def _use_chunk_on_train(self):
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
                use_chunk_on_train=self._use_chunk_on_train(),
            )


class NearestNeighborsParrallelXtrain(NearestNeighbors):
    def _use_chunk_on_train(self):
        return True


class NearestNeighborsParrallelXtest(NearestNeighbors):
    def _use_chunk_on_train(self):
        return False


class KeOpsNearestNeighbors(NearestNeighbors):
    """Adapted from this example:
    https://github.com/getkeops/keops/blob/master/pykeops/examples/numpy/plot_test_ArgKMin.py
    """

    def __init__(self, n_neighbors=1, algorithm="brute"):
        self._K = n_neighbors

    def fit(self, X):
        self.X_ = X
        D = X.shape[1]

        formula = "SqDist(x,y)"
        variables = [
            "x = Vi(" + str(D) + ")",
            "y = Vj(" + str(D) + ")",
        ]

        dtype = "float64"
        self._argkmin_routine = Genred(
            formula,
            variables,
            reduction_op="ArgKMin",
            axis=1,
            dtype=dtype,
            opt_arg=self._K,
        )

        # Apparently warming helps
        self._argkmin_routine(
            np.random.rand(10, D).astype(dtype), np.random.rand(10, D).astype(dtype)
        )
        return self

    def kneighbors(self, X, working_memory=4_000_000, return_distance=False):
        return self._argkmin_routine(X, self.X_, backend="auto")
