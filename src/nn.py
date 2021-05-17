import numpy as np
from pykeops.numpy import Genred
from sklearn.utils.validation import check_array

from pdist_aggregation import parallel_knn


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

    def kneighbors(self, X, working_memory=4_000_000, return_distance=False):
        X = check_array(X, order="C")
        return parallel_knn(
            X,
            self.X_,
            k=self.n_neighbors,
            working_memory=working_memory,
            use_chunks_on_Y=self.use_chunks_on_Y,
        )


class NearestNeighborsSingleChunking(NearestNeighbors):
    @property
    def use_chunks_on_Y(self):
        return False


class NearestNeighborsDoubleChunking(NearestNeighbors):
    @property
    def use_chunks_on_Y(self):
        return True


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
