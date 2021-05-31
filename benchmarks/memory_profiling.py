#!/usr/bin/env python
import argparse
import importlib

import numpy as np

"""Simple memory profiling script intended to be used with mprof"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser("memory_profiling", description=__doc__)
    parser.add_argument("estimator")
    parser.add_argument("--n", type=int, default=1e3)
    parser.add_argument("--d", type=int, default=100)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--chunk_size", type=int, default=0)
    parser.add_argument("--ratio_train_test", type=float, default=0.5)

    args = parser.parse_args()
    n_train = int(args.n)
    n_test = int(n_train / args.ratio_train_test)

    X_train = np.random.rand(n_train, args.d)
    X_test = np.random.rand(n_test, args.d)

    splitted_path = args.estimator.split(".")
    module, class_name = ".".join(splitted_path[:-1]), splitted_path[-1]
    estim_class = getattr(importlib.import_module(module), class_name)

    nn_instance = estim_class(n_neighbors=args.k, algorithm="brute").fit(X_train)

    knn_kwargs = {"X": X_test, "return_distance": False}
    if args.chunk_size != 0:
        knn_kwargs["chunk_size"] = args.chunk_size

    knn_res = nn_instance.kneighbors(**knn_kwargs)
