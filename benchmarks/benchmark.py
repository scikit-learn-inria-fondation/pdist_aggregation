import importlib
import itertools
import time
from pprint import pprint

import numpy as np
import pandas as pd
import yaml
from scipy.sparse import data
from sklearn.neighbors import NearestNeighbors
from sklearnex import patch_sklearn

from distances import knearest_neighbors

with open("config.yml", "r") as f:
    config = yaml.full_load(f)

datasets = config["datasets"]
chunk_sizes = config["chunk_sizes"]
estimators = config["estimators"]

benchmarks = pd.DataFrame()

for estimator in estimators:
    name, estimator, chunk = estimator.values()
    splitted_path = estimator.split(".")
    module, class_name = ".".join(splitted_path[:-1]), splitted_path[-1]
    estim_class = getattr(importlib.import_module(module), class_name)

    for dataset in datasets:
        dataset = {k: int(float(v)) for k, v in dataset.items()}
        ns_train, ns_test, nf = dataset.values()
        X_train = np.random.rand(ns_train, nf)
        X_test = np.random.rand(ns_test, nf)
        bytes_processed_data = X_train.nbytes + X_test.nbytes

        if chunk:
            for chunk_size in chunk_sizes:
                nn_instance = estim_class(n_neighbors=1, algorithm="brute").fit(X_train)

                t0_ = time.perf_counter()
                nn_instance.kneighbors(
                    X_test, return_distance=False, chunk_size=chunk_size
                )
                t1_ = time.perf_counter()
                time_elapsed = round(t1_ - t0_, 5)

                row = dict(
                    implementation=name,
                    n_samples_train=ns_train,
                    n_samples_test=ns_test,
                    n_features=nf,
                    chunk_size=chunk_size,
                )
                row["time_elapsed"] = time_elapsed
                row["throughput"] = bytes_processed_data / time_elapsed / 1e9
                benchmarks = benchmarks.append(row, ignore_index=True)
                pprint(row)
                print("---")
        else:
            nn_instance = estim_class(n_neighbors=1, algorithm="brute").fit(X_train)
            t0_ = time.perf_counter()
            nn_instance.kneighbors(X_test, return_distance=False)
            t1_ = time.perf_counter()
            time_elapsed = round(t1_ - t0_, 5)

            row = dict(
                implementation=name,
                n_samples_train=ns_train,
                n_samples_test=ns_test,
                n_features=nf,
                chunk_size=0,
            )
            row["time_elapsed"] = time_elapsed
            row["throughput"] = bytes_processed_data / time_elapsed / 1e9
            benchmarks = benchmarks.append(row, ignore_index=True)
            pprint(row)
            print("---")

benchmarks.to_csv(
    "benchmarks/results.csv",
    mode="w+",
    index=False,
)
