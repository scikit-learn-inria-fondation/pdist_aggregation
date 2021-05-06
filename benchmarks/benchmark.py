import importlib
import time
from pprint import pprint

import numpy as np
import pandas as pd
import yaml

with open("config.yml", "r") as f:
    config = yaml.full_load(f)

datasets = config["datasets"]
CHUNK_SIZES = config["chunk_sizes"]
n_neighbors = config["n_neighbors"]
estimators = config["estimators"]

N_TRIALS = 10
one_GiB = 1e9
benchmarks = pd.DataFrame()

for estimator in estimators:
    name, estimator, chunk = estimator.values()
    splitted_path = estimator.split(".")
    module, class_name = ".".join(splitted_path[:-1]), splitted_path[-1]
    estim_class = getattr(importlib.import_module(module), class_name)

    for dataset in datasets:
        for trial in range(N_TRIALS):
            dataset = {k: int(float(v)) for k, v in dataset.items()}
            ns_train, ns_test, nf = dataset.values()
            X_train = np.random.rand(ns_train, nf)
            X_test = np.random.rand(ns_test, nf)
            bytes_processed_data = X_train.nbytes + X_test.nbytes

            for k in n_neighbors:
                chunk_sizes = CHUNK_SIZES if chunk else [0]
                for chunk_size in chunk_sizes:
                    nn_instance = estim_class(n_neighbors=k,
                                              algorithm="brute").fit(X_train)

                    knn_kwargs = {
                        "X": X_test,
                        "return_distance": False
                    }
                    if chunk:
                        knn_kwargs["chunk_size"] = chunk_size

                    t0_ = time.perf_counter()
                    nn_instance.kneighbors(**knn_kwargs)
                    t1_ = time.perf_counter()
                    time_elapsed = round(t1_ - t0_, 5)

                    row = dict(
                        trial=trial,
                        implementation=name,
                        n_samples_train=ns_train,
                        n_samples_test=ns_test,
                        n_features=nf,
                        chunk_size=chunk_size,
                        n_neighbors=k,
                    )
                    row["time_elapsed"] = time_elapsed
                    row["throughput"] = bytes_processed_data / time_elapsed / one_GiB
                    benchmarks = benchmarks.append(row, ignore_index=True)
                    pprint(row)
                    print("---")

benchmarks.to_csv(
    "results.csv",
    mode="w+",
    index=False,
)
