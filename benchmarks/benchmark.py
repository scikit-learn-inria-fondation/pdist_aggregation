import importlib
import time
from pprint import pprint

import numpy as np
import pandas as pd
import yaml

with open("benchmarks/config.yml", "r") as f:
    config = yaml.full_load(f)

datasets = config["datasets"]
working_memory_range = config["working_memory_range"]
n_neighbors = config["n_neighbors"]
estimators = config["estimators"]

N_TRIALS = 5
one_GiB = 1e9
benchmarks = pd.DataFrame()


for dataset in datasets:
    for trial in range(N_TRIALS):
        dataset = {k: int(float(v)) for k, v in dataset.items()}
        ns_train, ns_test, nf = dataset.values()
        X_train = np.random.rand(ns_train, nf)
        X_test = np.random.rand(ns_test, nf)
        bytes_processed_data = X_train.nbytes + X_test.nbytes

        for estimator in estimators:
            name, estimator, chunk = estimator.values()
            splitted_path = estimator.split(".")
            module, class_name = ".".join(splitted_path[:-1]), splitted_path[-1]
            estim_class = getattr(importlib.import_module(module), class_name)

            for k in n_neighbors:
                for working_memory in working_memory_range if chunk else [0]:
                    nn_instance = estim_class(n_neighbors=k, algorithm="brute").fit(
                        X_train
                    )

                    knn_kwargs = {"X": X_test, "return_distance": False}
                    if chunk:
                        knn_kwargs["working_memory"] = working_memory

                    t0_ = time.perf_counter()
                    knn_res = nn_instance.kneighbors(**knn_kwargs)
                    t1_ = time.perf_counter()
                    time_elapsed = round(t1_ - t0_, 5)

                    # Parallel_knn returns the size of samples at run time
                    # We report it in the benchmarks results
                    n = knn_res[1] if isinstance(knn_res, tuple) else np.nan

                    row = dict(
                        trial=trial,
                        implementation=name,
                        n_samples_train=ns_train,
                        n_samples_test=ns_test,
                        n_features=nf,
                        working_memory=(working_memory, n),
                        n_neighbors=k,
                    )
                    row["time_elapsed"] = time_elapsed
                    row["throughput"] = bytes_processed_data / time_elapsed / one_GiB
                    benchmarks = benchmarks.append(row, ignore_index=True)
                    pprint(row)
                    print("---")

                    benchmarks.to_csv(
                        "benchmarks/results/results.csv",
                        mode="w+",
                        index=False,
                    )
