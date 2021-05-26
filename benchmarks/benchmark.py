import importlib
import subprocess
import time
from pprint import pprint

import numpy as np
import pandas as pd
import yaml
from memory_profiler import memory_usage

with open("benchmarks/config.yml", "r") as f:
    config = yaml.full_load(f)

datasets = config["datasets"]
chunk_sizes = config["chunk_size"]
n_neighbors = config["n_neighbors"]
estimators = config["estimators"]

n_trials = config.get("n_trials", 5)
one_GiB = 1e9
benchmarks = pd.DataFrame()

# TODO: This is ugly, but I haven't found something better.
commit = (
    str(subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]))
    .replace("b'", "")
    .replace("\\n'", "")
)

for dataset in datasets:
    for trial in range(n_trials):
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
                for chunk_size in chunk_sizes if chunk else [0]:
                    nn_instance = estim_class(n_neighbors=k, algorithm="brute").fit(
                        X_train
                    )

                    knn_kwargs = {"X": X_test, "return_distance": False}
                    if chunk:
                        knn_kwargs["chunk_size"] = chunk_size

                    print(
                        f"Start trial #{trial + 1} for: {name}, "
                        f"n_samples_train={ns_train}, "
                        f"n_samples_test={ns_test}, "
                        f"n_features={nf}, "
                        f"n_neighbors={k}"
                    )

                    t0_ = time.perf_counter()
                    mem_usage, knn_res = memory_usage(
                        (nn_instance.kneighbors, knn_kwargs),
                        interval=0.01,
                        retval=True,
                        include_children=True,
                        multiprocess=True,
                    )
                    t1_ = time.perf_counter()
                    time_elapsed = round(t1_ - t0_, 5)

                    # Parallel_knn returns n_chunks run in parallel
                    # We report it in the benchmarks results
                    n_parallel_chunks = (
                        knn_res[1] if isinstance(knn_res, tuple) else np.nan
                    )

                    row = dict(
                        trial=trial,
                        implementation=name,
                        n_samples_train=ns_train,
                        n_samples_test=ns_test,
                        n_features=nf,
                        chunk_info=(chunk_size, n_parallel_chunks),
                        n_neighbors=k,
                        max_mem_usage=np.max(mem_usage),
                        time_elapsed=time_elapsed,
                        throughput=bytes_processed_data / time_elapsed / one_GiB,
                    )
                    benchmarks = benchmarks.append(row, ignore_index=True)
                    pprint(row)
                    print("---")

                    benchmarks.to_csv(
                        f"benchmarks/results/results_{commit}.csv",
                        mode="w+",
                        index=False,
                    )
