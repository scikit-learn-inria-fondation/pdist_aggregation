import argparse
import glob
import importlib
import json
import os
import subprocess
import time
from pprint import pprint

import numpy as np
import pandas as pd
import seaborn as sns
import threadpoolctl
import yaml
from matplotlib import pyplot as plt
from memory_profiler import memory_usage
from sklearn import set_config

# Be gentle with eyes
plt.rcParams["figure.dpi"] = 200


def get_cache_info():
    """
    Return cache size, cache line size, cache associativity for each
    level in a dict.
    """
    # TODO: getconf provide useful and reference info, but we might need some
    #  adaptation for other OSes. Also ugly but haven't found better.
    cache_info = {}
    for line in (
        str(subprocess.check_output(["getconf", "-a"])).replace("b'", "").split("\\n")
    ):
        info = line.split(" ")
        if "cache" in info[0].lower():
            cache_info[info[0]] = int(info[-1]) if info[-1] != "" else np.nan

    return cache_info


def datastructures_sizes(
    n: int,
    d: int,
    k: int,
    fdtype=np.float64,
    idtype=np.int64,
    parallel_on_xtrain=True,
):
    """Return the sizes of all the datastructure.

    @param n: chunk_size
    @param d: number of features
    @param k: number of neighbours
    @param fdtype: float datatype
    @param idtype: int datatype
    @param parallel_on_xtrain: if True, analyse for the
        implementation parallelising on X_train, else
        analyse for the one on X_test
    """
    # dtype size
    sf = 8 if fdtype == np.float64 else 4
    si = 8 if idtype == np.int64 else 4

    # dist_middle_terms_chunks
    dist_middle_terms_chunks_size = n * n * sf

    # X_train_sq_norms
    X_train_sq_norms_size = n * sf

    # X_train[X_train_start:X_train_end, :] and
    # X_test[X_test_start:X_test_end, :]
    Xc_and_Yc_size = 2 * n * d * sf

    # knn_indices and knn_red_distances
    chunk_heaps_size = n * k * (sf + si)

    # heaps_indices_chunks, needed for synchronisation
    heaps_indices_chunks_size = n * k * si if parallel_on_xtrain else 0

    total = (
        dist_middle_terms_chunks_size
        + X_train_sq_norms_size
        + Xc_and_Yc_size
        + chunk_heaps_size
        + heaps_indices_chunks_size
    )

    datastructures_size = {
        "dist_middle_terms_chunks": dist_middle_terms_chunks_size,
        "X_train_sq_norms": X_train_sq_norms_size,
        "Xc_and_Yc": Xc_and_Yc_size,
        "chunk_heaps": chunk_heaps_size,
        "heaps_indices_chunks": heaps_indices_chunks_size,
        "total": total,
    }
    return datastructures_size


def benchmark(config, results_folder, bench_name):
    datasets = config["datasets"]
    chunk_sizes = config["chunk_size"]
    n_neighbors = config["n_neighbors"]
    estimators = config["estimators"]

    n_trials = config.get("n_trials", 3)
    one_GiB = 1e9
    benchmarks = pd.DataFrame()

    env_specs_file = f"{results_folder}/{bench_name}.json"

    env_specs = dict(
        threadpool_info=threadpoolctl.threadpool_info(),
        commit=commit,
        cache_info=get_cache_info(),
        config=config,
    )

    # We explicitly remove checks on inputs (defined in sklearn, but also
    # used by daal4py and the proposed implementation) as we solely want
    # to compare the implementations.
    set_config(assume_finite=True)

    with open(env_specs_file, "w") as outfile:
        json.dump(env_specs, outfile)

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
                            (nn_instance.kneighbors, (), knn_kwargs),
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

                        structs_sizes = datastructures_sizes(
                            n=chunk_size,
                            d=X_train.shape[1],
                            k=k,
                            fdtype=X_train.dtype,
                            idtype=np.int64,
                            parallel_on_xtrain="train" in class_name.lower(),
                        )

                        row = dict(
                            trial=trial,
                            implementation=name,
                            n_samples_train=ns_train,
                            n_samples_test=ns_test,
                            n_features=nf,
                            chunk_size=chunk_size,
                            n_parallel_chunks=n_parallel_chunks,
                            n_neighbors=k,
                            max_mem_usage=np.max(mem_usage),
                            time_elapsed=time_elapsed,
                            throughput=bytes_processed_data / time_elapsed / one_GiB,
                            **structs_sizes,
                        )

                        benchmarks = benchmarks.append(row, ignore_index=True)
                        pprint(row)
                        print("---")

                        benchmarks.to_csv(
                            f"{results_folder}/{bench_name}.csv",
                            mode="w+",
                            index=False,
                        )

    # Overriding again now that all the dyn. lib. have been loaded
    env_specs["threadpool_info"] = threadpoolctl.threadpool_info()

    with open(env_specs_file, "w") as outfile:
        json.dump(env_specs, outfile)


def report(results_folder, bench_name):
    df = pd.read_csv(glob.glob(f"{results_folder}/*.csv")[0])
    with open(glob.glob(f"{results_folder}/*.json")[0], "r") as json_file:
        env_specs = json.load(json_file)

    cols = [
        "n_samples_train",
        "n_samples_test",
        "n_features",
        "n_neighbors",
        "chunk_size",
    ]

    df[cols] = df[cols].astype(np.uint32)

    # We need string for grouping
    df["chunk_info"] = df.chunk_size.apply(str)

    df_grouped = df.groupby(
        ["n_samples_train", "n_samples_test", "n_features", "n_neighbors"]
    )

    for i, (vals, df) in enumerate(df_grouped):
        # 16:9 ratio
        fig = plt.figure(figsize=(24, 13.5))
        ax = plt.gca()
        splot = sns.barplot(
            y="chunk_info", x="throughput", hue="implementation", data=df, ax=ax
        )
        _ = ax.set_xlabel("Throughput (in GB/s)")
        _ = ax.set_ylabel("Chunk size (number of vectors)")
        _ = ax.tick_params(labelrotation=45)

        # Adding the numerical values of "x" to bar
        for p in splot.patches:
            _ = splot.annotate(
                f"{p.get_width():.4e}",
                (p.get_width(), p.get_y() + p.get_height() / 2),
                ha="center",
                va="center",
                size=10,
                xytext=(0, -12),
                textcoords="offset points",
            )

        title = (
            f"NearestNeighbors@{env_specs['commit']} - "
            f"Euclidean Distance, dtype=np.float64, {df.trial.max() + 1} trials - Bench. Name: {bench_name}\n"
        )
        title += (
            "n_samples_train=%s - n_samples_test=%s - n_features=%s - n_neighbors=%s"
            % vals
        )
        _ = fig.suptitle(title, fontsize=16)
        plt.savefig(f"{results_folder}/{bench_name}_{i}.pdf", bbox_inches="tight")

    # Unifying pdf files into one
    pdf_files = sorted(glob.glob(f"{results_folder}/{bench_name}*.pdf"))
    subprocess.check_output(
        ["pdfunite", *pdf_files, f"{results_folder}/{bench_name}.pdf"]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("benchmark")

    parser.add_argument("bench_name_suffix")

    args = parser.parse_args()
    with open("benchmarks/config.yml", "r") as f:
        config = yaml.full_load(f)

    # TODO: This is ugly, but I haven't found something better.
    commit = (
        str(subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]))
        .replace("b'", "")
        .replace("\\n'", "")
    )

    bench_name = f"{commit}_{args.bench_name_suffix}"
    results_folder = f"benchmarks/results/{bench_name}"
    os.makedirs(results_folder, exist_ok=True)

    print(f"Benchmarking {bench_name}")
    benchmark(config, results_folder, bench_name)
    print(f"Benchmark results wrote in {results_folder}")

    print(f"Reporting results for {bench_name}")
    report(results_folder, bench_name)
    print(f"Reporting results wrote in {results_folder}")
