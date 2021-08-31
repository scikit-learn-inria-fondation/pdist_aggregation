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

    parser.add_argument("bench_name")

    args = parser.parse_args()

    bench_name = args.bench_name
    with open("benchmarks/config.yml", "r") as f:
        config = yaml.full_load(f)

    results_folder = f"benchmarks/results/{bench_name}"
    os.makedirs(results_folder, exist_ok=True)

    print(f"Reporting results for {bench_name}")
    report(results_folder, bench_name)
    print(f"Reporting results wrote in {results_folder}")
