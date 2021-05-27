import argparse
import glob
import json
import subprocess

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.rcParams["figure.dpi"] = 100


if __name__ == "__main__":
    parser = argparse.ArgumentParser("report")

    parser.add_argument("prefix")

    args = parser.parse_args()

    # TODO: This is ugly, but I haven't found something better.
    commit = (
        str(subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]))
        .replace("b'", "")
        .replace("\\n'", "")
    )

    BENCH_NAME = f"{args.prefix}_{commit}"

    RESULTS_FOLDER = f"benchmarks/results/{BENCH_NAME}"

    df = pd.read_csv(glob.glob(f"{RESULTS_FOLDER}/*.csv")[0])
    with open(glob.glob(f"{RESULTS_FOLDER}/*.json")[0], "r") as json_file:
        env_specs = json.load(json_file)

    cols = [
        "n_samples_train",
        "n_samples_test",
        "n_features",
        "n_neighbors",
        "chunk_size",
    ]
    df[cols] = df[cols].astype(np.uint32)

    df["info"] = df[["chunk_size", "total"]].apply(lambda x: str(tuple(x)), axis=1)

    df_grouped = df.groupby(
        ["n_samples_train", "n_samples_test", "n_features", "n_neighbors"]
    )

    for i, (vals, df) in enumerate(df_grouped):
        # 16:9 ratio
        fig = plt.figure(figsize=(24, 13.5))
        ax = plt.gca()
        splot = sns.barplot(
            y="chunk_size", x="throughput", hue="implementation", data=df, ax=ax
        )
        _ = ax.set_xlabel("Throughput (in GB/s)")
        _ = ax.set_ylabel(
            "Chunk size (number of vectors)"
        )
        _ = ax.tick_params(labelrotation=45)
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
        title = f"NearestNeighbors@{env_specs['commit']} - Euclidean Distance, dtype=np.float64, {df.trial.max() + 1} trials\n"
        title += (
            "n_samples_train=%s - n_samples_test=%s - n_features=%s - n_neighbors=%s"
            % vals
        )
        _ = fig.suptitle(title, fontsize=16)
        plt.savefig(f"{BENCH_NAME}_{i}.pdf", bbox_inches="tight")
