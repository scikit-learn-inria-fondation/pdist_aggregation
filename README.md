# Experiments on pairwise distances aggregations
[See benchmark results](https://scikit-learn-inria-fondation.github.io/pdist_aggregation/)

## `tl;dr`

```bash
git clone git@github.com:scikit-learn-inria-fondation/pdist_aggregation.git
cd pdist_aggregation
make
```

## Setup

Simply:
```bash
git clone git@github.com:scikit-learn-inria-fondation/pdist_aggregation.git
cd pdist_aggregation
conda env create -f environment.yml
conda activate pdist_aggregation
```

See:
```bash
make help
```

## Running benchmarks on GNU/Linux

You can adapt the benchmarks' configuration editing
[`benchmarks/config.yml`](benchmarks/config.yml).

And then simply run the benchmark script:
```bash
make benchmark-parallel
```

PDF reports will be written in a subfolder in `results`.

The implementation can be capped to the sequential execution using:
```bash
make benchmark-sequential
```

> ⚠ Currently this make target has been written for GNU/Linux as it makes uses
> of `taskset(1)` but you can adapt it easily using environment variables.
