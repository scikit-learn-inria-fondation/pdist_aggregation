# Experiments on pairwise distances aggregations
[See benchmark results](https://scikit-learn-inria-fondation.github.io/pdist_aggregation/)

## Setup

Simply:
```bash
git clone git@github.com:scikit-learn-inria-fondation/pdist_aggregation.git
cd pdist_aggregation
conda env create -f environment.yml
```

## Running benchmarks

You can adapt the benchmarks' configuration editing
[`benchmarks/config.yml`](benchmarks/config.yml).

And then simply run the benchmark script:
```bash
python benchmarks/benchmark.py
```

### Selecting the number of threads

Note that the implementation is parallelised using OpemMP threads pool on
all the core of your machines.

You can specify the number of threads setting the `OMP_NUM_THREADS`, e.g:
```bash
OMP_NUM_THREADS=2 python benchmarks/benchmark.py
```

If running GNU/Linux, [`taskset(1)`](https://www.man7.org/linux/man-pages/man1/taskset.1.html)
might probably be  the best alternative:

```bash
# Selecting the core that you want to use.
taskset -c 0, 2 python benchmarks/benchmark.py
```

### Avoiding threads over-subscription

To avoid threads' over-subscription by BLAS, you can cap the number of
threads to use to 1:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 python benchmarks/benchmark.py
```
