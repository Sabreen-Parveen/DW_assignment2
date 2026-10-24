"""
Reproduce Table 1 of Bachem, Lucic & Krause (KDD 2018): relative error and
speedup of k-means++ on a coreset (Uniform, LWCS, CS) versus on the full data.

relative error eta = (phi_X(Q_coreset) - phi_X(Q_full)) / phi_X(Q_full)
speedup           = time(full k-means) / time(coreset construction + k-means on coreset)

Usage examples:
    python reproduce_table1.py --dataset synthetic --k 100 --reps 5
    python reproduce_table1.py --dataset kdd --k 100 500
    python reproduce_table1.py --dataset song --k 100
    python reproduce_table1.py --dataset rna --k 100 500

Real datasets are read from datasets/ (see datasets/download_datasets.py);
pass --data-path to read them from somewhere else.
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import load_svmlight_file, make_blobs

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PATHS = {
    "kdd": os.path.join(HERE, "datasets", "bio_train.dat"),
    "song": os.path.join(HERE, "datasets", "YearPredictionMSD.txt"),
    "rna": os.path.join(HERE, "datasets", "cod-rna.all"),
}

from coresets import kmeans_cost, lightweight_coreset, sensitivity_coreset, uniform_sample


def load_dataset(name, path):
    """Return X as float64. Labels and IDs are dropped: only the features are clustered."""
    if name == "synthetic":
        # Imbalanced clusters (Zipf-like sizes, varied spread): uniform sampling
        # misses the small clusters, which is where coresets should help
        rng = np.random.default_rng(0)
        sizes = (200_000 / np.arange(1, 51) ** 1.5)
        sizes = np.maximum((sizes / sizes.sum() * 200_000).astype(int), 20)
        X, _ = make_blobs(n_samples=sizes, n_features=20, center_box=(-100, 100),
                          cluster_std=rng.uniform(0.5, 5.0, size=50), random_state=0)
        return X
    path = path or DEFAULT_PATHS.get(name)
    if path is None or not os.path.exists(path):
        raise SystemExit(f"Data file for {name} not found ({path}). "
                         "Run datasets/download_datasets.py or pass --data-path.")
    if name == "kdd":
        # KDD Cup 2004 protein homology, bio_train.dat: 145,751 rows; columns 1-3 are
        # block id, example id and label, the remaining 74 are features
        return np.loadtxt(path)[:, 3:]
    if name == "song":
        # UCI YearPredictionMSD.txt: 515,345 rows; first column is the year (label)
        return pd.read_csv(path, header=None).values[:, 1:].astype(np.float64)
    if name == "rna":
        # LIBSVM cod-rna: cod-rna + cod-rna.t + cod-rna.r concatenated = 488,565 x 8
        X, _ = load_svmlight_file(path)
        return X.toarray()
    if name == "csv":
        # Any numeric CSV, e.g. the CSN accelerometer data
        return pd.read_csv(path).select_dtypes("number").values.astype(np.float64)
    raise ValueError(name)


def run_kmeans(X, k, seed, weights=None, max_iter=300):
    km = KMeans(n_clusters=k, init="k-means++", n_init=1, max_iter=max_iter, random_state=seed)
    km.fit(X, sample_weight=weights)
    return km.cluster_centers_


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["synthetic", "kdd", "song", "rna", "csv"], default="synthetic")
    p.add_argument("--data-path", default=None)
    p.add_argument("--k", type=int, nargs="+", default=[100, 500])
    p.add_argument("--m", type=int, nargs="+", default=[1000, 2000, 5000])
    p.add_argument("--methods", nargs="+", default=["uniform", "lwcs", "cs"])
    p.add_argument("--reps", type=int, default=10, help="paper uses 50")
    p.add_argument("--full-reps", type=int, default=3, help="k-means++ runs on the full data")
    p.add_argument("--max-iter", type=int, default=300)
    p.add_argument("--out", default=os.path.join(HERE, "results"))
    args = p.parse_args()

    X = load_dataset(args.dataset, args.data_path)
    print(f"{args.dataset}: n={X.shape[0]}, d={X.shape[1]}")
    os.makedirs(args.out, exist_ok=True)

    rows = []
    for k in args.k:
        # Reference: k-means++ on the full data; take the mean cost and mean time
        full_costs, full_times = [], []
        for r in range(args.full_reps):
            t0 = time.perf_counter()
            Q = run_kmeans(X, k, seed=r, max_iter=args.max_iter)
            full_times.append(time.perf_counter() - t0)
            full_costs.append(kmeans_cost(X, Q))
        full_cost, full_time = np.mean(full_costs), np.mean(full_times)
        print(f"k={k}: full cost {full_cost:.4g}, time {full_time:.1f} s")

        for m in args.m:
            for method in args.methods:
                errs, speedups, times = [], [], []
                for r in range(args.reps):
                    rng = np.random.default_rng(1000 + r)
                    t0 = time.perf_counter()
                    if method == "uniform":
                        C, w = uniform_sample(X, m, rng)
                    elif method == "lwcs":
                        C, w = lightweight_coreset(X, m, rng)
                    else:
                        C, w = sensitivity_coreset(X, m, k, rng)
                    Q = run_kmeans(C, k, seed=r, weights=w, max_iter=args.max_iter)
                    elapsed = time.perf_counter() - t0
                    errs.append((kmeans_cost(X, Q) - full_cost) / full_cost)
                    speedups.append(full_time / elapsed)
                    times.append(elapsed)
                errs = np.array(errs)
                ci = 1.96 * errs.std(ddof=1) / np.sqrt(len(errs)) if len(errs) > 1 else 0.0
                rows.append({"k": k, "m": m, "method": method,
                             "rel_error_pct": 100 * errs.mean(), "ci95_pct": 100 * ci,
                             "speedup": np.mean(speedups), "time_s": np.mean(times),
                             "full_time_s": full_time})
                print(f"  m={m:>6} {method:>7}: error {100 * errs.mean():6.1f}% "
                      f"(±{100 * ci:.1f}), speedup {np.mean(speedups):7.1f}x")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, f"{args.dataset}_table1.csv"), index=False)
    with open(os.path.join(args.out, f"{args.dataset}_run.json"), "w") as f:
        json.dump(vars(args), f, indent=2)


if __name__ == "__main__":
    main()
