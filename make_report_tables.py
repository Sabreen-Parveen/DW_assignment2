"""
Turn the result files into the tables and the plot for the report.

    python make_report_tables.py                  # every results/*_table1.csv
    python make_report_tables.py --results results

Prints one Markdown table per dataset (rows: k and m; cells: relative error with
95% CI, and speedup) plus the full-run times, and saves results/error_vs_m.png.
"""

import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
METHODS = [("uniform", "Uniform"), ("lwcs", "LWCS"), ("cs", "CS")]


def cell(row):
    return f"{row.rel_error_pct:.1f}% ± {row.ci95_pct:.1f}, {row.speedup:.1f}x"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default=os.path.join(HERE, "results"))
    args = p.parse_args()

    files = sorted(glob.glob(os.path.join(args.results, "*_table1.csv")))
    if not files:
        raise SystemExit(f"No *_table1.csv in {args.results}. Run reproduce_table1.py first.")

    data = {os.path.basename(f)[: -len("_table1.csv")]: pd.read_csv(f) for f in files}

    for name, df in data.items():
        print(f"\n### {name.upper()}\n")
        print("| k | m | " + " | ".join(label for _, label in METHODS) + " |")
        print("| --- | --- |" + " --- |" * len(METHODS))
        for (k, m), g in df.groupby(["k", "m"]):
            by = {r.method: r for r in g.itertuples()}
            print(f"| {k} | {m:,} | " + " | ".join(cell(by[mt]) if mt in by else "" for mt, _ in METHODS) + " |")
        if "full_time_s" in df:
            for k, g in df.groupby("k"):
                print(f"\nFull k-means++ on {name}, k = {k}: {g.full_time_s.iloc[0]:.1f} s")

    # One panel per (dataset, k): relative error against m, one line per method
    panels = [(name, k, g) for name, df in data.items() for k, g in df.groupby("k")]
    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 3.2), squeeze=False)
    for ax, (name, k, g) in zip(axes[0], panels):
        for mt, label in METHODS:
            s = g[g.method == mt].sort_values("m")
            if len(s):
                ax.errorbar(s.m, s.rel_error_pct, yerr=s.ci95_pct, marker="o", capsize=3, label=label)
        ax.set_title(f"{name.upper()}, k = {k}")
        ax.set_xlabel("sample size m")
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("relative error (%)")
    axes[0][0].legend()
    fig.tight_layout()
    out = os.path.join(args.results, "error_vs_m.png")
    fig.savefig(out, dpi=200)
    print(f"\nplot written: {out}")


if __name__ == "__main__":
    main()
