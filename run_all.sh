#!/usr/bin/env bash
# Run the full reproduction on SONG and RNA, then build the report tables.
# Usage (from the repository root):  bash run_all.sh            # 10 reps, as in the report
#                          REPS=50 bash run_all.sh    # the paper's 50 reps (much slower)
set -euo pipefail
cd "$(dirname "$0")"
REPS="${REPS:-10}"

python datasets/download_datasets.py song rna

for ds in song rna; do
    python reproduce_table1.py --dataset "$ds" --k 100 500 --m 1000 2000 5000 --reps "$REPS"
done

# KDD runs too if you have bio_train.dat in datasets/
if [ -f datasets/bio_train.dat ]; then
    python reproduce_table1.py --dataset kdd --k 100 500 --m 1000 2000 5000 --reps "$REPS"
fi

python make_report_tables.py
