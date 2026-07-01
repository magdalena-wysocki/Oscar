"""Aggregate OSCAR test-time-optimization results into a summary table.

Recursively finds ``test_results.json`` files under one or more roots and reports
mean +/- std of Dice and HD95 (per subject and overall). Each ``test_results.json``
is written by ``test_time_optimize.py``.

Usage:
    python scripts/aggregate_results.py ./logs_test            # one or more roots
    python scripts/aggregate_results.py ./logs_test --csv out.csv
"""

import argparse
import glob
import json
import os

import numpy as np


def collect(roots):
    rows = []
    for root in roots:
        for path in glob.glob(os.path.join(root, "**", "test_results.json"), recursive=True):
            try:
                with open(path) as f:
                    data = json.load(f)
            except Exception as e:
                print(f"[skip] {path}: {e}")
                continue
            metrics = data.get("metrics", {})
            dice, hd95 = metrics.get("occ_dice"), metrics.get("occ_hd95")
            if dice is None or hd95 is None:
                continue
            ds = data.get("dataset")
            subject = os.path.basename(ds[0].rstrip("/")) if isinstance(ds, list) and ds else \
                os.path.basename(os.path.dirname(path))
            rows.append({"subject": subject, "dice": float(dice), "hd95": float(hd95),
                         "n_iters": data.get("n_iters_used"), "path": path})
    return rows


def summarize(rows):
    if not rows:
        print("No test_results.json with metrics found.")
        return
    by_subject = {}
    for r in rows:
        by_subject.setdefault(r["subject"], []).append(r)

    print(f"\n{'subject':<28} {'n':>3} {'Dice (mean+/-std)':>22} {'HD95 (mean+/-std)':>22}")
    print("-" * 78)
    for subject in sorted(by_subject):
        d = np.array([r["dice"] for r in by_subject[subject]])
        h = np.array([r["hd95"] for r in by_subject[subject]])
        print(f"{subject:<28} {len(d):>3} {d.mean():>10.4f} +/- {d.std():<7.4f} "
              f"{h.mean():>10.4f} +/- {h.std():<7.4f}")

    d = np.array([r["dice"] for r in rows])
    h = np.array([r["hd95"] for r in rows])
    print("-" * 78)
    print(f"{'OVERALL':<28} {len(d):>3} {d.mean():>10.4f} +/- {d.std():<7.4f} "
          f"{h.mean():>10.4f} +/- {h.std():<7.4f}\n")


def write_csv(rows, csv_path):
    import csv
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["subject", "dice", "hd95", "n_iters", "path"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("roots", nargs="+", help="directories to search for test_results.json")
    parser.add_argument("--csv", type=str, default=None, help="optional CSV output path")
    args = parser.parse_args()
    rows = collect(args.roots)
    summarize(rows)
    if args.csv:
        write_csv(rows, args.csv)
