"""Batch-preprocess every sweep under a data directory.

For each subject folder this rebuilds the two *derived* files from the shipped
``poses.npy`` / ``images.npy`` / ``masks_gt.npy`` (and ``transform.npy`` for
translation-augmented sweeps). It does not reimplement anything: it just calls
the single-subject functions already defined in this folder.

  poses_labels.npy   <-  create_poses_labels.create_poses_labels
  gt_volume.npy      <-  create_gt_volume.create_gt_volume

A folder is processed only if it contains ``poses.npy``. With ``--skip_existing``
folders that already have both derived files are left untouched.

Usage:
    python scripts/preprocess_all.py --data_root ./data
    python scripts/preprocess_all.py --data_root ./data --skip_existing
"""

import argparse
import os
import sys

# Make the sibling scripts importable regardless of the working directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from create_poses_labels import create_poses_labels
from create_gt_volume import create_gt_volume


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data_root", type=str, required=True,
                        help="directory containing one subdirectory per sweep")
    parser.add_argument("--grid_res", type=int, default=128,
                        help="GT occupancy grid resolution (default 128)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="occupied/observed ratio to mark a voxel occupied (default 0.5)")
    parser.add_argument("--probe_depth", type=float, default=110.0,
                        help="probe imaging depth in mm used during simulation (default 110)")
    parser.add_argument("--probe_width", type=float, default=60.0,
                        help="probe width in mm used during simulation (default 60)")
    parser.add_argument("--skip_existing", action="store_true",
                        help="skip folders that already have poses_labels.npy and gt_volume.npy")
    args = parser.parse_args()

    subjects = sorted(
        name for name in os.listdir(args.data_root)
        if os.path.isfile(os.path.join(args.data_root, name, "poses.npy"))
    )
    if not subjects:
        sys.exit(f"No subject folders with poses.npy found under {args.data_root}")

    print(f"Preprocessing {len(subjects)} sweeps under {args.data_root}\n")
    for i, name in enumerate(subjects, 1):
        d = os.path.join(args.data_root, name)
        print(f"[{i}/{len(subjects)}] {name}")
        have_both = os.path.isfile(os.path.join(d, "poses_labels.npy")) and \
                    os.path.isfile(os.path.join(d, "gt_volume.npy"))
        if args.skip_existing and have_both:
            print("  skip (derived files already present)")
            continue
        create_poses_labels(d, args.probe_depth, args.probe_width)
        create_gt_volume(d, args.grid_res, args.threshold)
        print()

    print(f"Done: {len(subjects)} sweeps preprocessed.")


if __name__ == "__main__":
    main()
