"""Build a ground-truth 3D occupancy volume (``gt_volume.npy``) for a sweep.

For every occupied pixel of every frame, its precomputed [0, 1]^3 coordinate is
splatted into a ``grid_res^3`` grid; a voxel is marked occupied if the ratio of
occupied to observed hits exceeds ``--threshold``. The resulting volume is what
``test_time_optimize.py`` / validation compare predictions against (Dice, HD95).

Inputs (in ``--data_dir``):  poses_labels.npy [N, w, h, 3], masks_gt.npy [N, H, W]
Output (in ``--data_dir``):  gt_volume.npy [grid_res, grid_res, grid_res] uint8

Usage:
    python scripts/create_gt_volume.py --data_dir ./data/<subject>
"""

import argparse
import os

import numpy as np
from tqdm import trange


def create_gt_volume(data_dir, grid_res=128, threshold=0.5, save=True):
    poses_labels = np.load(os.path.join(data_dir, "poses_labels.npy"), mmap_mode="r")
    masks_gt = np.load(os.path.join(data_dir, "masks_gt.npy"), mmap_mode="r")
    print(f"poses_labels: {poses_labels.shape}, masks_gt: {masks_gt.shape}")

    occ_count = np.zeros((grid_res, grid_res, grid_res), dtype=np.float64)
    hit_count = np.zeros((grid_res, grid_res, grid_res), dtype=np.float64)

    for i in trange(poses_labels.shape[0], desc="Building GT volume"):
        pts = poses_labels[i].astype(np.float32)   # [w, h, 3]
        mask = masks_gt[i].astype(np.float32)      # [H, W]
        # masks_gt is stored transposed relative to poses_labels.
        if mask.shape != pts.shape[:2]:
            mask = mask.T
        if mask.shape != pts.shape[:2]:
            raise ValueError(f"Frame {i}: mask {mask.shape} != pts {pts.shape[:2]}")

        all_idx = np.clip((pts.reshape(-1, 3) * (grid_res - 1)).astype(np.int64), 0, grid_res - 1)
        np.add.at(hit_count, (all_idx[:, 0], all_idx[:, 1], all_idx[:, 2]), 1.0)

        occupied = mask > 0.5
        if occupied.any():
            idx = np.clip((pts[occupied] * (grid_res - 1)).astype(np.int64), 0, grid_res - 1)
            np.add.at(occ_count, (idx[:, 0], idx[:, 1], idx[:, 2]), 1.0)

    volume = np.zeros((grid_res, grid_res, grid_res), dtype=np.uint8)
    observed = hit_count > 0
    ratio = np.zeros_like(occ_count)
    ratio[observed] = occ_count[observed] / hit_count[observed]
    volume[ratio > threshold] = 1
    print(f"Final volume: {volume.sum()} occupied / {observed.sum()} observed voxels")

    if save:
        out_path = os.path.join(data_dir, "gt_volume.npy")
        np.save(out_path, volume)
        print(f"Saved {out_path}")
    return volume


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--grid_res", type=int, default=128)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    create_gt_volume(args.data_dir, args.grid_res, args.threshold)
