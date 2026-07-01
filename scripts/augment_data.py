"""Augment a subject for training by rotating or translating its 3D points.

OSCAR augmentation acts on the normalized per-pixel coordinates
(``poses_labels.npy``): the same 2D B-mode frames are reinterpreted in a
transformed 3D frame, which encourages the latent code to be pose-robust.

  * rotate    : spin the points about the cube center [0.5, 0.5, 0.5]
  * translate : shift the points by a random vector (and shift the physical
                extent in point_cloud_min_max.txt accordingly)

Images and masks are copied unchanged. After augmenting, run
``create_gt_volume.py`` on the output directory to rebuild ``gt_volume.npy``.
Augmented folders are local training artifacts -- they are not part of the
published dataset.

Inputs  (in --src): poses_labels.npy, images.npy, masks_gt.npy [, point_cloud_min_max.txt]
Outputs (in --out): augmented poses_labels.npy + copied images/masks [+ point_cloud_min_max.txt]

Usage:
    python scripts/augment_data.py --mode rotate    --src ./data/<subj> --out ./data_aug/<subj>_rot   --max_angle 10
    python scripts/augment_data.py --mode translate --src ./data/<subj> --out ./data_aug/<subj>_trans --ratio 0.075
"""

import argparse
import os
import shutil

import numpy as np


def rotation_matrix_from_axis_angle(axis, angle_degrees):
    angle = np.radians(angle_degrees)
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def _read_min_max(path):
    lines = open(path).readlines()
    mn = np.array(eval(lines[0].split(": ")[1]))
    mx = np.array(eval(lines[1].split(": ")[1]))
    return mn, mx


def augment(mode, src, out, max_angle=10.0, ratio=0.075, seed=None):
    rng = np.random.default_rng(seed)
    os.makedirs(out, exist_ok=True)
    for name in ("images.npy", "masks_gt.npy"):
        if os.path.exists(os.path.join(src, name)):
            shutil.copy2(os.path.join(src, name), os.path.join(out, name))

    poses_labels = np.load(os.path.join(src, "poses_labels.npy"))
    min_max_path = os.path.join(src, "point_cloud_min_max.txt")

    if mode == "rotate":
        axis = rng.standard_normal(3)
        R = rotation_matrix_from_axis_angle(axis, rng.uniform(0.0, max_angle))
        center = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        flat = poses_labels.reshape(-1, 3)
        out_pl = (((flat - center) @ R.T) + center).reshape(poses_labels.shape).astype(np.float32)
        # Rotation about the cube center keeps the physical extent unchanged.
        if os.path.exists(min_max_path):
            shutil.copy2(min_max_path, os.path.join(out, "point_cloud_min_max.txt"))
    else:  # translate
        direction = rng.standard_normal(3)
        direction = direction / np.linalg.norm(direction)
        t = (direction * ratio).astype(np.float32)
        out_pl = (poses_labels + t).astype(np.float32)
        if os.path.exists(min_max_path):
            mn, mx = _read_min_max(min_max_path)
            t_mm = t * (mx - mn)  # normalized shift -> physical shift
            with open(os.path.join(out, "point_cloud_min_max.txt"), "w") as f:
                f.write(f"Point cloud min: {(mn + t_mm).tolist()}\n")
                f.write(f"Point cloud max: {(mx + t_mm).tolist()}\n")

    np.save(os.path.join(out, "poses_labels.npy"), out_pl)
    print(f"[{mode}] wrote {out}: poses_labels {out_pl.shape}. "
          f"Run create_gt_volume.py on it to rebuild gt_volume.npy.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["rotate", "translate"], required=True)
    parser.add_argument("--src", type=str, required=True, help="source subject directory")
    parser.add_argument("--out", type=str, required=True, help="output augmented directory")
    parser.add_argument("--max_angle", type=float, default=10.0, help="max rotation angle (deg)")
    parser.add_argument("--ratio", type=float, default=0.075, help="translation magnitude (normalized)")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed (optional)")
    args = parser.parse_args()
    augment(args.mode, args.src, args.out, args.max_angle, args.ratio, args.seed)
