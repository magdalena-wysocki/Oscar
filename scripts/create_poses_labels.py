"""Precompute per-pixel 3D coordinates (``poses_labels.npy``) from ``poses.npy``.

For every B-mode frame, each pixel is ray-marched from its (already registered)
probe pose to a 3D location; all locations are then min-max normalized into the
unit cube [0, 1]^3. OSCAR queries the network directly at these coordinates.

This reproduces ``poses_labels.npy`` **exactly** (bit-for-bit) from the small
``poses.npy``, so the large ``poses_labels.npy`` does not need to be
distributed -- only ``poses.npy``, ``images.npy`` and ``masks_gt.npy`` do. The
probe pose already contains the tracking + registration transform, so no
additional transformation is required.

Inputs  (in --data_dir): poses.npy [N, 4, 4] (mm), images.npy [N, H, W]
Outputs (in --data_dir): poses_labels.npy [N, W, H, 3], point_cloud_min_max.txt

Usage:
    python scripts/create_poses_labels.py --data_dir ./data/<subject>
"""

import argparse
import os

import numpy as np
import torch


def get_rays_us_linear(H, W, sw, sh, c2w):
    t = c2w[:3, -1]
    R = c2w[:3, :3]
    x = torch.arange(-W / 2, W / 2, dtype=torch.float32) * sw
    origin_base = torch.stack([x, torch.zeros_like(x), torch.zeros_like(x)], dim=1)
    rays_o = (R @ origin_base.transpose(0, 1)).transpose(0, 1) + t
    rays_d = (R @ torch.tensor([0.0, 1.0, 0.0])).expand_as(rays_o)
    return rays_o, rays_d


def compute_pts_from_pose(H, W, sw, sh, pose, near, far):
    o, d = get_rays_us_linear(H, W, sw, sh, pose)
    o, d = o.reshape(-1, 3), d.reshape(-1, 3)
    t_vals = torch.linspace(0.0, 1.0, H)
    z_vals = (near * (1.0 - t_vals) + far * t_vals).expand(W, H)
    return d.unsqueeze(-2) * z_vals.unsqueeze(-1) + o.unsqueeze(-2)


def create_poses_labels(data_dir, probe_depth=110.0, probe_width=60.0):
    # Optional per-folder probe geometry: a probe.txt holding "depth width"
    # overrides the depth/width arguments for that folder.
    probe_file = os.path.join(data_dir, "probe.txt")
    if os.path.isfile(probe_file):
        vals = open(probe_file).read().split()
        probe_depth, probe_width = float(vals[0]), float(vals[1])
    poses = np.load(os.path.join(data_dir, "poses.npy")).astype(np.float32)
    images = np.load(os.path.join(data_dir, "images.npy"), mmap_mode="r")
    H, W = images[0].shape[:2]
    sh, sw = probe_depth / float(H), probe_width / float(W)

    pts = np.empty((poses.shape[0], W, H, 3), dtype=np.float32)
    for i, c2w in enumerate(poses):
        pts[i] = compute_pts_from_pose(
            H, W, sw, sh, torch.from_numpy(c2w).float(), near=0.0, far=probe_depth
        ).cpu().numpy()

    flat = pts.reshape(-1, 3)
    min_vals, max_vals = flat.min(axis=0), flat.max(axis=0)
    normalized = ((flat - min_vals) / (max_vals - min_vals)).reshape(pts.shape).astype(np.float32)

    # Augmented (rotated/translated) sweeps ship a tiny rigid transform instead of the
    # full poses_labels.npy. transform.npy is a (4, 3) array [Mᵀ (3 rows); b (1 row)] such
    # that the augmented points are: normalized_points @ Mᵀ + b  (R = rotation, b = translation).
    tf_path = os.path.join(data_dir, "transform.npy")
    if os.path.exists(tf_path):
        sol = np.load(tf_path).astype(np.float32)            # (4, 3): [Mᵀ (3 rows); b (1 row)]
        norm_flat = (flat - min_vals) / (max_vals - min_vals)
        normalized = (norm_flat @ sol[:3] + sol[3]).reshape(pts.shape).astype(np.float32)
        np.save(os.path.join(data_dir, "poses_labels.npy"), normalized)
        print(f"Saved poses_labels.npy {normalized.shape} (applied transform.npy) in {data_dir}; "
              f"kept existing point_cloud_min_max.txt")
        return

    np.save(os.path.join(data_dir, "poses_labels.npy"), normalized)
    with open(os.path.join(data_dir, "point_cloud_min_max.txt"), "w") as f:
        f.write(f"Point cloud min: {min_vals.tolist()}\n")
        f.write(f"Point cloud max: {max_vals.tolist()}\n")
    print(f"Saved poses_labels.npy {normalized.shape} and point_cloud_min_max.txt in {data_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--probe_depth", type=float, default=110.0,
                        help="probe imaging depth in mm used during simulation (default 110)")
    parser.add_argument("--probe_width", type=float, default=60.0,
                        help="probe width in mm used during simulation (default 60)")
    args = parser.parse_args()
    create_poses_labels(args.data_dir, args.probe_depth, args.probe_width)
