"""Render the partial-to-complete teaser figure for OSCAR (static PNG + rotating GIF).

Shows the benefit of shape completion in one strip:

    [ what ultrasound sees ]  ->  [ OSCAR completed shape vs ground truth ]
      partial probe-facing            full solid shape (orange), overlaid on the
      surface (a shell)               ground-truth surface (grey) for reference

"What ultrasound sees" is the **probe-facing surface**: for every scanline we keep
only the *first* occupied sample (the near surface before the acoustic shadow),
so the left panel is genuinely partial -- roughly the half of the bone the beam
can reach. It is drawn from the ground-truth segmentation (``masks_gt.npy`` back-
projected through ``poses_labels.npy``), so it is a real surface, not a guess.

The completed shape is an OSCAR prediction (``mesh_completed.obj`` or an occupancy
``.npy``). The ground-truth surface is rebuilt from ``gt_volume.npy`` and drawn
translucent behind it, so agreement (and any recovered region) is visible at a
glance. Both panels share a side view; ``--gif`` also writes a rotating turntable.

Usage:
    python scripts/make_teaser.py --data_dir ./data/<subject> \
        --mesh logs_test/<run>/mesh_completed.obj \
        --out docs/assets/figures/teaser.png \
        --gif docs/assets/figures/teaser.gif [--view side]
"""

import argparse
import io
import os
import sys
import tempfile

import numpy as np
import trimesh

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mesh_utils import occupancy_to_mesh  # noqa: E402

# elev, azim per named view. The rotating GIF spins azim; elev stays fixed.
VIEWS = {"front": (12, -90), "side": (12, 0), "top": (90, -90)}
OBSERVED_RGBA = (0.72, 0.78, 0.86, 1.0)    # blue-grey: the observation the beam reaches
GT_RGBA = (0.55, 0.58, 0.63, 0.32)         # translucent grey: ground-truth reference shell
OSCAR_RGBA = (0.92, 0.50, 0.13, 1.0)       # accent orange: the OSCAR-completed solid
EDGE_RGBA = (0.10, 0.12, 0.15, 0.05)
GT_INFLATE = 1.015                         # push GT out ~1.5% so it is a stable outer shell
                                           # (avoids GT/OSCAR z-fighting as the view spins)


def load_probe_facing_points(data_dir):
    """The near surface the beam reaches: first occupied sample per scanline.

    ``poses_labels.npy`` is [N, W, H, 3] with depth along H (index 0 nearest the
    probe); ``masks_gt.npy`` is [N, H, W] (transposed). For each frame and each
    scanline w we take the first occupied depth -- the surface before the shadow.
    """
    poses_labels = np.load(os.path.join(data_dir, "poses_labels.npy"), mmap_mode="r")
    masks_gt = np.load(os.path.join(data_dir, "masks_gt.npy"), mmap_mode="r")

    pts = []
    for i in range(poses_labels.shape[0]):
        p = np.asarray(poses_labels[i], dtype=np.float32)   # [W, H, 3]
        m = np.asarray(masks_gt[i])                         # [H, W]
        if m.shape != p.shape[:2]:                          # masks are stored transposed
            m = m.T                                         # -> [W, H]
        occ = m > 0.5
        has_hit = occ.any(axis=1)                           # scanlines that see anything
        first_h = np.argmax(occ, axis=1)                    # first occupied depth per scanline
        w = np.where(has_hit)[0]
        if len(w):
            pts.append(p[w, first_h[w]])
    if not pts:
        raise ValueError(f"No occupied pixels found in {data_dir}")
    return np.concatenate(pts, axis=0)


def load_mesh(path, iso=0.5):
    """Load a mesh, first extracting one via marching cubes if given an occupancy grid."""
    if path.endswith(".npy"):
        occ = np.load(path)
        occ = (occ > 0).astype(np.float32) if occ.dtype != np.float32 else occ
        out = os.path.join(tempfile.mkdtemp(), "mesh.obj")
        occupancy_to_mesh(occ, iso=iso, out_path=out)
        path = out
    mesh = trimesh.load(path, force="mesh", process=False)
    if mesh.is_empty or len(mesh.faces) == 0:
        raise ValueError(f"Empty mesh: {path}")
    return np.asarray(mesh.vertices), np.asarray(mesh.faces)


def observed_face_mask(vertices, faces, pts, res=64, radius=1):
    """True for faces near the observed points (the probe-facing part of the mesh).

    Voxelises the observation at ``res^3``, dilates by ``radius`` voxels, and flags
    faces whose vertices mostly fall in covered voxels. Pure-numpy, no scipy.
    """
    grid = np.zeros((res, res, res), dtype=bool)
    idx = np.clip((pts * (res - 1)).astype(np.int64), 0, res - 1)
    grid[idx[:, 0], idx[:, 1], idx[:, 2]] = True

    dilated = grid.copy()
    for _ in range(radius):                                 # 6-neighbour dilation
        shifted = dilated.copy()
        shifted[:-1] |= dilated[1:]; shifted[1:] |= dilated[:-1]
        shifted[:, :-1] |= dilated[:, 1:]; shifted[:, 1:] |= dilated[:, :-1]
        shifted[:, :, :-1] |= dilated[:, :, 1:]; shifted[:, :, 1:] |= dilated[:, :, :-1]
        dilated = shifted

    vidx = np.clip((vertices * (res - 1)).astype(np.int64), 0, res - 1)
    v_obs = dilated[vidx[:, 0], vidx[:, 1], vidx[:, 2]]
    return v_obs[faces].mean(axis=1) > 0.5


def _subsample(faces, n, seed=0):
    if len(faces) > n:
        faces = faces[np.random.default_rng(seed).choice(len(faces), size=n, replace=False)]
    return faces


def _style(ax, center, extent, elev, azim):
    ax.set_xlim(center[0] - extent / 2, center[0] + extent / 2)
    ax.set_ylim(center[1] - extent / 2, center[1] + extent / 2)
    ax.set_zlim(center[2] - extent / 2, center[2] + extent / 2)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()


def build_figure(partial_tris, gt_tris, oscar_tris, center, extent, dice=None):
    """Two-panel figure; returns (fig, ax_left, ax_right) with view left unset."""
    fig = plt.figure(figsize=(11, 5.4))

    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.add_collection3d(Poly3DCollection(
        partial_tris, facecolor=OBSERVED_RGBA, edgecolor=EDGE_RGBA, linewidths=0.03))
    ax1.set_title("What ultrasound sees\n(partial, probe-facing surface)", fontsize=13)

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax2.add_collection3d(Poly3DCollection(gt_tris, facecolor=GT_RGBA, edgecolor="none"))
    ax2.add_collection3d(Poly3DCollection(
        oscar_tris, facecolor=OSCAR_RGBA, edgecolor=EDGE_RGBA, linewidths=0.03))
    title = "OSCAR completed shape\n(orange) vs ground truth (grey)"
    if dice is not None:
        title += f"  —  Dice {dice:.2f}"
    ax2.set_title(title, fontsize=13)

    for ax in (ax1, ax2):
        _style(ax, center, extent, *VIEWS["side"])
    fig.text(0.5, 0.52, r"$\rightarrow$", ha="center", va="center", fontsize=34)
    return fig, ax1, ax2


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir", required=True, help="subject dir with masks_gt/poses_labels/gt_volume")
    parser.add_argument("--mesh", required=True, help="OSCAR completed mesh (.obj) or occupancy grid (.npy)")
    parser.add_argument("--gt", default=None, help="GT occupancy grid (default: <data_dir>/gt_volume.npy)")
    parser.add_argument("--out", default="docs/assets/figures/teaser.png")
    parser.add_argument("--gif", default=None, help="also write a rotating turntable GIF to this path")
    parser.add_argument("--iso", type=float, default=0.5, help="iso level for .npy mesh inputs")
    parser.add_argument("--view", choices=list(VIEWS), default="side")
    parser.add_argument("--radius", type=int, default=1, help="dilation (voxels) for the observed shell")
    parser.add_argument("--dice", type=float, default=None, help="optional Dice to annotate")
    parser.add_argument("--max_faces", type=int, default=120000,
                        help="face budget per mesh; keep high for solid surfaces "
                             "(matplotlib draws individual triangles, so subsampling makes gaps)")
    parser.add_argument("--gif_step", type=int, default=10, help="azimuth degrees per GIF frame")
    parser.add_argument("--gif_dpi", type=int, default=100)
    args = parser.parse_args()

    gt_path = args.gt or os.path.join(args.data_dir, "gt_volume.npy")

    partial = load_probe_facing_points(args.data_dir)
    gt_v, gt_f = load_mesh(gt_path, args.iso)
    os_v, os_f = load_mesh(args.mesh, args.iso)

    # Left panel: the probe-facing part of the GT surface (a real partial shell).
    obs = observed_face_mask(gt_v, gt_f, partial, radius=args.radius)
    obs_frac = float(obs.mean())
    partial_f = _subsample(gt_f[obs], args.max_faces)
    gt_f = _subsample(gt_f, args.max_faces)
    os_f = _subsample(os_f, args.max_faces)

    # Inflate GT slightly so it stays a consistent outer shell behind OSCAR at every
    # angle (matplotlib paints collections in add-order, not by true depth).
    gt_center = gt_v.mean(axis=0)
    gt_v_inflated = gt_center + (gt_v - gt_center) * GT_INFLATE

    partial_tris = gt_v[partial_f]
    gt_tris = gt_v_inflated[gt_f]
    oscar_tris = os_v[os_f]

    allv = np.vstack([gt_v, os_v])
    center = allv.mean(axis=0)
    extent = float(np.max(allv.max(axis=0) - allv.min(axis=0))) or 1.0

    # ---- static PNG ----
    fig, ax1, ax2 = build_figure(partial_tris, gt_tris, oscar_tris, center, extent, args.dice)
    elev, azim0 = VIEWS[args.view]
    for ax in (ax1, ax2):
        ax.view_init(elev=elev, azim=azim0)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=180, bbox_inches="tight", pad_inches=0.05)
    print(f"Saved teaser -> {args.out}  "
          f"(probe-facing shell = {obs_frac:.0%} of the GT surface at radius {args.radius})")

    # ---- rotating GIF (reuses the same artists; just spins the camera) ----
    if args.gif:
        frames = []
        for azim in range(0, 360, args.gif_step):
            for ax in (ax1, ax2):
                ax.view_init(elev=elev, azim=azim)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=args.gif_dpi, bbox_inches="tight", pad_inches=0.05)
            buf.seek(0)
            frames.append(matplotlib.image.imread(buf))
        # assemble with Pillow (ships with matplotlib)
        from PIL import Image
        imgs = [Image.fromarray((f[:, :, :3] * 255).astype(np.uint8)) for f in frames]
        os.makedirs(os.path.dirname(os.path.abspath(args.gif)), exist_ok=True)
        imgs[0].save(args.gif, save_all=True, append_images=imgs[1:],
                     duration=90, loop=0, optimize=True)
        print(f"Saved rotating teaser -> {args.gif}  ({len(imgs)} frames)")

    plt.close(fig)


if __name__ == "__main__":
    main()
