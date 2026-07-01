"""Mesh export and rendering helpers for OSCAR occupancy predictions."""

import os

import numpy as np
import trimesh

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def occupancy_to_mesh(occ, iso: float = 0.5, out_path: str = "mesh.obj") -> str:
    """Marching-cubes mesh from a [G, G, G] occupancy probability grid.

    Also saves the probability grid (``*_prob.npy``) and the thresholded binary
    grid (``*_occ.npy``). Falls back to a voxel-surface mesh if marching cubes
    cannot produce a surface at ``iso``.
    """
    import mcubes

    occ_np = occ.detach().float().cpu().numpy() if hasattr(occ, "detach") else np.asarray(occ)
    occ_np = np.nan_to_num(occ_np, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)
    occ_np = np.clip(occ_np, 0.0, 1.0)

    np.save(out_path.replace(".obj", "_prob.npy"), occ_np)
    occ_bin = (occ_np >= iso).astype(np.uint8)
    np.save(out_path.replace(".obj", "_occ.npy"), occ_bin)

    if not (occ_np.min() < iso < occ_np.max()):
        print(f"[MESH] iso={iso} outside occupancy range "
              f"[{occ_np.min():.3f}, {occ_np.max():.3f}]; using voxel fallback")
        return occupancy_mask_to_surface_mesh(occ_bin.astype(bool), out_path)

    try:
        verts, faces = mcubes.marching_cubes(occ_np, float(iso))
        verts = verts / (occ_np.shape[0] - 1.0)
        trimesh.Trimesh(vertices=verts, faces=faces, process=False).export(out_path)
    except Exception as e:
        print(f"[MESH] marching_cubes failed ({e}); using voxel fallback")
        return occupancy_mask_to_surface_mesh(occ_bin.astype(bool), out_path)
    return out_path


def occupancy_mask_to_surface_mesh(mask: np.ndarray, out_path: str, max_voxels: int = 50000):
    """Fallback: build a cube-surface mesh from the boundary voxels of a binary mask."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 3 or not mask.any():
        print(f"[MESH] No occupied voxels for fallback mesh: {out_path}")
        return None

    padded = np.pad(mask, 1, constant_values=False)
    core = padded[1:-1, 1:-1, 1:-1]
    surface = core & (
        ~padded[:-2, 1:-1, 1:-1] | ~padded[2:, 1:-1, 1:-1]
        | ~padded[1:-1, :-2, 1:-1] | ~padded[1:-1, 2:, 1:-1]
        | ~padded[1:-1, 1:-1, :-2] | ~padded[1:-1, 1:-1, 2:]
    )
    coords = np.argwhere(surface)
    if len(coords) > max_voxels:
        rng = np.random.default_rng(0)
        coords = coords[rng.choice(len(coords), size=max_voxels, replace=False)]

    R = float(max(mask.shape) - 1) or 1.0
    half = 0.5 / R
    offsets = np.array([[-half, -half, -half], [half, -half, -half], [half, half, -half],
                        [-half, half, -half], [-half, -half, half], [half, -half, half],
                        [half, half, half], [-half, half, half]], dtype=np.float32)
    cube_faces = [[0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6], [0, 4, 5], [0, 5, 1],
                  [1, 5, 6], [1, 6, 2], [2, 6, 7], [2, 7, 3], [3, 7, 4], [3, 4, 0]]
    vertices, faces = [], []
    for coord in coords:
        base = len(vertices)
        center = coord.astype(np.float32) / R
        vertices.extend((center[None, :] + offsets).tolist())
        faces.extend([[base + a, base + b, base + c] for a, b, c in cube_faces])

    trimesh.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces), process=False).export(out_path)
    print(f"[MESH] Saved fallback voxel mesh ({len(coords)} surface voxels) to {out_path}")
    return out_path


def save_mesh_view_images(mesh_path: str, max_faces: int = 20000):
    """Render front/side/top PNG previews of a mesh next to it."""
    if not os.path.isfile(mesh_path):
        return []
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    if mesh.is_empty or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        print(f"[MESH] view skipped for empty mesh: {mesh_path}")
        return []

    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    if len(faces) > max_faces:
        rng = np.random.default_rng(0)
        faces = faces[rng.choice(len(faces), size=max_faces, replace=False)]

    center = vertices.mean(axis=0)
    extent = float(np.max(vertices.max(axis=0) - vertices.min(axis=0))) or 1.0
    out_dir = os.path.dirname(mesh_path)
    stem = os.path.splitext(os.path.basename(mesh_path))[0]

    image_paths = []
    for name, elev, azim in [("front", 12, -90), ("side", 12, 0), ("top", 90, -90)]:
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, projection="3d")
        ax.add_collection3d(Poly3DCollection(
            vertices[faces], facecolor=(0.72, 0.78, 0.86, 1.0),
            edgecolor=(0.18, 0.22, 0.28, 0.10), linewidths=0.05))
        ax.set_xlim(center[0] - extent / 2, center[0] + extent / 2)
        ax.set_ylim(center[1] - extent / 2, center[1] + extent / 2)
        ax.set_zlim(center[2] - extent / 2, center[2] + extent / 2)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()
        out_path = os.path.join(out_dir, f"{stem}_view_{name}.png")
        fig.savefig(out_path, dpi=180, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        image_paths.append(out_path)

    print(f"[MESH] Saved views: {', '.join(os.path.basename(p) for p in image_paths)}")
    return image_paths
