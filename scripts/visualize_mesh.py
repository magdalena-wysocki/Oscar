"""Render front/side/top previews of an OSCAR occupancy prediction.

Accepts either a mesh (``.obj``) or an occupancy grid (``.npy``, either a
probability grid or a binary grid). For ``.npy`` inputs a mesh is first extracted
with marching cubes, then PNG previews are written next to the input.

Usage:
    python scripts/visualize_mesh.py path/to/mesh_completed.obj
    python scripts/visualize_mesh.py path/to/mesh_completed_prob.npy --iso 0.5
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mesh_utils import occupancy_to_mesh, save_mesh_view_images  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", help="path to a .obj mesh or a .npy occupancy grid")
    parser.add_argument("--iso", type=float, default=0.5, help="iso level for .npy inputs")
    args = parser.parse_args()

    mesh_path = args.path
    if args.path.endswith(".npy"):
        occ = np.load(args.path)
        mesh_path = args.path.replace(".npy", ".obj")
        occupancy_to_mesh(occ, iso=args.iso, out_path=mesh_path)
        print(f"Extracted mesh -> {mesh_path}")

    paths = save_mesh_view_images(mesh_path)
    if not paths:
        print("No views produced (empty mesh?).")


if __name__ == "__main__":
    main()
