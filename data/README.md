# OSCAR dataset

Simulated ultrasound sweeps of VerSe vertebrae used to train and evaluate OSCAR.
**152 sweeps, ~9 GB (uint8).** Use it with the OSCAR code repository (the preprocessing
and training scripts referenced below live there).

## Split (matches `configs/oscar.txt`)

| Split | Sweeps | Subjects |
|---|---:|---|
| **Train** | 132 | 33 base + 33 rotation-augmented + 66 translation-augmented copies of 12 subjects: `004 005 008 010 014 031 072 073 078 124 139 152` |
| **Validation** | 10 | `046 091 102 134` (3 vertebrae each, minus one incomplete) |
| **Test** | 10 | `030 048 065` (3 vertebrae each) + `076_21` |

Validation and test subjects are disjoint from each other and from training.

## What each folder contains

```
data/<sweep>/
  poses.npy                [N,4,4]  float  probe poses (tracking + registration baked in, mm)
  images.npy               [N,H,W]  uint8  B-mode frames (0-255)
  masks_gt.npy             [N,H,W]  uint8  per-frame occupancy mask ({0,1})
  point_cloud_min_max.txt                  physical extent (the mm -> unit-cube mapping)
  transform.npy            [4,3]    float  translation-augmented sweeps ONLY: rigid transform (~48 B)
```

Only these small inputs are shipped. The large derived files that training and
evaluation actually consume — `poses_labels.npy` and `gt_volume.npy` — are **not**
included; they are rebuilt locally (next section) and reproduce the originals
bit-for-bit, so they never have to be distributed.

## Preprocess (run once, after downloading)

Place this `data/` folder at the root of the code repository, then:

```bash
python scripts/preprocess_all.py --data_root ./data
```

This regenerates `poses_labels.npy` and `gt_volume.npy` for all 152 sweeps (it
simply calls `scripts/create_poses_labels.py` and `scripts/create_gt_volume.py`
per folder). The regenerated `poses_labels.npy` are large (hundreds of MB per
sweep), so the on-disk footprint grows to tens of GB. To redo a single sweep:

```bash
python scripts/create_poses_labels.py --data_dir ./data/<sweep>
python scripts/create_gt_volume.py   --data_dir ./data/<sweep>
```

Then train directly with the ready-made config:

```bash
python train.py --config configs/oscar.txt
```

## Notes

- `poses.npy` is in **millimetres**; `point_cloud_min_max.txt` holds the min/max
  corner used to normalize the ray-marched points into the unit cube `[0,1]^3`.
- Rotation-augmented sweeps reproduce from `poses.npy` alone; translation-augmented
  sweeps additionally apply the bundled `transform.npy` (done automatically by
  `create_poses_labels.py`).
- Sizes as shipped (uint8): base 2.1 GB, rotation-aug 2.1 GB, translation-aug 4.1 GB,
  val+test 1.3 GB (~9 GB total).
