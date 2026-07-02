# OSCAR dataset

Ultrasound sweeps used to train and evaluate OSCAR: **152 simulated** VerSe-vertebra
sweeps (~9 GB, uint8) + **3 real phantom** sweeps (`phantom_01/02/03`, ~1.8 GB).
**Everything regenerates from `poses.npy`.** Use it with the OSCAR code repository
(the preprocessing and training scripts referenced below live there).

## Split (matches `configs/oscar.txt`)

| Split | Sweeps | Subjects |
|---|---:|---|
| **Train** | 132 | 33 base + 33 rotation-augmented + 66 translation-augmented copies of 12 subjects: `004 005 008 010 014 031 072 073 078 124 139 152` |
| **Validation** | 10 | `046 091 102 134` (3 vertebrae each, minus one incomplete) |
| **Test** | 10 | `030 048 065` (3 vertebrae each) + `076_21` |
| **Phantom test** | 3 | real tissue-mimicking phantoms `phantom_01 / 02 / 03` (CT-registered GT) |

Validation and test subjects are disjoint from each other and from training.

## What each folder contains

```
data/<sweep>/                                (simulated sweeps)
  poses.npy                [N,4,4]  float  probe poses (tracking + registration baked in, mm)
  images.npy               [N,H,W]  uint8  B-mode frames (0-255)
  masks_gt.npy             [N,H,W]  uint8  per-frame occupancy mask ({0,1})
  point_cloud_min_max.txt                  physical extent (the mm -> unit-cube mapping)
  transform.npy            [4,3]    float  translation-augmented sweeps ONLY: rigid transform (~48 B)
```

The large derived files that training/evaluation consume — `poses_labels.npy` and
`gt_volume.npy` — are **not** included; they are rebuilt locally (next section) and
reproduce the originals bit-for-bit, so they never have to be distributed.

**Phantom folders** are real ultrasound, collected with a different probe geometry and
axis convention, so they add two tiny files that tell `create_poses_labels` how to
rebuild them exactly:

```
data/phantom_0X/
  poses.npy                [N,4,4]     float    probe poses
  images.npy               [N,H,W]     uint8    real B-mode frames
  masks_gt.npy             [N,H,W]     uint8    occupancy mask
  transform.npy            [4,3]       float    axis swap applied after normalization (~48 B)
  probe.txt                                     probe geometry, "depth width" (110 51.3)
  point_cloud_min_max.txt                       physical extent
```

`poses_labels.npy` and `gt_volume.npy` are regenerated for phantoms too — bit-for-bit.

## Preprocess (run once, after downloading)

Place this `data/` folder at the root of the code repository, then:

```bash
python scripts/preprocess_all.py --data_root ./data
```

This regenerates `poses_labels.npy` and `gt_volume.npy` for **every** sweep (simulated
and phantom): `create_poses_labels` ray-marches `poses.npy` (using each folder's
`probe.txt` geometry if present) and applies `transform.npy` if present. The
regenerated `poses_labels.npy` are large (hundreds of MB per sweep), so the on-disk
footprint grows to tens of GB. To redo a single sweep:

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
- Rotation-augmented sweeps reproduce from `poses.npy` alone; translation-augmented and
  phantom sweeps additionally apply the bundled `transform.npy` (done automatically by
  `create_poses_labels.py`); phantoms also read `probe.txt` for their probe width (51.3).
- Sizes as shipped (uint8 images/masks + `poses.npy`, no `poses_labels`): simulated base
  2.1 GB, rotation-aug 2.1 GB, translation-aug 4.1 GB, val+test 1.3 GB (~9 GB);
  phantom ~1.8 GB (0.37 + 0.37 + 1.0). Total ~11 GB.
