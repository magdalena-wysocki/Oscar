"""Validation and test-time calibration utilities for OSCAR.

At each training epoch boundary the network is frozen and, for every validation
subject, a fresh latent code is optimized ("calibrated") for up to
``val_max_cal_steps`` steps. Every ``val_cal_eval_every`` steps the predicted 3D
occupancy is compared against the GT volume with Dice and HD95. This both tracks
the best model and selects the number of test-time optimization steps
(``best_cal_steps``) that generalize best, which ``test_time_optimize.py`` reuses.
"""

import copy
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt


# ──────────────────────────────────────────────────────────────────────
# 3D volumetric metrics
# ──────────────────────────────────────────────────────────────────────

def compute_dice(a: np.ndarray, b: np.ndarray, label: Optional[int] = None) -> float:
    if label is not None:
        a = (a == label)
        b = (b == label)
    intersection = np.logical_and(a, b).sum()
    union = a.sum() + b.sum()
    if union == 0:
        return float("nan")
    return (2.0 * intersection) / union


def compute_dice_multiclass(a: np.ndarray, b: np.ndarray) -> List[float]:
    classes = np.unique(np.concatenate((a.reshape(-1), b.reshape(-1))))
    scores = [compute_dice(a, b, label=c) for c in classes]
    scores = [s for s in scores if np.isfinite(s)]
    return scores or [float("nan")]


def compute_hd95(a: np.ndarray, b: np.ndarray,
                 label: Optional[int] = None,
                 voxel_spacing: Optional[Tuple[float, ...]] = None) -> float:
    if label is not None:
        a_mask, b_mask = (a == label), (b == label)
    else:
        a_mask, b_mask = (a > 0), (b > 0)

    if np.array_equal(a_mask, b_mask):
        return 0.0
    if not a_mask.any() or not b_mask.any():
        # One side empty: return the maximum possible distance in the volume.
        if voxel_spacing is not None:
            return float(np.sqrt(sum((s * sp) ** 2 for s, sp in zip(a_mask.shape, voxel_spacing))))
        return float(np.sqrt(sum(s ** 2 for s in a_mask.shape)))

    a_dt = distance_transform_edt(~a_mask, sampling=voxel_spacing)
    b_dt = distance_transform_edt(~b_mask, sampling=voxel_spacing)
    all_distances = np.concatenate([a_dt[b_mask], b_dt[a_mask]])
    return float(np.percentile(all_distances, 95))


def compute_hd95_multiclass(a: np.ndarray, b: np.ndarray,
                            voxel_spacing: Optional[Tuple[float, ...]] = None) -> List[float]:
    classes = np.unique(np.concatenate((a.reshape(-1), b.reshape(-1))))
    scores = [compute_hd95(a, b, label=c, voxel_spacing=voxel_spacing) for c in classes]
    scores = [s for s in scores if np.isfinite(s)]
    return scores or [float("nan")]


# ──────────────────────────────────────────────────────────────────────
# Occupancy grid prediction
# ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_occupancy_volume(args, render_kwargs, latent_id: int,
                             grid_res: int = 128, chunk: int = 131072,
                             device: torch.device = torch.device("cuda")) -> np.ndarray:
    """Evaluate the occupancy head on a [0,1]^3 grid; return a binary volume."""
    from nerf_utils import batchify_rays  # local import to avoid a circular import

    lin = torch.linspace(0.0, 1.0, grid_res, device=device)
    X, Y, Z = torch.meshgrid(lin, lin, lin, indexing="ij")
    pts = torch.stack([X, Y, Z], dim=-1).reshape(-1, 3)

    rk = dict(render_kwargs)
    rk["latent_id"] = torch.tensor(latent_id, device=device)
    rk["return_only_occ"] = True

    outs = []
    for i in range(0, pts.shape[0], chunk):
        rk["pts"] = pts[i:i + chunk]
        logits = batchify_rays(None, **rk)["occ"].squeeze(-1)
        outs.append(torch.sigmoid(logits))

    occ = torch.cat(outs, dim=0).reshape(grid_res, grid_res, grid_res)
    return (occ > 0.5).cpu().numpy().astype(np.uint8)


# ──────────────────────────────────────────────────────────────────────
# Calibration (latent optimization on validation subjects)
# ──────────────────────────────────────────────────────────────────────

def clone_render_kwargs(render_kwargs_train, device):
    """Deep-copy the network + latent codes so calibration cannot affect training."""
    rk = dict(render_kwargs_train)
    rk["network_fn"] = copy.deepcopy(render_kwargs_train["network_fn"]).to(device)
    rk["latent_codes"] = copy.deepcopy(render_kwargs_train["latent_codes"]).to(device)
    return rk


def run_calibration_and_evaluate(args, render_kwargs_train, val_dataset, val_indices,
                                 max_cal_steps, eval_every, grid_res, losses, device):
    """Calibrate a fresh latent per validation subject and evaluate occupancy metrics.

    Returns ``{cal_step: {"dice": [...], "hd95": [...], "occ_dice": [...], "occ_hd95": [...]}}``.
    """
    import torch.nn as nn
    from nerf_utils import render_us, compute_loss

    results_by_step: Dict[int, Dict[str, list]] = {}

    # Group validation frames by subject (latent id).
    subject_groups: Dict[int, List[int]] = {}
    for idx in val_indices:
        subject_groups.setdefault(val_dataset[idx]["latent_id"], []).append(idx)

    for subject_lid, indices in subject_groups.items():
        print(f"[VAL] Calibrating subject latent_id={subject_lid}, "
              f"{len(indices)} frames, up to {max_cal_steps} steps")

        rk = clone_render_kwargs(render_kwargs_train, device)
        # Single fresh latent for this subject, initialized from the training average.
        rk["latent_codes"] = nn.Embedding(1, args.latent_dim).to(device)
        with torch.no_grad():
            rk["latent_codes"].weight.data[:] = render_kwargs_train["latent_codes"].weight.data.mean(
                dim=0, keepdim=True)
        for p in rk["network_fn"].parameters():
            p.requires_grad = False
        cal_optimizer = torch.optim.Adam(rk["latent_codes"].parameters(), lr=args.lrate, betas=(0.9, 0.999))

        gt_volume = None
        voxel_spacing = None
        try:
            gt_volume = val_dataset.get_gt_volume(subject_lid)
            voxel_spacing = val_dataset._voxel_spacing[subject_lid]
        except Exception as e:
            print(f"  [VAL] Warning: could not load GT volume/spacing for subject {subject_lid}: {e}")

        perm = np.random.permutation(indices)
        perm_idx = 0
        for cal_step in range(max_cal_steps):
            if perm_idx >= len(perm):
                perm = np.random.permutation(indices)
                perm_idx = 0
            sample = val_dataset[int(perm[perm_idx])]
            perm_idx += 1

            rk["latent_id"] = torch.tensor(0, device=device)
            target = torch.tensor(sample["image"], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
            rk["pts"] = torch.from_numpy(sample["poses_labels"]).to(device)

            rendering_output = render_us(chunk=args.chunk, **rk)
            loss = compute_loss(rendering_output["intensity_map"], target, args, losses, cal_step)
            latent = rk["latent_codes"](torch.tensor([0], device=device))
            loss["reg_latent"] = (args.reg_latent, torch.mean(latent ** 2))

            total_loss = sum(w * v for w, v in loss.values())
            cal_optimizer.zero_grad()
            total_loss.backward()
            cal_optimizer.step()

            if gt_volume is not None and ((cal_step + 1) % eval_every == 0 or cal_step == 0):
                step_key = cal_step + 1 if cal_step > 0 else 1
                pred_vol = predict_occupancy_volume(args, rk, latent_id=0, grid_res=grid_res, device=device)
                dice = compute_dice_multiclass(gt_volume, pred_vol)
                hd95 = compute_hd95_multiclass(gt_volume, pred_vol, voxel_spacing=voxel_spacing)
                mean_dice, mean_hd95 = float(np.nanmean(dice)), float(np.nanmean(hd95))
                occ_dice = float(dice[1] if len(dice) > 1 else mean_dice)
                occ_hd95 = float(hd95[1] if len(hd95) > 1 else mean_hd95)

                bucket = results_by_step.setdefault(
                    step_key, {"dice": [], "hd95": [], "occ_dice": [], "occ_hd95": []})
                bucket["dice"].append(mean_dice)
                bucket["hd95"].append(mean_hd95)
                bucket["occ_dice"].append(occ_dice)
                bucket["occ_hd95"].append(occ_hd95)
                print(f"  [VAL] cal_step={step_key}, subject={subject_lid}, "
                      f"occ_dice={occ_dice:.4f}, occ_hd95={occ_hd95:.4f}")

        del rk, cal_optimizer
        torch.cuda.empty_cache()

    return results_by_step


def aggregate_val_results(results_by_step: Dict) -> Dict:
    """Average per-step metrics across subjects."""
    agg = {}
    for step, data in results_by_step.items():
        def _mean(key):
            vals = [v for v in data.get(key, []) if np.isfinite(v)]
            return float(np.mean(vals)) if vals else float("nan")
        agg[step] = {
            "mean_dice": _mean("dice"),
            "mean_hd95": _mean("hd95"),
            "occ_dice": _mean("occ_dice"),
            "occ_hd95": _mean("occ_hd95"),
        }
    return agg


def find_best_cal_steps(agg: Dict, metric: str = "occ_hd95",
                        higher_is_better: bool = False) -> Optional[int]:
    """Return the calibration step count that yields the best metric."""
    best_step = None
    best_val = float("-inf") if higher_is_better else float("inf")
    for step, metrics in agg.items():
        val = metrics.get(metric, float("nan"))
        if not np.isfinite(val):
            continue
        if (higher_is_better and val > best_val) or (not higher_is_better and val < best_val):
            best_val, best_step = val, step
    return best_step


# ──────────────────────────────────────────────────────────────────────
# Validation history tracker
# ──────────────────────────────────────────────────────────────────────

class ValidationTracker:
    """Tracks the best validation model + selected calibration steps across epochs."""

    def __init__(self, save_dir: str, metric: str = "occ_hd95", higher_is_better: bool = False):
        self.save_dir = save_dir
        self.metric = metric
        self.higher_is_better = higher_is_better
        self.history: List[Dict] = []
        self.best_epoch = -1
        self.best_metric_val = float("-inf") if higher_is_better else float("inf")
        self.best_cal_steps = 0

    def load_state(self, path: str):
        with open(path, "r") as f:
            data = json.load(f)
        if data.get("metric_name", self.metric) != self.metric:
            print(f"[VAL] History uses a different metric; resetting best state.")
            return
        self.history = data.get("history", [])
        self.best_epoch = data.get("best_epoch", -1)
        self.best_metric_val = data.get("best_metric", self.best_metric_val)
        self.best_cal_steps = data.get("best_cal_steps", 0)
        print(f"[VAL] Loaded history from {path}: best_epoch={self.best_epoch}, "
              f"best_metric={self.best_metric_val:.4f}, best_cal_steps={self.best_cal_steps}")

    def update(self, epoch: int, train_step: int, agg_results: Dict, best_cal_step: int) -> bool:
        current_val = agg_results.get(best_cal_step, {}).get(self.metric, float("nan"))
        self.history.append({
            "epoch": epoch,
            "train_step": train_step,
            "best_cal_steps": best_cal_step,
            "metrics_by_cal_step": {str(k): v for k, v in agg_results.items()},
            "best_metric": current_val,
        })

        is_new_best = np.isfinite(current_val) and (
            (self.higher_is_better and current_val > self.best_metric_val)
            or (not self.higher_is_better and current_val < self.best_metric_val))
        if is_new_best:
            self.best_epoch = epoch
            self.best_metric_val = current_val
            self.best_cal_steps = best_cal_step
            print(f"[VAL] *** New best at epoch {epoch}: {self.metric}={current_val:.4f}, "
                  f"best_cal_steps={best_cal_step} ***")

        self.save()
        return is_new_best

    def save(self):
        os.makedirs(self.save_dir, exist_ok=True)
        with open(os.path.join(self.save_dir, "validation_history.json"), "w") as f:
            json.dump({
                "best_epoch": self.best_epoch,
                "best_metric": self.best_metric_val,
                "best_cal_steps": self.best_cal_steps,
                "metric_name": self.metric,
                "higher_is_better": self.higher_is_better,
                "history": self.history,
            }, f, indent=2)

    @staticmethod
    def load(save_dir: str) -> Dict:
        with open(os.path.join(save_dir, "validation_history.json"), "r") as f:
            return json.load(f)
