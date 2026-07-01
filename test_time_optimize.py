"""Test-time optimization for OSCAR on an unseen ultrasound sweep.

Given a trained checkpoint, the network is frozen and a single latent code is
optimized to reconstruct the new subject's B-mode frames (image loss + physics
regularization). The number of optimization steps defaults to the
``best_cal_steps`` selected on the validation set during training. Occupancy
Dice/HD95 against the GT volume are logged per epoch, and the final predicted
occupancy is exported as a mesh.

Usage:
    python test_time_optimize.py --config configs/optimize_oscar.txt \
        --datadir ./data/<test_subject> \
        --latent_optimize_ckpt_path ./checkpoints/oscar_best_model.tar
"""

import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from monai.losses.ssim_loss import SSIMLoss
from monai.losses import LocalNormalizedCrossCorrelationLoss
from torch.nn import BCELoss

from unerf_config import config_parser
from data_utils import UltrasoundLazyDataset
from nerf_utils import create_one_network, img2mse, render_us, compute_loss, compute_regularization, batchify_rays
from validation_utils import (
    ValidationTracker, predict_occupancy_volume,
    compute_dice_multiclass, compute_hd95_multiclass,
)
from mesh_utils import occupancy_to_mesh, save_mesh_view_images

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_checkpoint(ckpt_path: str):
    """Resolve a checkpoint path to ``(ckpt_file, history_dir)``."""
    ckpt_path = os.path.abspath(ckpt_path)
    if os.path.isdir(ckpt_path):
        best = os.path.join(ckpt_path, "best_model.tar")
        if os.path.isfile(best):
            return best, ckpt_path
        tars = sorted(f for f in os.listdir(ckpt_path) if f.endswith(".tar"))
        if not tars:
            raise FileNotFoundError(f"No .tar checkpoint found in {ckpt_path}")
        return os.path.join(ckpt_path, tars[-1]), ckpt_path
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    return ckpt_path, os.path.dirname(ckpt_path)


@torch.no_grad()
def predict_occupancy_grid(render_kwargs_test, grid_res, device, out_path):
    """Predict a probability occupancy grid and export it as a mesh."""
    lin = torch.linspace(0.0, 1.0, grid_res, device=device)
    X, Y, Z = torch.meshgrid(lin, lin, lin, indexing="ij")
    pts = torch.stack([X, Y, Z], dim=-1).reshape(-1, 3)

    rk = dict(render_kwargs_test)
    rk["latent_id"] = torch.tensor(0, device=device)
    rk["return_only_occ"] = True
    outs = []
    for i in range(0, pts.shape[0], 131072):
        rk["pts"] = pts[i:i + 131072]
        outs.append(torch.sigmoid(batchify_rays(None, **rk)["occ"].squeeze(-1)))
    occ = torch.cat(outs, dim=0).reshape(grid_res, grid_res, grid_res).contiguous()
    occupancy_to_mesh(occ, out_path=out_path)
    return occ


def evaluate_occupancy_metrics(args, ds, render_kwargs_test, device, grid_res):
    gt_volume = ds.get_gt_volume(0)
    voxel_spacing = ds._voxel_spacing[0]
    pred_vol = predict_occupancy_volume(args, render_kwargs_test, latent_id=0,
                                        grid_res=grid_res, device=device)
    dice = compute_dice_multiclass(gt_volume, pred_vol)
    hd95 = compute_hd95_multiclass(gt_volume, pred_vol, voxel_spacing)
    mean_dice, mean_hd95 = float(np.nanmean(dice)), float(np.nanmean(hd95))
    return {
        "occ_dice": float(dice[1] if len(dice) > 1 else mean_dice),
        "occ_hd95": float(hd95[1] if len(hd95) > 1 else mean_hd95),
        "mean_dice": mean_dice, "std_dice": float(np.nanstd(dice)),
        "mean_hd95": mean_hd95, "std_hd95": float(np.nanstd(hd95)),
    }


def save_metric_plot(history, out_dir):
    rows = [m for m in history if "error" not in m]
    if not rows:
        return None
    epochs = [m["epoch"] for m in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].plot(epochs, [m["occ_dice"] for m in rows], marker="o")
    axes[0].set_xlabel("optimization epoch"); axes[0].set_ylabel("Dice"); axes[0].grid(alpha=0.3)
    axes[1].plot(epochs, [m["occ_hd95"] for m in rows], marker="o")
    axes[1].set_xlabel("optimization epoch"); axes[1].set_ylabel("HD95"); axes[1].grid(alpha=0.3)
    path = os.path.join(out_dir, "test_time_metrics_by_epoch.png")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return path


def test(args):
    ckpt_file, history_dir = resolve_checkpoint(args.latent_optimize_ckpt_path)
    args.latent_optimize_ckpt_path = ckpt_file
    print(f"[TEST] Checkpoint: {ckpt_file}")

    # Number of optimization steps: validation-selected best_cal_steps, with
    # a negative --n_iters as an explicit override.
    best_cal_steps = None
    try:
        best_cal_steps = ValidationTracker.load(history_dir).get("best_cal_steps")
    except Exception as e:
        print(f"[TEST] No validation_history.json ({e})")
    N_iters = best_cal_steps if best_cal_steps else args.n_iters
    if args.n_iters < 0:
        N_iters = -args.n_iters
    print(f"[TEST] Optimizing the latent for {N_iters} steps")

    if args.random_seed >= 0:
        np.random.seed(args.random_seed if args.random_seed > 0 else 42)
        torch.manual_seed(args.random_seed if args.random_seed > 0 else 42)

    basedir, expname = args.basedir, args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)

    ds = UltrasoundLazyDataset(args.datadir)
    args.dataset_names = list(ds.dataset_names_to_ids.keys())
    i_train = ds.i_train
    print(f"[TEST] Subject frames: {len(i_train)}")

    with open(os.path.join(basedir, expname, "args.txt"), "w") as f:
        for k in sorted(vars(args)):
            f.write(f"{k} = {getattr(args, k)}\n")
    if args.config is not None:
        with open(os.path.join(basedir, expname, "config.txt"), "w") as f:
            f.write(open(args.config).read())

    render_kwargs_train, render_kwargs_test, _, optimizer = create_one_network(
        args, device=device, mode="optimize")

    ssim_loss = SSIMLoss(spatial_dims=2, data_range=1.0, kernel_type="gaussian",
                         win_size=args.ssim_filter_size, k1=0.01, k2=0.1)
    losses = {
        "l2": img2mse, "l1": torch.nn.L1Loss(), "ssim": ssim_loss,
        "lncc": LocalNormalizedCrossCorrelationLoss(spatial_dims=2), "bce": BCELoss(),
    }

    grad_accum = args.gradient_accumulation_steps
    metric_history = []
    metrics_path = os.path.join(basedir, expname, "test_time_metrics_by_epoch.json")
    epoch_num_images = max(1, len(i_train))

    def eval_epoch(epoch_idx, opt_steps, phase):
        try:
            metrics = evaluate_occupancy_metrics(args, ds, render_kwargs_test, device,
                                                 grid_res=args.test_metric_grid_res)
            entry = {"epoch": int(epoch_idx), "opt_steps": int(opt_steps), "phase": phase, **metrics}
            print(f"[TEST] epoch={epoch_idx} steps={opt_steps} "
                  f"Dice={entry['occ_dice']:.4f} HD95={entry['occ_hd95']:.4f}")
        except Exception as e:
            print(f"[TEST] Could not evaluate epoch {epoch_idx}: {e}")
            entry = {"epoch": int(epoch_idx), "opt_steps": int(opt_steps), "phase": phase, "error": str(e)}
        metric_history.append(entry)
        with open(metrics_path, "w") as f:
            json.dump(metric_history, f, indent=2)
        save_metric_plot(metric_history, os.path.join(basedir, expname))

    # ── Phase 1: latent optimization ──
    print(f"\n{'=' * 60}\n[TEST] Phase 1: latent optimization\n{'=' * 60}")
    eval_epoch(epoch_idx=0, opt_steps=0, phase="initial")

    perm = np.random.permutation(i_train)
    perm_ind = 0
    for step in range(N_iters):
        if perm_ind >= len(perm):
            perm = np.random.permutation(i_train)
            perm_ind = 0
        sample = ds[int(perm[perm_ind])]
        perm_ind += 1

        render_kwargs_train["latent_id"] = torch.tensor(sample["latent_id"], device=device)
        target = torch.tensor(sample["image"], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        render_kwargs_train["pts"] = torch.from_numpy(sample["poses_labels"]).to(device)

        rendering_output = render_us(chunk=args.chunk, **render_kwargs_train)

        if step % grad_accum == 0:
            optimizer.zero_grad()

        loss = compute_loss(rendering_output["intensity_map"], target, args, losses, step)
        if args.reg and not args.nisf and step > args.r_warm_up_it:
            loss.update(compute_regularization(
                rendering_output, losses,
                weights=(args.r_lcc_penalty, args.r_tv_penalty, args.r_max_reflection)))

        total_loss = sum(w * v for w, v in loss.values())
        (total_loss / grad_accum).backward()
        if (step + 1) % grad_accum == 0 or (step + 1) == N_iters:
            optimizer.step()

        completed = step + 1
        if completed % epoch_num_images == 0 or completed == N_iters:
            eval_epoch(epoch_idx=int(np.ceil(completed / epoch_num_images)),
                       opt_steps=completed,
                       phase="final" if completed == N_iters else "epoch")

    # ── Phase 2: export final mesh ──
    print(f"\n{'=' * 60}\n[TEST] Phase 2: exporting final mesh\n{'=' * 60}")
    mesh_path = os.path.join(basedir, expname, "mesh_completed.obj")
    predict_occupancy_grid(dict(render_kwargs_test), grid_res=128, device=device, out_path=mesh_path)
    mesh_views = save_mesh_view_images(mesh_path)

    # ── Phase 3: final metrics + results ──
    print(f"\n{'=' * 60}\n[TEST] Phase 3: final metrics\n{'=' * 60}")
    try:
        final_metrics = evaluate_occupancy_metrics(args, ds, render_kwargs_test, device,
                                                   grid_res=args.test_metric_grid_res)
        print(f"[TEST] Dice = {final_metrics['mean_dice']:.4f} (+/-{final_metrics['std_dice']:.4f}), "
              f"HD95 = {final_metrics['mean_hd95']:.4f} (+/-{final_metrics['std_hd95']:.4f})")
    except Exception as e:
        final_metrics = {"error": str(e)}
        print(f"[TEST] Could not evaluate final metrics: {e}")

    results = {
        "checkpoint_path": ckpt_file,
        "best_cal_steps": best_cal_steps,
        "n_iters_used": N_iters,
        "dataset": args.datadir,
        "num_frames": len(ds),
        "metrics": final_metrics,
        "test_time_metrics_by_epoch": metric_history,
        "mesh_completed_path": mesh_path,
        "mesh_completed_view_paths": mesh_views,
        "final_latent_code": render_kwargs_train["latent_codes"].weight.data.cpu().numpy().tolist(),
    }
    with open(os.path.join(basedir, expname, "test_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"[TEST] Saved test_results.json\n[TEST] Done.")


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    if torch.cuda.is_available():
        torch.set_default_device("cuda")
        print("GPU:", torch.cuda.get_device_name(0))
    parser = config_parser()
    args = parser.parse_args()
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    args.expname = f"{timestamp}_{args.expname}"
    test(args)
