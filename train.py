"""Train OSCAR on a set of ultrasound sweeps.

A single network and one latent code per training subject are optimized jointly
to (a) reconstruct each B-mode frame through the differentiable acoustic renderer
and (b) predict the per-frame occupancy mask. At each epoch boundary the model is
validated by calibrating a fresh latent on held-out subjects and measuring 3D
Dice/HD95; the best model and the best calibration-step count are saved for
test-time optimization.

Usage:
    python train.py --config configs/train_oscar.txt
"""

import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from monai.losses.ssim_loss import SSIMLoss
from monai.losses import LocalNormalizedCrossCorrelationLoss
from torch.nn import BCELoss
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from unerf_config import config_parser
from data_utils import UltrasoundLazyDataset
from nerf_utils import create_one_network, img2mse, render_us, compute_loss, compute_regularization, soft_dice_loss
from validation_utils import (
    run_calibration_and_evaluate,
    aggregate_val_results,
    find_best_cal_steps,
    ValidationTracker,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train():
    parser = config_parser()
    args = parser.parse_args()

    if args.random_seed >= 0:
        np.random.seed(args.random_seed if args.random_seed > 0 else 42)
        torch.manual_seed(args.random_seed if args.random_seed > 0 else 42)

    basedir, expname = args.basedir, args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)

    # ── Datasets ──
    ds = UltrasoundLazyDataset(args.datadir)
    args.dataset_names = list(ds.dataset_names_to_ids.keys())
    i_train = ds.i_train
    print(f"[TRAIN] {len(args.dataset_names)} subjects, {len(i_train)} frames")

    val_dataset = None
    if args.val_datadir is not None:
        val_dataset = UltrasoundLazyDataset(args.val_datadir)
        print(f"[VAL] {len(val_dataset.dataset_names_to_ids)} validation subjects, "
              f"{len(val_dataset.i_train)} frames")

    # ── Logging + config dump ──
    writer = SummaryWriter(log_dir=os.path.join(basedir, "summaries", expname)) if args.tensorboard else None
    with open(os.path.join(basedir, expname, "args.txt"), "w") as f:
        for k in sorted(vars(args)):
            f.write(f"{k} = {getattr(args, k)}\n")
    if args.config is not None:
        with open(os.path.join(basedir, expname, "config.txt"), "w") as f:
            f.write(open(args.config).read())

    # ── Model / optimizer (resumes automatically if a checkpoint exists) ──
    render_kwargs_train, _, start, optimizer = create_one_network(args, device=device, mode="train")

    def checkpoint_state(global_step, extra=None):
        state = {
            "global_step": global_step,
            "network_fn_state_dict": render_kwargs_train["network_fn"].state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "latent_codes": render_kwargs_train["latent_codes"].state_dict(),
        }
        if extra:
            state.update(extra)
        return state

    # ── Losses ──
    ssim_loss = SSIMLoss(spatial_dims=2, data_range=1.0, kernel_type="gaussian",
                         win_size=args.ssim_filter_size, k1=0.01, k2=0.1)
    losses = {
        "l2": img2mse,
        "l1": torch.nn.L1Loss(),
        "ssim": ssim_loss,
        "lncc": LocalNormalizedCrossCorrelationLoss(spatial_dims=2),
        "bce": BCELoss(),
    }

    # ── Validation tracker ──
    val_tracker = ValidationTracker(save_dir=os.path.join(basedir, expname),
                                    metric="occ_hd95", higher_is_better=False)
    val_history_path = os.path.join(basedir, expname, "validation_history.json")
    if os.path.exists(val_history_path):
        val_tracker.load_state(val_history_path)
    epoch_counter = start // max(1, len(i_train))

    grad_accum = args.gradient_accumulation_steps
    accumulated_loss = {}
    perm = np.random.permutation(i_train)
    perm_ind = 0
    start = start + 1

    for i in trange(start, args.n_iters, mininterval=60):
        if perm_ind >= len(perm):
            perm = np.random.permutation(i_train)
            perm_ind = 0

            # ── Epoch boundary: validate ──
            epoch_counter += 1
            if val_dataset is not None and args.completion:
                print(f"\n[VAL] epoch {epoch_counter} (train step {i})")
                render_kwargs_train["network_fn"].eval()
                results = run_calibration_and_evaluate(
                    args, render_kwargs_train, val_dataset, val_dataset.i_train,
                    max_cal_steps=args.val_max_cal_steps, eval_every=args.val_cal_eval_every,
                    grid_res=args.val_grid_res, losses=losses, device=device,
                )
                agg = aggregate_val_results(results)
                best_cal = find_best_cal_steps(agg, metric=val_tracker.metric, higher_is_better=False)
                if best_cal is not None:
                    if val_tracker.update(epoch_counter, i, agg, best_cal):
                        torch.save(checkpoint_state(i, {"epoch": epoch_counter, "best_cal_steps": best_cal}),
                                   os.path.join(basedir, expname, "best_model.tar"))
                        print(f"[VAL] Saved best_model.tar")
                    if writer is not None:
                        writer.add_scalar("Val/best_cal_steps", val_tracker.best_cal_steps, i)
                        writer.add_scalar(f"Val/{val_tracker.metric}_best", val_tracker.best_metric_val, i)
                render_kwargs_train["network_fn"].train()

                patience = args.early_stop_patience
                if patience >= 0 and val_tracker.best_epoch >= 0 and \
                        (epoch_counter - val_tracker.best_epoch) >= patience:
                    print(f"[EARLY STOP] No improvement for {patience} validation epochs.")
                    break

        # ── One training step ──
        img_i = int(perm[perm_ind])
        perm_ind += 1
        sample = ds[img_i]

        render_kwargs_train["latent_id"] = torch.tensor(sample["latent_id"], device=device)
        target = torch.tensor(sample["image"], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        render_kwargs_train["pts"] = torch.from_numpy(sample["poses_labels"]).to(device)

        rendering_output = render_us(chunk=args.chunk, **render_kwargs_train)
        rendering_output["target"] = target

        if i % grad_accum == 0:
            optimizer.zero_grad()

        loss = compute_loss(rendering_output["intensity_map"], target, args, losses, i)

        # Physics regularization needs the acoustic maps, which NISF does not produce.
        if args.reg and not args.nisf and i > args.r_warm_up_it:
            loss.update(compute_regularization(
                rendering_output, losses,
                weights=(args.r_lcc_penalty, args.r_tv_penalty, args.r_max_reflection)))

        if args.completion:
            mask_gt = torch.tensor(sample["mask_gt"], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
            pred_occ = rendering_output["occ"].permute(2, 1, 0).unsqueeze(0)
            occ_loss = torch.nn.functional.binary_cross_entropy_with_logits(pred_occ, mask_gt)
            loss["occ"] = (args.occ_lambda, occ_loss)
            occ_pred = torch.sigmoid(pred_occ)
            rendering_output["occ_gt"] = mask_gt
            rendering_output["occ_pred"] = occ_pred
            if args.dice_loss:
                loss["dice"] = (args.dice_lambda, soft_dice_loss(occ_pred, mask_gt))

        # Occupancy (occ + optional dice) supervises the shared backbone only, not the latents.
        if args.completion and args.occ_loss_model_only:
            shape_keys = ("occ", "dice")
            total_other = sum(w * t for k, (w, t) in loss.items() if k not in shape_keys)
            (total_other / grad_accum).backward(retain_graph=True)
            shape_loss = sum(w * t for k, (w, t) in loss.items() if k in shape_keys)
            torch.autograd.backward(shape_loss / grad_accum,
                                    inputs=list(render_kwargs_train["network_fn"].parameters()))
        else:
            total_loss = sum(w * v for w, v in loss.values())
            (total_loss / grad_accum).backward()

        for k, (w, v) in loss.items():
            accumulated_loss[k] = accumulated_loss.get(k, 0.0) + (w * v).item()

        if (i + 1) % grad_accum == 0:
            optimizer.step()
            if writer is not None:
                writer.add_scalar("Loss/total_loss", sum(accumulated_loss.values()) / grad_accum, i)
                for k, v in accumulated_loss.items():
                    writer.add_scalar(f"Loss/{k}", v / grad_accum, i)
            accumulated_loss = {}

        # ── Periodic debug render ──
        if (i + 1) % args.i_print == 0 or i in (0, 1):
            _save_debug_render(rendering_output, args, sample, i,
                               os.path.join(basedir, expname, "train_rendering"))

        # ── Periodic checkpoint ──
        if (i + 1) % args.i_weights == 0 or perm_ind == len(perm):
            torch.save(checkpoint_state(i), os.path.join(basedir, expname, f"{i + 1:06d}.tar"))

    if writer is not None:
        writer.close()
    print("[TRAIN] Done.")


def _save_debug_render(rendering_output, args, sample, i, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    keys = ["intensity_map", "target", "confidence_maps", "attenuation_coeff",
            "reflection_coeff", "scatter_amplitude", "occ_pred", "occ_gt"]
    keys = [k for k in keys if k in rendering_output]
    plt.figure(figsize=(16, 8))
    for j, key in enumerate(keys):
        ax = plt.subplot(2, 4, j + 1)
        plt.title(key)
        img = rendering_output[key].detach().cpu().numpy()[0, 0].T
        if key in ("intensity_map", "target"):
            im = ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
        elif key in ("confidence_maps", "occ_pred", "occ_gt"):
            im = ax.imshow(img, cmap="viridis", vmin=0.0, vmax=1.0)
        else:
            im = ax.imshow(img, cmap="viridis")
        plt.colorbar(im, ax=ax, location="bottom")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{i + 1:07d}_{sample['dataset_name']}.png"),
                bbox_inches="tight", dpi=150)
    plt.close()


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    if torch.cuda.is_available():
        torch.set_default_device("cuda")
        print("GPU:", torch.cuda.get_device_name(0))
    train()
