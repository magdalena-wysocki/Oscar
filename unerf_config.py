"""Command-line / config-file argument parser for the OSCAR pipeline.

OSCAR is a single network that jointly renders ultrasound images (via a
differentiable acoustic renderer) and predicts a 3D occupancy field, both
conditioned on a per-subject latent code. The same parser is shared by the
training script (``train.py``) and the test-time optimization script
(``test_time_optimize.py``).

Arguments can be passed on the command line or through a ``--config`` file
(see ``configs/``). Command-line values override config-file values.
"""

import configargparse


def config_parser():
    parser = configargparse.ArgumentParser()
    parser.add_argument("--config", is_config_file=True, help="config file path")

    # ── Experiment / IO ────────────────────────────────────────────────
    parser.add_argument("--expname", type=str, help="experiment name")
    parser.add_argument("--basedir", type=str, default="./logs/",
                        help="where to store checkpoints and logs")
    parser.add_argument("--datadir", type=lambda s: [item.strip() for item in s.split(",")],
                        help="comma-separated list of data directories (one per subject/scan)")
    parser.add_argument("--val_datadir", type=lambda s: [item.strip() for item in s.split(",")],
                        default=None,
                        help="comma-separated list of validation data dirs (same format as --datadir)")
    # ── Network architecture ───────────────────────────────────────────
    parser.add_argument("--netdepth", type=int, default=8, help="number of layers in the MLP")
    parser.add_argument("--netwidth", type=int, default=128, help="channels per layer")
    parser.add_argument("--output_ch", type=int, default=4,
                        help="network output channels: 3 acoustic (attenuation, reflection, "
                             "scatter) + 1 occupancy")
    parser.add_argument("--nisf", action="store_true",
                        help="NISF baseline: bypass the differentiable ultrasound renderer; the "
                             "acoustic head directly regresses the B-mode intensity (1 channel) "
                             "instead of the 3 acoustic parameters. The network then outputs "
                             "[intensity, occupancy]. Still label-free (trains on B-mode).")
    parser.add_argument("--latent_dim", type=int, default=128, help="per-subject latent code dimension")
    parser.add_argument("--i_embed", type=int, default=-1,
                        help="positional encoding: -1 for identity (default), >=0 for Fourier encoding")
    parser.add_argument("--multires", type=int, default=10,
                        help="log2 of max frequency for positional encoding (used when i_embed >= 0)")
    parser.add_argument("--N_samples", type=int, default=512, help="number of samples per ray")

    # ── Batching / memory ──────────────────────────────────────────────
    parser.add_argument("--chunk", type=int, default=4096 * 16,
                        help="rays processed in parallel; decrease if running out of memory")
    parser.add_argument("--netchunk", type=int, default=4096 * 16,
                        help="points sent through the network in parallel; decrease on OOM")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="steps to accumulate gradients before an optimizer step")

    # ── Optimization ───────────────────────────────────────────────────
    parser.add_argument("--n_iters", type=int, default=100000, help="number of training/optimization iterations")
    parser.add_argument("--lrate", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--random_seed", type=int, default=-1, help="set >=0 for deterministic behaviour")

    # ── Losses ─────────────────────────────────────────────────────────
    parser.add_argument("--loss", type=str, default="ssim", choices=["ssim", "l2", "l1"],
                        help="image reconstruction loss")
    parser.add_argument("--ssim_lambda", type=float, default=0.75, help="weight of the SSIM term")
    parser.add_argument("--ssim_filter_size", type=int, default=7, help="SSIM gaussian window size")
    parser.add_argument("--occ_lambda", type=float, default=0.3, help="weight of the occupancy (completion) loss")
    parser.add_argument("--completion", action="store_true",
                        help="enable the occupancy/shape-completion head")
    parser.add_argument("--occ_loss_model_only", action="store_true",
                        help="backpropagate the occupancy loss into the network weights only "
                             "(not the latent codes)")
    parser.add_argument("--dice_loss", action="store_true",
                        help="add a soft-Dice term on the predicted occupancy, alongside the BCE "
                             "occupancy loss (independent of --nisf)")
    parser.add_argument("--dice_lambda", type=float, default=1.0,
                        help="weight of the --dice_loss term")
    parser.add_argument("--reg_latent", type=float, default=1e-3,
                        help="L2 regularization weight on the latent code (used during calibration)")

    # ── Physics regularization (acoustic renderer) ─────────────────────
    parser.add_argument("--reg", action="store_true", help="enable physics regularization")
    parser.add_argument("--r_tv_penalty", type=float, default=0.00001, help="total-variation penalty weight")
    parser.add_argument("--r_lcc_penalty", type=float, default=0.001,
                        help="local-normalized-cross-correlation penalty weight")
    parser.add_argument("--r_max_reflection", type=float, default=0.34,
                        help="reflection weighting constant in the TV penalty")
    parser.add_argument("--r_warm_up_it", type=int, default=10000,
                        help="iterations before regularization is switched on")

    # ── Checkpoints ────────────────────────────────────────────────────
    parser.add_argument("--ft_path", type=str, default=None,
                        help="specific checkpoint .tar to reload (otherwise the latest in basedir/expname)")
    parser.add_argument("--latent_optimize_ckpt_path", type=str, default=None,
                        help="trained checkpoint to load for test-time latent optimization")
    parser.add_argument("--latent_avg_start", action="store_true",
                        help="initialize the test-time latent from the average of the trained latent codes "
                             "(otherwise seeded from a single trained latent code)")

    # ── Validation / test-time calibration ─────────────────────────────
    parser.add_argument("--val_max_cal_steps", type=int, default=10000,
                        help="max calibration (latent optimization) steps per validation subject")
    parser.add_argument("--val_cal_eval_every", type=int, default=500,
                        help="evaluate validation metrics every N calibration steps")
    parser.add_argument("--val_grid_res", type=int, default=128,
                        help="occupancy grid resolution during validation")
    parser.add_argument("--early_stop_patience", type=int, default=-1,
                        help="stop training after this many validation epochs without improvement "
                             "(negative disables)")
    parser.add_argument("--test_metric_grid_res", type=int, default=128,
                        help="occupancy grid resolution for per-epoch test metric evaluation")

    # ── Logging ────────────────────────────────────────────────────────
    parser.add_argument("--tensorboard", action="store_true", help="log scalars/images to TensorBoard")
    parser.add_argument("--i_print", type=int, default=5000, help="console + debug-render frequency")
    parser.add_argument("--i_weights", type=int, default=10000, help="checkpoint save frequency")

    return parser
