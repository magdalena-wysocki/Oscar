"""Core OSCAR helpers: positional encoding, network construction, rendering
entry point, losses and physics regularization.

The OSCAR model is a single MLP (``model.NeRF``) that, conditioned on a
per-subject latent code, predicts acoustic parameters + occupancy at each 3D
point of an ultrasound frame. Training optimizes the network weights and all
subject latent codes jointly; test-time optimization freezes the network and
optimizes a single latent code for an unseen subject.
"""

import os

import torch
import torch.nn as nn

from model import NeRF
from rendering import render_rays_us_with_pts

# Mean-squared error helper (also used as the L2 image loss term).
img2mse = lambda x, y: torch.mean((x - y) ** 2)


# ──────────────────────────────────────────────────────────────────────
# Positional encoding
# ──────────────────────────────────────────────────────────────────────

class Embedder:
    """Standard NeRF Fourier positional encoding (used when i_embed >= 0)."""

    def __init__(self, input_dims, max_freq_log2, num_freqs, device,
                 include_input=True, log_sampling=True,
                 periodic_fns=(torch.sin, torch.cos)):
        embed_fns = []
        out_dim = 0
        if include_input:
            embed_fns.append(lambda x: x)
            out_dim += input_dims

        if log_sampling:
            freq_bands = 2.0 ** torch.linspace(0.0, max_freq_log2, steps=num_freqs, device=device)
        else:
            freq_bands = torch.linspace(2.0 ** 0.0, 2.0 ** max_freq_log2, steps=num_freqs, device=device)

        for freq in freq_bands:
            for p_fn in periodic_fns:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += input_dims

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, device, i=0, input_dim=3):
    """Return ``(embed_fn, out_dim)``.

    ``i == -1`` selects the identity encoding (the OSCAR default); a non-negative
    ``i`` selects a Fourier encoding with ``multires`` frequency bands.
    """
    if i == -1:
        return nn.Identity(), input_dim

    embedder = Embedder(
        input_dims=input_dim,
        max_freq_log2=multires - 1,
        num_freqs=multires,
        device=device,
    )
    return (lambda x, eo=embedder: eo.embed(x)), embedder.out_dim


# ──────────────────────────────────────────────────────────────────────
# Network evaluation
# ──────────────────────────────────────────────────────────────────────

def batchify(fn, chunk):
    """Wrap ``fn`` so it is applied to ``chunk``-sized minibatches of points."""
    if chunk is None:
        return fn

    def ret(inputs):
        return torch.cat([fn(inputs[i:i + chunk]) for i in range(0, inputs.shape[0], chunk)], 0)

    return ret


def run_network(inputs, fn, embed_fn, latent_codes=None, latent_id=None, netchunk=1024 * 64):
    """Embed ``inputs``, concatenate the subject latent code, and apply ``fn``."""
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)
    if latent_codes is not None and latent_id is not None:
        latent = latent_codes(latent_id)
        latent = latent.unsqueeze(0).expand(embedded.shape[0], -1)
        embedded = torch.cat([embedded, latent], dim=-1)
    outputs_flat = batchify(fn, netchunk)(embedded)
    return torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])


def batchify_rays(rays_flat=None, chunk=1024 * 64 * 8, **kwargs):
    """Render the precomputed points of a frame in ``chunk``-sized batches."""
    if kwargs.get("pts") is None:
        raise ValueError("batchify_rays requires precomputed 'pts' in kwargs")
    pts = kwargs["pts"]
    all_ret = {}
    for i in range(0, pts.shape[0], chunk):
        ret = render_rays_us_with_pts(pts[i:i + chunk], **kwargs)
        for k, v in ret.items():
            all_ret.setdefault(k, []).append(v)
    return {k: torch.cat(v, 0) for k, v in all_ret.items()}


def render_us(chunk=1024 * 32, **kwargs):
    """Render one ultrasound frame from its precomputed 3D points (``pts``)."""
    if kwargs.get("pts") is None:
        raise ValueError("render_us requires precomputed 'pts' in kwargs")
    return batchify_rays(chunk=chunk, **kwargs)


# ──────────────────────────────────────────────────────────────────────
# Network construction (train + test-time-optimize modes)
# ──────────────────────────────────────────────────────────────────────

def create_one_network(args, device, mode="train"):
    """Build the OSCAR network, latent codes and optimizer.

    mode="train":
        Trainable network + one latent code per training subject. Resumes from
        the latest checkpoint in ``basedir/expname`` (or ``--ft_path``) if present.
    mode="optimize":
        Load a trained checkpoint (``--latent_optimize_ckpt_path``), freeze the
        network, and create a single trainable latent code for the unseen
        subject (initialized from the average or a single trained latent code).
    """
    embed_fn, input_ch = get_embedder(args.multires, device, args.i_embed)
    input_ch = input_ch + args.latent_dim
    # NISF replaces the 3 acoustic channels with a single direct B-mode intensity channel,
    # so the network outputs [intensity, occupancy] instead of [atten, refl, scatter, occ].
    output_ch = 2 if getattr(args, "nisf", False) else args.output_ch
    print(f"Input channel {input_ch}, output channel {output_ch}")

    model = NeRF(D=args.netdepth, W=args.netwidth, input_ch=input_ch,
                 output_ch=output_ch, skips=[4]).to(device)

    def network_query_fn(inputs, network_fn, latent_codes=None, latent_id=None):
        return run_network(inputs, network_fn, embed_fn=embed_fn, netchunk=args.netchunk,
                           latent_codes=latent_codes, latent_id=latent_id)

    start = 0

    if mode == "train":
        n_subjects = len(getattr(args, "dataset_names", [1]))
        latent_codes = nn.Embedding(max(1, n_subjects), args.latent_dim).to(device)
        nn.init.normal_(latent_codes.weight, mean=0.0, std=1e-3)

        grad_vars = list(model.parameters()) + list(latent_codes.parameters())
        optimizer = torch.optim.Adam(grad_vars, lr=args.lrate, betas=(0.9, 0.999))

        ckpt_path = _latest_checkpoint(args)
        if ckpt_path is not None:
            print(f"[RESUME] Reloading from {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=device)
            start = ckpt["global_step"]
            model.load_state_dict(ckpt["network_fn_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if ckpt.get("latent_codes") is not None:
                latent_codes.load_state_dict(ckpt["latent_codes"])
    else:  # optimize
        if args.latent_optimize_ckpt_path is None:
            raise ValueError("mode='optimize' requires --latent_optimize_ckpt_path")
        print(f"[OPTIMIZE] Loading trained checkpoint {args.latent_optimize_ckpt_path}")
        ckpt = torch.load(args.latent_optimize_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["network_fn_state_dict"])
        for p in model.parameters():
            p.requires_grad = False

        latent_codes = nn.Embedding(1, args.latent_dim).to(device)
        trained = ckpt["latent_codes"]["weight"]
        with torch.no_grad():
            if args.latent_avg_start:
                latent_codes.weight.data[:] = trained.mean(dim=0, keepdim=True)
                print("[OPTIMIZE] Initialized latent from the average of trained latents")
            else:
                seed_idx = int(max(args.random_seed, 0)) % trained.shape[0]
                latent_codes.weight.data[:] = trained[seed_idx:seed_idx + 1]
                print(f"[OPTIMIZE] Initialized latent from trained subject index {seed_idx}")
        optimizer = torch.optim.Adam(latent_codes.parameters(), lr=args.lrate, betas=(0.9, 0.999))

    render_kwargs = {
        "network_query_fn": network_query_fn,
        "N_samples": args.N_samples,
        "network_fn": model,
        "latent_codes": latent_codes,
        "nisf": getattr(args, "nisf", False),
    }
    render_kwargs_test = dict(render_kwargs)
    return render_kwargs, render_kwargs_test, start, optimizer


def _latest_checkpoint(args):
    """Return the checkpoint to resume from, or None."""
    if args.ft_path not in (None, "None"):
        return args.ft_path
    ckpt_dir = os.path.join(args.basedir, args.expname)
    if not os.path.isdir(ckpt_dir):
        return None
    tars = sorted(f for f in os.listdir(ckpt_dir) if f.endswith(".tar") and f[:1].isdigit())
    return os.path.join(ckpt_dir, tars[-1]) if tars else None


# ──────────────────────────────────────────────────────────────────────
# Losses and regularization
# ──────────────────────────────────────────────────────────────────────

def compute_loss(output, target, args, losses, i):
    """Image reconstruction loss. Returns a dict mapping name -> (weight, value)."""
    loss = {}
    if args.loss == "l2" or i < args.r_warm_up_it:
        loss["l2"] = (1.0, img2mse(output, target))
    elif args.loss == "ssim":
        loss["ssim"] = (args.ssim_lambda, losses["ssim"](output, target))
        loss["l2"] = (1.0 - args.ssim_lambda, img2mse(output, target))
    elif args.loss == "l1":
        loss["l1"] = (1.0, losses["l1"](output, target))
    return loss


def soft_dice_loss(pred, target, eps=1e-6):
    """Soft Dice loss (1 - Dice) on occupancy probabilities in [0, 1]."""
    pred = pred.reshape(-1)
    target = target.reshape(-1)
    inter = (pred * target).sum()
    denom = pred.sum() + target.sum()
    return 1.0 - (2.0 * inter + eps) / (denom + eps)


def compute_regularization(rendering_output, reg_funcs, weights=(0.01, 0.00001, 0.34)):
    """Physics priors on the acoustic maps.

    * lcc_penalty: local normalized cross-correlation between scatter amplitude
      and attenuation (couples the two acoustic channels).
    * tv_penalty: reflection-weighted total variation on the scatter amplitude
      (encourages piecewise-smooth speckle away from interfaces).
    """
    lncc = reg_funcs["lncc"]
    lncc_w, tv_w, refl_max = weights
    reg = {}

    reg["lcc_penalty"] = (lncc_w, lncc(rendering_output["scatter_amplitude"],
                                       rendering_output["attenuation_coeff"]))

    amp = rendering_output["scatter_amplitude"]
    dy = amp[:, :, :, 1:] - amp[:, :, :, :-1]
    dy = torch.cat([dy, dy[:, :, :, -1:]], dim=-1)
    dx = amp[:, :, 1:, :] - amp[:, :, :-1, :]
    dx = torch.cat([dx, dx[:, :, -1:, :]], dim=-2)

    reflection_weight = refl_max - rendering_output["reflection_coeff"]
    tv = (torch.sum(reflection_weight * torch.abs(dy.squeeze()))
          + torch.sum(reflection_weight * torch.abs(dx.squeeze())))
    reg["tv_penalty"] = (tv_w, tv)
    return reg
