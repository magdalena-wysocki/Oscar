"""Differentiable ultrasound renderer used by OSCAR.

The network predicts, at every 3D sample point, three acoustic parameters
(attenuation, reflection, scatter) plus an occupancy logit. ``render_method_3``
turns the per-point acoustic parameters of one B-mode frame into a simulated
ultrasound image following the convolutional ray-based model of
Wysocki et al. (Ultra-NeRF):

  * attenuation        -> depth-dependent signal decay (Beer-Lambert)
  * reflection         -> specular echoes at acoustic-impedance interfaces
  * backscatter        -> diffuse scattering convolved with a point-spread fn

``render_rays_us_with_pts`` runs the network on the precomputed 3D coordinates
of a frame (``poses_labels``) and splits the output into the rendered image and
the occupancy logits.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli


def cumsum_exclusive(tensor: torch.Tensor) -> torch.Tensor:
    """Exclusive cumulative sum along the last dimension (TF cumsum exclusive=True)."""
    cumsum = torch.cumsum(tensor, dim=-1)
    cumsum = torch.roll(cumsum, 1, dims=-1)
    cumsum[..., 0] = 0.0
    return cumsum


def gaussian_kernel(size: int, mean: float, std: float) -> torch.Tensor:
    """2D point-spread function used to blur the scatterer map."""
    d1 = torch.distributions.Normal(mean, std * 3)
    d2 = torch.distributions.Normal(mean, std)
    vals_x = d1.log_prob(torch.arange(-size, size + 1, dtype=torch.float32)).exp()
    vals_y = d2.log_prob(torch.arange(-size, size + 1, dtype=torch.float32)).exp()
    kernel = torch.einsum("i,j->ij", vals_x, vals_y)
    return kernel / torch.sum(kernel)


# Point-spread function, built once on CPU and moved to the active device when used.
_PSF_KERNEL = gaussian_kernel(3, 0.0, 1.0)


def render_method_3(raw: torch.Tensor) -> dict:
    """Render one ultrasound frame from per-point acoustic parameters.

    Args:
        raw: tensor of shape ``[W, H, 3]`` with channels
             ``[attenuation, reflection, scatter]``.

    Returns:
        Dict with the rendered ``intensity_map`` and intermediate acoustic maps
        (all shaped ``[1, 1, W, H]``), used by the loss and the regularizer.
    """
    raw = raw[None, None, ...]
    raw = raw.permute(0, 1, 3, 2, 4)
    batch_size, _, W, H, _ = raw.shape

    t_vals = torch.linspace(0.0, 1.0, H, device=raw.device)
    z_vals = t_vals.expand(batch_size, W, -1)
    dists = torch.abs(z_vals[..., :-1, None] - z_vals[..., 1:, None])
    dists = torch.squeeze(dists)
    dists = torch.cat([dists, dists[:, -1, None]], dim=-1)

    # ── Attenuation (Beer-Lambert decay accumulated along depth) ──
    attenuation_coeff = torch.abs(raw[..., 0])
    attenuation = torch.exp(-attenuation_coeff * dists).permute(0, 1, 3, 2)
    log_attenuation_total = cumsum_exclusive(torch.log(attenuation + 1e-12))
    attenuation_total = torch.exp(log_attenuation_total).permute(0, 1, 3, 2)

    # ── Reflection (transmission loss accumulated along depth) ──
    reflection_coeff = torch.sigmoid(raw[..., 1])
    reflection_transmission = (1.0 - reflection_coeff).permute(0, 1, 3, 2)
    log_reflection_total = cumsum_exclusive(torch.log(reflection_transmission + 1e-12))
    reflection_total = torch.exp(log_reflection_total).permute(0, 1, 3, 2)

    # ── Backscatter (random scatterers modulated by amplitude, blurred by PSF) ──
    density_coeff = torch.ones_like(reflection_coeff) * 0.75
    scatterers_density = RelaxedBernoulli(temperature=0.1, probs=density_coeff).sample()
    amplitude = torch.sigmoid(raw[..., 2])
    scatterers_map = scatterers_density * amplitude
    psf = _PSF_KERNEL.to(raw.device)
    psf_scatter = F.conv2d(scatterers_map, psf[None, None, ...], stride=1, padding="same")

    # ── Compose the final echo ──
    confidence_maps = attenuation_total * reflection_total
    b = confidence_maps * psf_scatter
    r = confidence_maps * reflection_coeff

    amplification_constant = torch.tensor(np.pi)
    alpha_amplification = lambda x: torch.log(1.0 + amplification_constant * x) * torch.log(
        1.0 + amplification_constant
    )
    r_amplified = alpha_amplification(r)
    intensity_map = b + r_amplified

    return {
        "intensity_map": intensity_map,
        "attenuation_coeff": attenuation_coeff,
        "reflection_coeff": reflection_coeff,
        "attenuation_total": attenuation_total,
        "reflection_total": reflection_total,
        "scatter_amplitude": amplitude,
        "confidence_maps": confidence_maps,
    }


def render_rays_us_with_pts(
    ray_batch,
    network_fn,
    network_query_fn,
    N_samples,
    pts,
    latent_codes=None,
    latent_id=None,
    **kwargs,
):
    """Query the network at the precomputed 3D points of one frame.

    The network output channels are ``[attenuation, reflection, scatter, occ]``.
    When ``return_only_occ`` is set (used to extract a dense 3D occupancy grid),
    only the occupancy logits are returned and the acoustic render is skipped.
    """
    raw = network_query_fn(ray_batch, network_fn, latent_codes, latent_id)
    occ = raw[..., -1:]
    raw = raw[..., :-1]

    if kwargs.get("return_only_occ"):
        return {"occ": occ}

    if kwargs.get("nisf"):
        # NISF baseline: no physics rendering. The single remaining channel IS the B-mode
        # intensity; shape it like render_method_3's intensity_map ([1, 1, H, W]).
        intensity_map = torch.sigmoid(raw[..., 0]).permute(1, 0)[None, None, ...]
        return {"intensity_map": intensity_map, "occ": occ}

    ret = render_method_3(raw)
    ret["occ"] = occ
    return ret
