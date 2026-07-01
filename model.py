"""OSCAR network definition.

A single coordinate-based MLP maps an embedded 3D point (concatenated with a
per-subject latent code) to ``output_ch`` channels. The first three channels
parameterize the acoustic renderer (attenuation, reflection, scatter) and the
last channel is the occupancy logit. See ``rendering.render_method_3`` and
``rendering.render_rays_us_with_pts`` for how the outputs are consumed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NeRF(nn.Module):
    def __init__(self, D=8, W=128, input_ch=3, output_ch=4, skips=(4,)):
        super().__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.skips = list(skips)

        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)]
            + [
                nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W)
                for i in range(D - 1)
            ]
        )
        for layer in self.pts_linears:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

        self.output_linear = nn.Linear(W, output_ch)
        nn.init.xavier_uniform_(self.output_linear.weight)
        nn.init.zeros_(self.output_linear.bias)

    def forward(self, x):
        h = x
        for i, layer in enumerate(self.pts_linears):
            h = F.relu(layer(h))
            if i in self.skips:
                h = torch.cat([x, h], -1)
        return self.output_linear(h)
