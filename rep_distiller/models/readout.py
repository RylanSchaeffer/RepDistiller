from __future__ import print_function

import torch
import torch.nn as nn
from typing import Tuple


class MLPReadout(nn.Module):

    def __init__(self,
                 sizes: Tuple[int, ...],
                 use_bias: bool = True,
                 act=nn.LeakyReLU,
                 encoder: nn.Module = None,
                 train_only_readout: bool = True, ):

        super(MLPReadout, self).__init__()
        self.encoder = encoder
        self.train_only_readout = train_only_readout
        self.act_callable = act()
        if self.train_only_readout:
            if self.encoder is not None:
                self.encoder.requires_grad_(False)

        mlp_layers = []
        for i in range(len(sizes) - 1):
            mlp_layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=use_bias))
            if i < len(sizes) - 2:
                mlp_layers.append(act())
        self.readout = nn.Sequential(*mlp_layers)

    def forward(self,
                x: torch.Tensor,
                ) -> torch.Tensor:
        if self.encoder is not None:
            x = self.encoder(x)
            x = self.act_callable(x)
        return self.readout(x)


class LinearReadout(MLPReadout):

    def __init__(self,
                 dim_out: int,
                 dim_in: int,
                 encoder: nn.Module = None,
                 train_only_readout: bool = True,
                 use_bias: bool = True,
                 ):

        super(LinearReadout, self).__init__(
            sizes=(dim_in, dim_out),
            encoder=encoder,
            train_only_readout=train_only_readout,
            use_bias=use_bias,
        )
