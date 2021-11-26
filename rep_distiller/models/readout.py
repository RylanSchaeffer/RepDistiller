from __future__ import print_function

import torch
import torch.nn as nn


#########################################
# ===== Readout Network ===== #
#########################################

class LinearReadout(nn.Module):

    def __init__(self,
                 dim_out: int,
                 dim_in: int,
                 encoder: nn.Module = None,
                 only_readout: bool = True,
                 use_bias: bool = True,
                 ):

        super(LinearReadout, self).__init__()
        self.encoder = encoder
        self.only_readout = only_readout
        if self.only_readout:
            self.encoder.requires_grad_(True)
        self.readout = nn.Linear(dim_in, dim_out, bias=use_bias)

    def forward(self, x):
        if self.encoder is not None:
            x = self.encoder(x)
        return self.readout(x)


class MLPReadout(nn.Module):

    def __init__(self,
                 dim_out: int = 10,
                 dim_in: int = None,
                 encoder: nn.Module = None,
                 only_readout: bool = True,
                 use_bias: bool = True,
                 dropout_prob: float = 0.1):
        super(MLPReadout, self).__init__()

        self.encoder = encoder
        self.only_readout = only_readout
        if self.only_readout:
            self.encoder.requires_grad_(True)

        self.net = nn.Sequential(
            nn.Linear(dim_in, 200, bias=use_bias),
            nn.Dropout(p=dropout_prob),
            nn.BatchNorm1d(200),
            nn.ReLU(inplace=True),
            nn.Linear(200, dim_out, bias=use_bias),
        )

    def forward(self,
                x: torch.Tensor):
        if self.encoder is not None:
            x = self.encoder(x)
        return self.net(x)
