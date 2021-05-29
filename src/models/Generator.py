import pblm
import sys
import torch
import torch.nn as nn


class Generator_A(pblm.PrebuiltLightningModule):
    def __init__(self, denoising=False):
        super().__init__(self.__class__.__name__)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(128, 256, normalize=False),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, 2500),
            nn.Tanh()
        )

    def forward(self, z):
        z = self.model(z)
        return z
