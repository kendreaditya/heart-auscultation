import sys
from datetime import datetime

import torch
import wandb
import numpy as np
import pandas as pd
import torch.nn as nn
import pytorch_lightning as pl

from torch.utils.data import TensorDataset, DataLoader
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.metrics.functional import accuracy, precision, recall, stat_scores_multiple_classes
from sklearn import preprocessing, metrics, model_selection


class wav2wav(pl.LightningModule):
    def __init__(self, Encoder, Decoder):
        super(wav2wav, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)

        return {"latent": latent,
                "reconstruction": reconstruction}


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv1d(1, 512, kernel_size=(10,),
                               stride=(5,), bias=False)
        self.group_norm = nn.GroupNorm(512, 512, eps=1e-05, affine=True)
        self.conv2 = nn.Conv1d(512, 512, kernel_size=(3,),
                               stride=(2,), bias=False)
        self.conv3 = nn.Conv1d(512, 512, kernel_size=(3,),
                               stride=(2,), bias=False)
        self.conv4 = nn.Conv1d(512, 512, kernel_size=(3,),
                               stride=(2,), bias=False)
        self.conv5 = nn.Conv1d(512, 512, kernel_size=(3,),
                               stride=(2,), bias=False)
        self.conv6 = nn.Conv1d(512, 512, kernel_size=(2,),
                               stride=(2,), bias=False)
        self.conv7 = nn.Conv1d(512, 512, kernel_size=(2,),
                               stride=(2,), bias=False)
        self.group_norm_last = nn.GroupNorm(1, 1, eps=1e-05, affine=True)

    def forward(self, x):
        x = self.group_norm_last(x)
        x = self.conv1(x)
        x = self.group_norm(x)
        x = self.conv2(x)
        x = self.group_norm(x)
        x = self.conv3(x)
        x = self.group_norm(x)
        x = self.conv4(x)
        x = self.group_norm(x)
        x = self.conv5(x)
        x = self.group_norm(x)
        x = self.conv6(x)
        x = self.group_norm(x)
        x = self.conv7(x)
        x = self.group_norm(x)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.conv1 = nn.ConvTranspose1d(512, 512, kernel_size=(
            2,), stride=(2,), bias=False, output_padding=1)
        self.conv2 = nn.ConvTranspose1d(
            512, 512, kernel_size=(2,), stride=(2,), bias=False)
        self.conv3 = nn.ConvTranspose1d(
            512, 512, kernel_size=(3,), stride=(2,), bias=False)
        self.conv4 = nn.ConvTranspose1d(512, 512, kernel_size=(
            3,), stride=(2,), bias=False, output_padding=1)
        self.conv5 = nn.ConvTranspose1d(
            512, 512, kernel_size=(3,), stride=(2,), bias=False)
        self.conv6 = nn.ConvTranspose1d(
            512, 512, kernel_size=(3,), stride=(2,), bias=False)
        self.group_norm = nn.GroupNorm(512, 512, eps=1e-05, affine=True)
        self.conv7 = nn.ConvTranspose1d(
            512, 1, kernel_size=(10,), stride=(5,), bias=False)
        self.group_norm_last = nn.GroupNorm(1, 1, eps=1e-05, affine=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.group_norm(x)
        x = self.conv2(x)
        x = self.group_norm(x)
        x = self.conv3(x)
        x = self.group_norm(x)
        x = self.conv4(x)
        x = self.group_norm(x)
        x = self.conv5(x)
        x = self.group_norm(x)
        x = self.conv6(x)
        x = self.group_norm(x)
        x = self.conv7(x)
        x = self.group_norm_last(x)
        return x
