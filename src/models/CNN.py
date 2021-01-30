from . import pblm
import sys
import torch
import torch.nn as nn


class ModelA(pblm.PrebuiltLightningModule):
    def __init__(self, denoising=False):
        super().__init__()

        self.model_tags.append(self.__class__.__name__)

        # Model Layer Declaration
        #self.embeding1 = nn.Embedding(2500, 200)
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5)
        self.dense1 = nn.Linear(32*2496, 256)
        self.dense2 = nn.Linear(256, 2)

    def forward(self, x):

        # Embedding Layer
        #x = self.embeding1(x)

        x = x.reshape(x.shape[0], 1, -1)
        # Convolutional Layer
        x = self.conv1(x)
        x = nn.functional.relu(x)

        # Flattening
        x = x.reshape(x.shape[0], -1)

        # Dense Layers
        x = self.dense1(x)
        x = nn.functional.relu(x)
        x = self.dense2(x)

        return x
