from . import pblm
import sys
import torch
import torch.nn as nn


class ModelA(pblm.PrebuiltLightningModule):
    def __init__(self, denoising=False):
        super().__init__()

        self.model_tags.append(self.__class__.__name__)
        self.set_model_name(self.__class__.__name__)

        # Model Layer Declaration
        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, stride=2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, stride=2)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=5, stride=2)
        self.dense1 = nn.Linear(64*309, 512)
        self.dense2 = nn.Linear(512, 256)
        self.dense3 = nn.Linear(256, 2)

    def forward(self, x):

        x = x.reshape(x.shape[0], 1, -1)

        # Convolutional Layer
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.conv3(x)
        x = nn.functional.relu(x)

        # Flattening
        x = x.reshape(x.shape[0], -1)

        # Dense Layers
        x = self.dense1(x)
        x = nn.functional.relu(x)
        x = self.dense2(x)
        x = nn.functional.relu(x)
        x = self.dense3(x)

        return x
