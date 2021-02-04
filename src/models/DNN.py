from . import pblm
import torch
import torch.nn as nn


class DNN_A(pblm.PrebuiltLightningModule):
    def __init__(self):
        super().__init__(self.__class__.__name__)

        # Model Tags
        self.model_tags.append(self.__class__.__name__)

        # Model Layer Declaration
        self.dense1 = nn.Linear(5*500, 1024)
        self.dense2 = nn.Linear(1024, 512)
        self.dense3 = nn.Linear(512, 128)
        self.dense4 = nn.Linear(128, 64)
        self.dense5 = nn.Linear(64, 3)

    def forward(self, x):
        x = self.dense1(x)
        x = nn.functional.relu(x)
        x = self.dense2(x)
        x = nn.functional.relu(x)
        x = self.dense3(x)
        x = nn.functional.relu(x)
        x = self.dense4(x)
        x = self.dense5(x)

        return x
