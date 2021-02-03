from . import pblm
import torch
import torch.nn as nn


class BiLSTM_A(pblm.PrebuiltLightningModule):
    def __init__(self):
        super().__init__(self.__class__.__name__)

        self.hidden_size = 256
        self.num_layers = 4
        self.input_size = 5*500
        self.num_classes = 3

        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers,
                            batch_first=True, bidirectional=True)
        self.dense = nn.Linear(self.hidden_size*2, self.num_classes)

    def forward(self, x):
        x = x.reshape(x.shape[0], 1, -1)

        # Hidden State
        h0 = torch.zeros(self.num_layers*2,
                         x.shape[0], self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers*2,
                         x.shape[0], self.hidden_size).to(self.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.dense(out[:, -1, :])
        return out
