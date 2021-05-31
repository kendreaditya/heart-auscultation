import torch
import torch.nn as nn
import numpy as np


class ConditionalGenerator(nn.Module):
    def __init__(self, sample_length, classes):
        super(ConditionalGenerator, self).__init__()

        self.label_emb = nn.Embedding(classes, classes)
        self.classes = classes
        self.sample_length = sample_length

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(64 + classes, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, self.sample_length),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # Concatenate label embedding and image to produce input
        emb = self.label_emb(labels).reshape(-1, self.classes)
        x = torch.cat((emb, noise), -1)
        x = self.model(x)
        return x


if __name__ == "__main__":
    model = ConditionalGenerator(2500, 4)
    x = torch.rand(5, 4)
    y = torch.rand(5, 1).long()
    print(model(x, y).shape)
