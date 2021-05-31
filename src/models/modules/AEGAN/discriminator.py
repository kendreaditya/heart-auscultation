import torch
from torch import nn


class Discriminator(nn.Module):
    def __init__(self, classes):
        super(Discriminator, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d((512, 1))
        self.fc = nn.Linear(512, classes)

    def forward(self, x):
        x = self.pool(x)
        x = x.reshape(-1, 512)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    discriminator = Discriminator(4)
    out = discriminator(torch.rand(5, 512, 7))
    print(out.shape)
