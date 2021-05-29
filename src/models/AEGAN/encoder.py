import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv1d(1, 512, kernel_size=(10,), stride=(5,), bias=False)
        self.group_norm = nn.GroupNorm(512, 512, eps=1e-05, affine=True)
        self.conv2 = nn.Conv1d(512, 512, kernel_size=(3,), stride=(2,), bias=False)
        self.conv3 = nn.Conv1d(512, 512, kernel_size=(3,), stride=(2,), bias=False)
        self.conv4 = nn.Conv1d(512, 512, kernel_size=(3,), stride=(2,), bias=False)
        self.conv5 = nn.Conv1d(512, 512, kernel_size=(3,), stride=(2,), bias=False)
        self.conv6 = nn.Conv1d(512, 512, kernel_size=(2,), stride=(2,), bias=False)
        self.conv7 = nn.Conv1d(512, 512, kernel_size=(2,), stride=(2,), bias=False)
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


if __name__ == "__main__":
    x = torch.rand(1, 1, 2500)
    encoder = Encoder()
    print(encoder(x).shape)
