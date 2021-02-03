from . import pblm
from . import CNN
from . import Generator
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F


class GAN_A(pblm.PrebuiltLightningModule):
    def __init__(self, denoising=False):
        super().__init__(self.__class__.__name__)

        # networks
        self.generator = Generator.Generator_A()
        self.discriminator = CNN.Discriminator_A()

    def forward(self, x):
        return self.discriminator(x)

    def adversarial_loss(self, y_hat, y):
        return F.cross_entropy(y_hat, y)

    def training_step(self, batch, batch_idx, optimizer_idx):
        data, targets = batch

        # sample noise
        z = torch.randn(data.shape[0], 128)
        z = z.type_as(data)

        # train generator
        if optimizer_idx == 0:

            # ground truth result (ie: all fake)
            # what should label be
            # if 0 or 1 loss added?
            # adversarial loss is binary cross-entropy
            x_hat = self.generator(z)
            y_hat = self.discriminator(x_hat)

            y = []
            # artificallly checks if discrimator guessed not fake
            for y_idx in torch.argmax(y_hat, dim=1):
                if y_idx < 2:
                    y.append(y_idx.item())
                else:
                    y.append(2)
            y = torch.tensor(y).type_as(targets)
            g_loss = self.criterion(y_hat, y)

            self.log("g-loss", g_loss, prog_bar=False,
                     on_step=True, on_epoch=False)

            return g_loss

        # train discriminator
        if optimizer_idx == 1:
            # Measure discriminator's ability to classify real from generated samples

            outputs = self.discriminator(data)
            real_loss = self.criterion(outputs, targets)

            # how well can it label as fake?
            y_fake = torch.add(torch.zeros(data.shape[0]), 2)
            y_fake = y_fake.type_as(targets)

            x_hat = self.generator(z)
            y_hat = self.discriminator(x_hat)

            fake_loss = self.criterion(y_hat, y_fake)

            # discriminator loss is the average of these
            d_loss = (real_loss + fake_loss) / 2

            # Logs metrics
            metrics = self.metrics_step(outputs, targets, d_loss, prefix="")

            self.log_step(metrics, prog_bar=False,
                          on_step=True, on_epoch=False)
            return metrics

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(
            self.generator.parameters(), lr=1e-5)
        opt_d = torch.optim.Adam(
            self.discriminator.parameters(), lr=1e-5)
        return [opt_g, opt_d], []

    def validation_setp(self, batch, batch_idx, optimizer_idx):
        if optimizer_idx == 1:
            super().validation_setp(batch, batch_idx)
