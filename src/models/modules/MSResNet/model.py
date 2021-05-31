from . import pblm
from . import MSResNet
from . import Generator
import torch
from torch import nn


class model(pblm.PrebuiltLightningModule):
    def __init__(self, num_classes, sample_length):
        super().__init__(self.__class__.__name__)
        self.num_classes = num_classes
        self.sample_length = sample_length
        self.generator = Generator.ConditionalGenerator(sample_length, num_classes)
        self.discriminator = MSResNet.MSResNet(1, num_classes=num_classes)

    def forward(self, x, y=None, seeds=None):
        if seeds != None and y != None:
            x_g = self.generator(seeds, y)
            x_g = (x_g - torch.mean(x_g)) / (torch.std(x_g))
            y_g = self.discriminator(x_g)
        else:
            y_g = None

        y_hat = self.discriminator(x)

        return {"y_hat": y_hat, "y_g": y_g}
        # return y_hat

    def training_step(self, batch, batch_idx, optimizer_idx):
        data, targets = batch

        # sample noise
        z = torch.randn(data.shape[0], 64)
        z = z.type_as(data)

        # train generator
        if optimizer_idx == 0:
            for param in self.discriminator.parameters():
                param.requires_grad = False
            for param in self.generator.parameters():
                param.requires_grad = True

            # ground truth result (ie: all fake)
            # what should label be
            # if 0 or 1 loss added?
            # adversarial loss is binary cross-entropy
            x_hat = self.generator(z, targets)
            x_hat = x_hat.reshape(-1, 1, self.sample_length)
            y_hat = self.discriminator(x_hat)

            # artificallly checks if discrimator guessed not fake
            g_loss = self.criterion(y_hat, targets)

            self.log("g-loss", g_loss, prog_bar=False,
                     on_step=True, on_epoch=False)

            for param in self.discriminator.parameters():
                param.requires_grad = True
            for param in self.generator.parameters():
                param.requires_grad = True

            return g_loss

        # train discriminator
        if optimizer_idx == 1:
            # Measure discriminator's ability to classify real from generated samples
            for param in self.discriminator.parameters():
                param.requires_grad = True
            for param in self.generator.parameters():
                param.requires_grad = False

            outputs = self.discriminator(data)
            real_loss = self.criterion(outputs, targets)

            # how well can it label as fake?
            y_fake = torch.add(torch.zeros(data.shape[0]), self.num_classes - 1)
            y_fake = y_fake.type_as(targets)

            x_hat = self.generator(z, targets)
            x_hat = x_hat.reshape(-1, 1, self.sample_length)
            y_hat = self.discriminator(x_hat)

            fake_loss = self.criterion(y_hat, y_fake)

            self.log("d-fake-loss", fake_loss, prog_bar=False,
                     on_step=True, on_epoch=False)

            self.log("d-real-loss", real_loss, prog_bar=False,
                     on_step=True, on_epoch=False)

            # discriminator loss is the average of these
            d_loss = (real_loss + fake_loss) / 2

            # Logs metrics
            metrics = self.metrics_step(outputs, targets, d_loss, prefix="")

            self.log_step(metrics, prog_bar=False,
                          on_step=True, on_epoch=False)

            for param in self.discriminator.parameters():
                param.requires_grad = True
            for param in self.generator.parameters():
                param.requires_grad = True

            return metrics

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(
            self.generator.parameters(), lr=1e-5)
        opt_d = torch.optim.Adam(
            self.discriminator.parameters(), lr=1e-5)
        lr_g = torch.optim.lr_scheduler.StepLR(opt_g, step_size=5, gamma=0.05)
        lr_d = torch.optim.lr_scheduler.StepLR(opt_d, step_size=5, gamma=0.05)
        return [opt_g, opt_d], [lr_g, lr_d]

    def validation_step(self, batch, batch_idx):
        data, targets = batch
        outputs = self.forward(data)["y_hat"]
        loss = self.criterion(outputs, targets)

        # Logs metrics
        metrics = self.metrics_step(
            outputs, targets, loss, prefix="validation-")
        return metrics

    def test_step(self, batch, batch_idx):
        data, targets = batch
        outputs = self.forward(data)["y_hat"]
        loss = self.criterion(outputs, targets)

        # Logs metrics
        metrics = self.metrics_step(outputs, targets, loss, prefix="test-")
        return metrics
