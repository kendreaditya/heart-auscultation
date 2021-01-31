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
from pytorch_lightning.metrics.functional import accuracy, precision, recall, f1_score, fbeta_score, stat_scores_multiple_classes
from sklearn import preprocessing, metrics, model_selection


class PrebuiltLightningModule(pl.LightningModule):
    def __init__(self):
        super().__init__()

        # Metrics
        self.seed = np.random.randint(220)
        pl.seed_everything(seed=self.seed)

        # Loss
        self.criterion = nn.CrossEntropyLoss()

        # Run Name
        self.set_model_name()

        # Model Tags
        self.model_tags = [str(self.criterion)]

    def set_model_name(self):
        delimter = "\\"
        if "/" in sys.argv[0]:
            delimter = "/"
        self.file_name = sys.argv[0].split(delimter)[-1].replace(".py", "")
        timestamp = datetime.now().strftime("%Y%m%d%H%M")
        self.model_name = f"{self.file_name}-{timestamp}"

    def configure_optimizers(self):
        optimzer = torch.optim.Adam(self.parameters(), lr=1e-5)
        return optimzer

    def metrics_step(self, outputs, targets, loss, prefix=""):
        pred = torch.argmax(outputs, dim=1)

        accuracy_score = accuracy(pred, targets)
        learning_rate = self.optimizers().param_groups[0]['lr']

        tps, fps, tns, fns, sups = stat_scores_multiple_classes(pred, targets)
        stat_scores_table = pd.DataFrame(data={
            'TP': tps.cpu(),
            'FP': fps.cpu(),
            'TN': tns.cpu(),
            'FN': fns.cpu(),
            'SUP': sups.cpu()
        })

        return {f'{prefix}accuracy': accuracy_score,
                f'{prefix}loss': loss,
                f'{prefix}lr': learning_rate,
                f'{prefix}stats': stat_scores_table}

    def log_step(self, metrics, **kwargs):
        for key in metrics:
            if "stats" not in key:
                self.log(
                    f"{key}",
                    metrics[key],
                    prog_bar=kwargs["prog_bar"],
                    on_step=kwargs["on_step"],
                    on_epoch=kwargs["on_epoch"])
            else:
                continue
                self.log(
                    f"{key}",
                    wandb.Table(dataframe=metrics[key]),
                    prog_bar=kwargs["prog_bar"],
                    on_step=kwargs["on_step"],
                    on_epoch=kwargs["on_epoch"])

    def training_step(self, batch, batch_idx):
        data, targets = batch
        outputs = self.forward(data)
        loss = self.criterion(outputs, targets)

        # Logs metrics
        metrics = self.metrics_step(outputs, targets, loss, prefix="")

        self.log_step(metrics, prog_bar=False, on_step=True, on_epoch=False)
        return metrics

    # Only track validatoin loss for eniter dataset not batch
    def validation_step(self, batch, batch_idx):
        data, targets = batch
        outputs = self.forward(data)
        loss = self.criterion(outputs, targets)

        # Logs metrics
        metrics = self.metrics_step(
            outputs, targets, loss, prefix="validation-")
        return metrics

    # Can't take average of pd DataFrame
    def validation_epoch_end(self, outputs):
        avg_metrics = {key: 0.0 for key in outputs[0]}
        avg_metrics["validation-stats"] = pd.DataFrame(data={
            'TP': [0],
            'FP': [0],
            'TN': [0],
            'FN': [0],
            'SUP': [0]
        })

        n = len(outputs)

        for metrics in outputs:
            for key in metrics:
                if "stats" not in key:
                    avg_metrics[key] = ((n-1)*avg_metrics[key]+metrics[key])/n
                else:
                    avg_metrics[key] += metrics[key]

        self.log_step(avg_metrics, prog_bar=True, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        data, targets = batch
        outputs = self.forward(data)
        loss = self.criterion(outputs, targets)

        # Logs metrics
        metrics = self.metrics_step(outputs, targets, loss, prefix="test-")
        return metrics

    def test_epoch_end(self, outputs):
        avg_metrics = {key: 0.0 for key in outputs[0]}
        avg_metrics["test-stats"] = pd.DataFrame(data={
            'TP': [0],
            'FP': [0],
            'TN': [0],
            'FN': [0],
            'SUP': [0]
        })

        n = len(outputs)

        for metrics in outputs:
            for key in metrics:
                if "stats" not in key:
                    avg_metrics[key] = ((n-1)*avg_metrics[key]+metrics[key])/n
                else:
                    avg_metrics[key] += metrics[key]

        self.log_step(avg_metrics, prog_bar=False,
                      on_step=False, on_epoch=True)
