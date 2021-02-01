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

    # wandb table
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
                self.log(
                    f"{key}",
                    CustomTable(metrics[key]),
                    prog_bar=False,
                    on_step=kwargs["on_step"],
                    on_epoch=kwargs["on_epoch"],
                    reduce_fx=lambda x: None,
                    tbptt_reduce_fx=lambda x: None,
                    sync_dist=False,
                    enable_graph=False)

    def training_step(self, batch, batch_idx):
        data, targets = batch
        outputs = self.forward(data)
        loss = self.criterion(outputs, targets)

        # Logs metrics
        metrics = self.metrics_step(outputs, targets, loss, prefix="")

        self.log_step(metrics, prog_bar=False, on_step=True, on_epoch=False)
        return metrics

    def validation_step(self, batch, batch_idx):
        data, targets = batch
        outputs = self.forward(data)
        loss = self.criterion(outputs, targets)

        # Logs metrics
        metrics = self.metrics_step(
            outputs, targets, loss, prefix="validation-")
        return metrics

    # check metric average calculation
    def validation_epoch_end(self, outputs):
        avg_metrics = {key: 0.0 for key in outputs[0]}
        rows = outputs[0]["validation-stats"].index
        avg_metrics["validation-stats"] = pd.DataFrame(data={
            'TP': [0 for _ in rows],
            'FP': [0 for _ in rows],
            'TN': [0 for _ in rows],
            'FN': [0 for _ in rows],
            'SUP': [0 for _ in rows]
        })

        for n, metrics in enumerate(outputs):
            for key in metrics:
                if "stats" not in key:
                    avg_metrics[key] = (
                        (n)*avg_metrics[key]+metrics[key])/(n+1)
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
        rows = outputs[0]["test-stats"].index
        avg_metrics["test-stats"] = pd.DataFrame(data={
            'TP': [0 for _ in rows],
            'FP': [0 for _ in rows],
            'TN': [0 for _ in rows],
            'FN': [0 for _ in rows],
            'SUP': [0 for _ in rows]
        })

        for n, metrics in enumerate(outputs):
            for key in metrics:
                if "stats" not in key:
                    avg_metrics[key] = (
                        (n)*avg_metrics[key]+metrics[key])/(n+1)
                else:
                    avg_metrics[key] += metrics[key]

        self.log_step(avg_metrics, prog_bar=False,
                      on_step=False, on_epoch=True)


class CustomTable(wandb.Table):
    def __init__(self, dataframe):
        super().__init__(list(dataframe.values), dataframe=dataframe)
        self._list = list(dataframe.values)

    def __len__(self):
        """List length"""
        return len(self._list)

    def __getitem__(self, ii):
        """Get a list item"""
        return self._list[ii]

    def __str__(self):
        return str(self._list)
