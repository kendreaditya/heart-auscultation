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
from torchsampler import ImbalancedDatasetSampler


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
            'TP': tps,
            'FP': fps,
            'TN': tns,
            'FN': fns,
            'SUP': sups
        })

        return {f'{prefix}accuracy': accuracy_score,
                f'{prefix}loss': loss,
                f'{prefix}lr': learning_rate}
        # f'{prefix}stats': wandb.Table(dataframe=stat_scores_table)}

    def training_step(self, batch, batch_idx):
        data, targets = batch
        outputs = self.forward(data)
        loss = self.criterion(outputs, targets)

        # Logs metrics
        metrics = self.metrics_step(outputs, targets, loss, prefix="train-")

        for key in metrics:
            self.log(
                f"train-{key}", metrics[key], prog_bar=False, on_step=True, on_epoch=False)
        print(metrics)
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
        avg_metrics = {key: [] for key in outputs[0]}
        for metrics in outputs:
            for key in avg_metrics:
                if 'stats' in key:
                    continue
                avg_metrics[key].append(metrics[key])
        avg_metrics = {key: torch.as_tensor(
            avg_metrics[key]).mean() for key in avg_metrics}
        for key in avg_metrics:
            self.log(key, avg_metrics[key], prog_bar=True,
                     on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        data, targets = batch
        outputs = self.forward(data)
        loss = self.criterion(outputs, targets)

        # Logs metrics
        metrics = self.metrics_step(outputs, targets, loss, prefix="test-")
        return metrics

    def test_epoch_end(self, outputs):
        avg_metrics = {key: [] for key in outputs[0]}
        for metrics in outputs:
            for key in avg_metrics:
                avg_metrics[key].append(metrics[key])

        avg_metrics = {key: torch.as_tensor(
            avg_metrics[key]).mean() for key in avg_metrics}
        for key in avg_metrics:
            self.log(key, avg_metrics[key], prog_bar=False,
                     on_step=False, on_epoch=True)
