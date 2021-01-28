import sys
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.metrics.functional import accuracy, precision, recall, f1_score, fbeta_score, stat_scores_multiple_classes
from sklearn import preprocessing, metrics, model_selection
import numpy as np
from torchsampler import ImbalancedDatasetSampler

"""
TODO
    Specificity
    Sensitivity
    PPV
"""


class PrebuiltLightningModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # Metrics
        self.seed = np.random.randint(220)
        pl.seed_everything(seed=self.seed)
        self.criterion = nn.CrossEntropyLoss()

        # Run Name
        self.set_model_name()

    def set_model_name(self):
        delimter = "\\"
        if "/" in sys.argv[0]:
            delimter = "/"
        self.file_name = sys.argv[0].split(delimter)[-1].replace(".py", "")
        timestamp = datetime.now().strftime("%Y%m%d%H%M")
        self.model_name = f"{self.file_name}-{timestamp}"

    def set_model_tags(self, input_size):
        self.model_tags = [str(self.criterion), "SGD", str(input_size)]

    def configure_optimizers(self):
        optimzer = torch.optim.SGD(self.parameters(), lr=1e-5, momentum=1)
        return optimzer

    def training_step(self, batch, batch_idx):
        data, targets = batch
        outputs = self.forward(data)
        loss = self.criterion(outputs, targets)

        # Logs metrics
        metrics = self.metrics_step(outputs, targets)
        metrics["loss"] = loss
        for key in metrics:
            self.log(
                f"train-{key}", metrics[key], prog_bar=False, on_step=True, on_epoch=False)
        return metrics

    def validation_step(self, batch, batch_idx):
        data, targets = batch
        outputs = self.forward(data)
        loss = self.criterion(outputs, targets)

        # Logs metrics
        metrics = self.metrics_step(outputs, targets, prefix="validation-")
        metrics["validation-loss"] = loss
        return metrics

    def validation_epoch_end(self, outputs):
        avg_metrics = {key: [] for key in outputs[0]}
        for metrics in outputs:
            for key in avg_metrics:
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
        metrics = self.metrics_step(outputs, targets, prefix="test-")
        metrics["test-loss"] = loss
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

    def metrics_step(self, outputs, targets, prefix=""):
        pred = torch.argmax(outputs, dim=1)

        accuracy_score = accuracy(pred, targets)
        recall_score = recall(pred, targets, num_classes=3)
        precision_score = precision(pred, targets, num_classes=3)
        f1 = f1_score(pred, targets, num_classes=3)

        learning_rate = self.optimizers().param_groups[0]['lr']

        #tps, fps, tns, fns, sups = stat_scores_multiple_classes(pred, targets)

        return {f'{prefix}accuracy': accuracy_score,
                f'{prefix}recall': recall_score,
                f'{prefix}precision': precision_score,
                f'{prefix}f1': f1,
                f'{prefix}lr': learning_rate}

    def multiclass_roc_auc_score(self, y_test, y_pred, average="macro"):
        y_test, y_pred = y_test.cpu(), y_pred.cpu()
        lb = preprocessing.LabelBinarizer()
        lb.fit(y_test)
        y_test = lb.transform(y_test)
        y_pred = lb.transform(y_pred)
        return metrics.roc_auc_score(y_test, y_pred, average=average)

    def datasets(self, dataset_path, train_split_ratio):
        data, labels = [], []
        for dir in dataset_path:
            dataset = torch.load(dir)
            data.append(dataset["data"])
            labels.append(dataset["labels"])

        data = torch.cat(data, dim=0)
        labels = torch.cat(labels, dim=0)

        train_n_validation_data, test_data, train_n_validation_labels, test_labels = model_selection.train_test_split(
            data, labels, train_size=0.8, test_size=0.2, shuffle=False)
        train_data, validation_data, train_labels, validation_labels = model_selection.train_test_split(
            train_n_validation_data, train_n_validation_labels, train_size=0.9, test_size=0.1)

        train_dataset = TensorDataset(train_data, train_labels.long())
        validation_dataset = TensorDataset(
            validation_data, validation_labels.long())
        test_dataset = TensorDataset(test_data, test_labels)

        return train_dataset, validation_dataset, test_dataset

    def dataloaders(self, train_dataset, validation_dataset, test_dataset, **kwargs):
        train_dataloader = DataLoader(
            test_dataset, sampler=ImbalancedDatasetSampler(test_dataset), **kwargs)
        validation_dataloader = DataLoader(
            validation_dataset, sampler=ImbalancedDatasetSampler(validation_dataset), **kwargs)
        test_dataloader = DataLoader(train_dataset, **kwargs)
        return train_dataloader, validation_dataloader, test_dataloader

    def callback_get_label(dataset, idx):
        # callback function used in imbalanced dataset loader.
        input, target = dataset[idx]
        return target.nonzero().item()
