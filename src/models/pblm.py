import sys
from datetime import datetime
import torch
import torch.nn as nn
import torch.utils.data as data
import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.metrics.functional import accuracy, precision, recall, f1_score, fbeta_score, stat_scores_multiple_classes
from sklearn import preprocessing, metrics
import numpy as np

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
        self.file_name = sys.argv[0].split("/")[-1].replace(".py", "")
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
            self.log(f"train-{key}", metrics[key], prog_bar=False, on_step=True, on_epoch=False)
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
        avg_metrics = {key:[] for key in outputs[0]}
        for metrics in outputs:
            for key in avg_metrics:
                avg_metrics[key].append(metrics[key])
        avg_metrics = {key:torch.as_tensor(avg_metrics[key]).mean() for key in avg_metrics}
        for key in avg_metrics:
            self.log(key, avg_metrics[key], prog_bar=True, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        data, targets = batch
        outputs = self.forward(data)
        loss = self.criterion(outputs, targets)

        # Logs metrics
        metrics = self.metrics_step(outputs, targets, prefix="test-")
        metrics["test-loss"] = loss
        return metrics

    def test_epoch_end(self, outputs):
        avg_metrics = {key:[] for key in outputs[0]}
        for metrics in outputs:
            for key in avg_metrics:
                avg_metrics[key].append(metrics[key])

        avg_metrics = {key:torch.as_tensor(avg_metrics[key]).mean() for key in avg_metrics}
        for key in avg_metrics:
            self.log(key, avg_metrics[key], prog_bar=False, on_step=False, on_epoch=True)

    def metrics_step(self, outputs, targets, prefix=""):
        pred = torch.argmax(outputs, dim=1)

        accuracy_score = accuracy(pred, targets)
        recall_score = recall(pred, targets, num_classes=3)
        precision_score = precision(pred, targets, num_classes=3)
        f1 = f1_score(pred, targets, num_classes=3)
        roc_auc_score = self.multiclass_roc_auc_score(targets, pred)

        learning_rate = self.optimizers().param_groups[0]['lr']

        tps, fps, tns, fns, sups = stat_scores_multiple_classes(pred, targets)

        return {f'{prefix}accuracy': accuracy_score,
                f'{prefix}recall': recall_score,
                f'{prefix}precision': precision_score,
                f'{prefix}f1': f1,
                f'{prefix}ROC-AUC': roc_auc_score,
                f'{prefix}TP': tps,
                f'{prefix}TN': tns,
                f'{prefix}FP': fps,
                f'{prefix}FN': fns,
                f'{prefix}sups': sups,
                f'{prefix}lr': learning_rate}

    def multiclass_roc_auc_score(self, y_test, y_pred, average="macro"):
        y_test, y_pred = y_test.cpu(), y_pred.cpu()
        lb = preprocessing.LabelBinarizer()
        lb.fit(y_test)
        y_test = lb.transform(y_test)
        y_pred = lb.transform(y_pred)
        return metrics.roc_auc_score(y_test, y_pred, average=average)

    def sparce_split(self, X, labels, train_split_ratio, num_classes=3):
        train_dataset = {"data":[], "labels": []}
        train_dist = {i:0 for i in range(num_classes)}
        validation_dataset = {"data":[], "labels": []}
        validation_dist = {i:0 for i in range(num_classes)}

        trans_x_y = list(zip(X, labels))
        np.random.shuffle(trans_x_y)
        
        for x,y in trans_x_y:
            x = np.array(x)
            if validation_dist[int(y)] < (train_split_ratio[1]//num_classes):
                validation_dataset["data"].append(x)
                validation_dataset["labels"].append(y)
                validation_dist[int(y)] += 1
            elif train_dist[int(y)] < (train_split_ratio[0]//num_classes):
                train_dataset["data"].append(x)
                train_dataset["labels"].append(y)
                train_dist[int(y)] += 1

        train_dataset = data.TensorDataset(torch.Tensor(train_dataset["data"]), torch.Tensor(train_dataset["labels"]).long())
        validation_dataset = data.TensorDataset(torch.Tensor(validation_dataset["data"]), torch.Tensor(validation_dataset["labels"]).long())
        return train_dataset, validation_dataset

    def datasets(self, dataset_path, split, band_type, train_split_ratio):
        dataset = torch.load(dataset_path)[split]
        train_dataset, validation_dataset = self.sparce_split(dataset["train"][band_type], dataset["train"]["labels"], train_split_ratio)
        test_dataset = data.TensorDataset(dataset["test"][band_type], dataset["test"]["labels"].long())
        return train_dataset, validation_dataset, test_dataset

    def dataloaders(self, train_dataset, validation_dataset, test_dataset, **kwargs):
        train_dataloader = data.DataLoader(test_dataset, **kwargs)
        validation_dataloader = data.DataLoader(validation_dataset, **kwargs)
        test_dataloader = data.DataLoader(train_dataset, **kwargs)
        return train_dataloader, validation_dataloader, test_dataloader