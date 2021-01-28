import pblm
import sys
import torch
import torch.nn as nn
import torch.utils.data as data
import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.metrics.functional import accuracy, precision, recall, f1_score, fbeta_score
from sklearn import preprocessing, metrics
import numpy as np
from pdb import set_trace as bp

class CNN(pblm.PrebuiltLightningModule):
    def __init__(self, input_size=(1,400)):
        super().__init__()

        # Model Tags
        self.set_model_tags(input_size)
        self.model_tags.append(self.file_name)

        # Model Layer Declaration
        self.dense1 = nn.Linear(3*500, 1024)
        self.dense2 = nn.Linear(1024, 512)
        self.dense3 = nn.Linear(512, 128)
        self.dense4 = nn.Linear(128, 64)
        self.dense5 = nn.Linear(64, 3)

    def forward(self, x):
        x = self.dense1(x)
        x = nn.functional.relu(x)
        x = self.dense2(x)
        x = nn.functional.relu(x)
        x = self.dense3(x)
        x = nn.functional.relu(x)
        x = self.dense4(x)
        x = self.dense5(x)

        return x

def train():
    # Model init
    model = CNN()
    train_dataset, validation_dataset, test_dataset = model.datasets(["./data/preprocessed/PASCAL.pt", "./data/preprocessed/PhysioNet.pt"], train_split_ratio=.8)

    train_dataloader, validation_dataloader, test_dataloader = model.dataloaders(train_dataset, validation_dataset, test_dataset,
                                                                                 batch_size=32)
    # Logging
    model.model_tags.append("train:"+str(len(train_dataset)))
    model.model_tags.append("validation:"+str(len(validation_dataset)))
    model.model_tags.append("test:"+str(len(test_dataset)))
    model.model_tags.append("seed:"+str(model.seed))

    wandb_logger = WandbLogger(name=model.model_name, tags=model.model_tags, project="pcg-arrhythmia-detection", log_model=True)
    wandb_logger.watch(model, log='gradients', log_freq=100)

    # Checkpoints
    val_loss_cp = pl.callbacks.ModelCheckpoint(monitor='validation-loss')

    trainer = pl.Trainer(max_epochs=1000, gpus=1, logger=wandb_logger, precision=16, fast_dev_run=False,
                         auto_lr_find=True, auto_scale_batch_size=True, log_every_n_steps=1,
                        checkpoint_callback=val_loss_cp)
    trainer.fit(model, train_dataloader, validation_dataloader)
    print("Done training.")

    print(f"Testing model with best validation loss\t{val_loss_cp.best_model_score}.")
    model = model.load_from_checkpoint(val_loss_cp.best_model_path)
    results = trainer.test(model, test_dataloader)

    print("Done testing.")

train()