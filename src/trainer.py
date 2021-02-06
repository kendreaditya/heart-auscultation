import pandas as pd
import torch
import wandb
import models
from preprocess.Preprocessor import Preprocessor
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

import os
os.environ['WANDB_MODE'] = 'dryrun'


class TrainerSetup():
    def __init__(self):

        # Model init
        model = models.GAN.GAN_A(discriminator=models.CNN.CNN_A())
        pp = Preprocessor()

        dataset, labels = pp.combineDatasets(["/content/drive/MyDrive/datasets/PCG_datasets/data/PASCAL.pt",
                                              "/content/drive/MyDrive/datasets/PCG_datasets/data/PhysioNet.pt"])

        datasets = pp.toTensorDatasets(
            dataset, labels, [0.8, .1, .1])

        # del dataset
        # del labels

        train_dataloader, validation_dataloader, test_dataloader = pp.dataloaders(
            datasets, batch_size=32)

        # Logging
        model.model_tags.append("train:"+str(len(datasets[0])))
        model.model_tags.append("validation:"+str(len(datasets[1])))
        model.model_tags.append("test:"+str(len(datasets[2])))
        model.model_tags.append("seed:"+str(model.seed))

        del datasets

        # wandb_logger = WandbLogger(name=model.model_name, save_dir="/content/drive/models/", tags=model.model_tags,
        wandb_logger = WandbLogger(name=model.model_name, tags=model.model_tags, id=model.model_name, save_dir="/content/",
                                   project="pcg-arrhythmia-detection", log_model=True, reinit=True)

        # Checkpoints
        val_loss_cp = pl.callbacks.ModelCheckpoint(monitor='validation-loss')

        trainer = pl.Trainer(max_epochs=100, gpus=1, logger=wandb_logger, fast_dev_run=False,
                             auto_lr_find=False, auto_scale_batch_size=True, log_every_n_steps=1,
                             checkpoint_callback=val_loss_cp)

        # Train Model
        trainer.fit(model, train_dataloader, validation_dataloader)

        # Load best model with lowest validation
        model = model.load_from_checkpoint(
            val_loss_cp.best_model_path, discriminator=models.CNN.CNN_A())

        # Test model on testing set
        self.results = trainer.test(model, test_dataloader)


for i in range(1000):
    trainerSetup = TrainerSetup()
    acc = float(trainerSetup.results[0]["test-accuracy"])
    loss = float(trainerSetup.results[0]["test-loss"])
    stats = trainerSetup.results[0]["test-stats"]
    f = open(
        "/content/drive/MyDrive/datasets/PCG_datasets/results/GAN-CNN-C3.log", "a")
    f.write(f"{acc}|{loss}|{stats}\n")
    f.close()
