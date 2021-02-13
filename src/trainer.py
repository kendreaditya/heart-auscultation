import time
import nvidia_smi
from psutil import virtual_memory
from memory_profiler import profile
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

        rnd_x, rnd_y = [], []
        for _ in range(int(len(test_dataloader/2)):
                       rnd_x.append(torch.rand(
                           1, 2500, device="cuda:0").long())
                       rnd_y.append(torch.tensor(2, device="cuda:0").long()))

        rnd_dataset = torch.utils.data.TensorDataset(rnd_x, rnd_y)

        test_dataloader = torch.utils.data.ConcatDataset(
            datasets[-1], rnd_dataset)

        self.results = trainer.test(model, test_dataloader)
        self.model = model

    def getComplexity(self):
        complexity = []

        for batch_size in range(1, 129):
            try:
                x = torch.rand(batch_size, 1, 2500, device='cuda:0')
                space = [self.getGPU_memory(), self.getRAM_memory()]
                start = time.time()
                self.model(x)
                delta_time = time.time()-start
                space = [space[0]-self.getGPU_memory(), space[1] -
                         self.getRAM_memory()]

                complexity.append([batch_size, delta_time, space])
            except Exception as e:
                print(e)
        return complexity

    def getGPU_memory(self):
        nvidia_smi.nvmlInit()
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        mem = info.used
        nvidia_smi.nvmlShutdown()
        return mem

    def getRAM_memory(self):
        return virtual_memory().used


for i in range(100):
    start = time.time()
    trainerSetup = TrainerSetup()
    train_time = time.time()-start
    acc = float(trainerSetup.results[0]["test-accuracy"])
    loss = float(trainerSetup.results[0]["test-loss"])
    stats = trainerSetup.results[0]["test-stats"]
    complexity = trainerSetup.getComplexity()
    f = open(
        "/content/drive/MyDrive/datasets/PCG_datasets/results/GAN-3v2.log", "a")
    f.write(f"{acc}|{loss}|{stats}|{train_time}|{complexity}\n")
    f.close()
