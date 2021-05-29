import time
import pytorch_lightning as pl
import datamodules
from tqdm import tqdm
import models
import wandb
import torch
from pytorch_lightning.loggers import WandbLogger

datamodule = datamodules.PASCAL.PASCAL(
    "K:\OneDrive - Cumberland Valley School District\Education\Activates\Science Fair\PCG-Science-Fair\PCG-arrhythmia-detection\data\PASCAL", augmentation_factor=5)
"""
datamodule = datamodules.physionet.physionet(
    "K:\OneDrive - Cumberland Valley School District\Education\Activates\Science Fair\PCG-Science-Fair\PCG-arrhythmia-detection\data\PhysioNet-2016")
datamodule = datamodules.bare_loader.Loader(
    "K:\OneDrive - Cumberland Valley School District\Education\Activates\Science Fair\PCG-Science-Fair\PCG-arrhythmia-detection\data\preprocessed\PhysioNet.pt")
"""

# add 3 classes to phsiuonet for nosie

#model = models.AEGAN.AEGAN.AEGAN("SimpleAEGAN", 4)
#model = models.CNN.CNN.CNN_A(2)
# model = models.MSResNet.MSResNet.MSResNet(1, num_classes=4)
model = models.MSResNet.model.model(4, 2500)

datamodule.prepare_data()
datamodule.setup()

train, val, test = datamodule.train_dataloader(), datamodule.val_dataloader(), datamodule.test_dataloader()

wandb_path = r"K:\OneDrive - Cumberland Valley School District\Education\Activates\Science Fair\PCG-Science-Fair\heart-sound-arrhythmia-classification\src\wandb"
wandb_logger = WandbLogger(name=model.model_name, tags=model.model_tags, id=model.model_name, save_dir=wandb_path,
                           project="pcg-arrhythmia-detection", log_model=True)

val_loss_cp = pl.callbacks.ModelCheckpoint(monitor='validation-loss')

trainer = pl.Trainer(max_epochs=10, gpus=1, logger=wandb_logger, fast_dev_run=False,
                     auto_lr_find=False, auto_scale_batch_size=True, log_every_n_steps=1,
                     checkpoint_callback=val_loss_cp)
loader = val
trainer.fit(model, train, val)

model = model.load_from_checkpoint(
    val_loss_cp.best_model_path, num_classes=4, sample_length=2500)

results = trainer.test(model, test)
