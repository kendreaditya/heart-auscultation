import csv
import pytorch_lightning as pl
import datamodules
import models
import torch
import wandb
from tqdm import tqdm
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.metrics.functional import accuracy, precision, recall, stat_scores_multiple_classes

#model = models.AEGAN.AEGAN.AEGAN("SimpleAEGAN", 4)
#model = models.CNN.CNN.CNN_A(2)
#model = models.MSResNet.MSResNet.MSResNet(1, num_classes=2)
#model = models.MSResNet.model.model(num_classes=4)
# model = models.MSResNet.MSResNet.MSResNet(1, num_classes=4)
model = models.MSResNet.model.model(4, 2500)

wandb_path = r"K:\OneDrive - Cumberland Valley School District\Education\Activates\Science Fair\PCG-Science-Fair\heart-sound-arrhythmia-classification\src\wandb"
# model_path = "K:\OneDrive - Cumberland Valley School District\Education\Activates\Science Fair\PCG-Science-Fair\heart-sound-arrhythmia-classification\src\wandb\pcg-arrhythmia-detection\CNN_A-202104180029\checkpoints\epoch=79-step=22799.ckpt"
#model_path = "K:\OneDrive - Cumberland Valley School District\Education\Activates\Science Fair\PCG-Science-Fair\heart-sound-arrhythmia-classification\src\wandb\pcg-arrhythmia-detection\MSResNet-202104181406\checkpoints\epoch=27-step=5599.ckpt"
model_path = "K:\OneDrive - Cumberland Valley School District\Education\Activates\Science Fair\PCG-Science-Fair\heart-sound-arrhythmia-classification\src\wandb\pcg-arrhythmia-detection\model-202105041122\checkpoints\epoch=7-step=399.ckpt"
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint["state_dict"])

model = torch.jit.trace(model, torch.rand(1, 1, 2500))
torch.jit.save(model, "K:\PCG-app\heart_sound_backend\model-jit.pt")

# datamodule.prepare_data()
# datamodule.setup()
# test_loader = datamodule.test_dataloader()

# fn = model_path + ".raw_outputs.csv"

# f = open(fn, "a")
# for x, y in tqdm(test_loader):
#     pred = model(x)
#     for x1, y1 in zip(pred, y):
#         f.write(f"{x1.detach().tolist()}|{y1}\n")
# f.close()
