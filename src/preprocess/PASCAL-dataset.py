import sys
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from Preprocess import Preprocess

"""
Dataset Snapshot:

Dataset A:
    Normal
    Murmur
    Extra Heart Sound
    Artifact

Dataset B:
    Normal
    Murmur
    Extrasystole
"""
class PASCAL(Preprocess):
    def __init__(self):
        super().__init__()
        self.dataset_dir = {"normal": ["./data/PASCAL/Atraining_artifact/", "./data/PASCAL/Training B Normal/"],
                        "murmur": ["./data/PASCAL/Atraining_murmur/", "./data/PASCAL/Btraining_murmur/"],
                        "extra-heart-sounds": ["./data/PASCAL/Atraining_extrahls/", "./data/PASCAL/Btraining_extrastole/"],
                        "artifact": ["./data/PASCAL/Atraining_artifact/"]}
        
        self.lbls = {"normal": 0, "murmur": 1, "extra-heart-sounds": 2, "artifact": 3}
        self.data = []
        self.data_lbls = []
       
    def processData(self, location):

        for label in tqdm(self.dataset_dir):
            data_lbl = self.lbls[label]
            for dir in self.dataset_dir[label]:
                files = self.getFiles(dir)
                for file in files:
                    raw_signal = self.getAudioSignal(f"{dir}{file}")
                    denosed = self.denoise(raw_signal)
                    energy = self.ShannonEnergy(denosed)
                    peaks = self.findPeaks(energy)
                    segmented = self.peakSegmentation(denosed, peaks, margin=200)
                    
                    for segment in segmented:
                        self.data.append(segment)
                        self.data_lbls.append(data_lbl)
        
        self.data = torch.Tensor(self.data)
        self.data_lbls = torch.Tensor(self.data_lbls)

        torch.save({'data': self.data, 'labels': self.data_lbls})
        
        
dataset = PASCAL()
dataset.processData("./data/preprocessed/PACAL.pt")