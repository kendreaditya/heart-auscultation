import sys
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from Preprocessor import Preprocessor

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


class PASCAL(Preprocessor):
    def __init__(self):
        super().__init__()
        self.dataset_dir = {"normal": ["./data/PASCAL/Atraining_artifact/", "./data/PASCAL/Training B Normal/"],
                            "murmur": ["./data/PASCAL/Atraining_murmur/", "./data/PASCAL/Btraining_murmur/"],
                            "extra-heart-sounds": ["./data/PASCAL/Atraining_extrahls/", "./data/PASCAL/Btraining_extrastole/"],
                            "artifact": ["./data/PASCAL/Atraining_artifact/"]}

        self.lbls = {"normal": 0, "murmur": 1,
                     "extra-heart-sounds": 1, "artifact": 2}
        self.data = []
        self.data_lbls = []

    def traverseDataset(self, location):

        for label in tqdm(self.dataset_dir):
            data_lbl = self.lbls[label]
            for dir in self.dataset_dir[label]:
                files = self.getFiles(dir)
                for file in files:
                    raw_signal = self.getAudioSignal(f"{dir}{file}", 500)
                    segmented_signal = self.signalPreprocess(
                        raw_signal, length=5, sampleRate=500, includeLast=False)
                    for segment in segmented_signal:
                        self.data.append(segment)
                        self.data_lbls.append(data_lbl)

        self.data = torch.Tensor(self.data)
        self.data_lbls = torch.Tensor(self.data_lbls).long()
        print(self.data.shape)
        print(self.data_lbls.shape)

        torch.save({'data': self.data, 'labels': self.data_lbls}, location)

    def signalPreprocess(self, data, **kargs):
        segmented_signal = self.timeSegmentation(
            data, length=kargs["length"], sampleRate=kargs["sampleRate"], includeLast=kargs["includeLast"])
        return segmented_signal


dataset = PASCAL()
dataset.traverseDataset("./data/preprocessed/PASCAL.pt")
