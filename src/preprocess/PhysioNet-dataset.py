import sys
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from Preprocess import Preprocess

"""
Dataset Snapshot:

training-a
training-b
training-c
training-d
training-e
training-f

"""
class PhysioNet(Preprocess):
    def __init__(self):
        super().__init__()
        self.dataset_dir = ["./data/PhysioNet-2016/training-a/", "./data/PhysioNet-2016/training-b/",
                            "./data/PhysioNet-2016/training-c/", "./data/PhysioNet-2016/training-d/",
                            "./data/PhysioNet-2016/training-e/", "./data/PhysioNet-2016/training-f/"]

        # -1: meaning classes 2 or 3
        self.lbls = {"normal": 0, "abnormal": -1}
        self.data = []
        self.data_lbls = []
       
    def processData(self, location):

        for dir in tqdm(self.dataset_dir):
            references = np.genfromtxt(f"{dir}REFERENCE.csv", delimiter=',', dtype=str)
            for record in references:
                data_lbl = self.lbls["abnormal"] if record[1]=="1" else self.lbls["normal"]

                metadata = np.genfromtxt(f"{dir}{record[0]}.hea", delimiter="\n", dtype=str)

                raw_signal = self.getAudioSignal(f"{dir}{record[0]}.wav")
                denosed = self.denoise(raw_signal)
                energy = self.ShannonEnergy(denosed)
                peaks = self.findPeaks(energy)
                segmented = self.peakSegmentation(denosed, peaks, margin=200)
                
                for segment in segmented:
                    if len(segment) != 400:
                        segment = np.append(segment, np.zeros(400-len(segment)))
                    self.data.append(segment)
                    self.data_lbls.append(data_lbl)
        
        self.data = torch.Tensor(self.data)
        self.data_lbls = torch.Tensor(self.data_lbls)

        torch.save({'data': self.data, 'labels': self.data_lbls}, location)
        
dataset = PhysioNet()
dataset.processData("./data/preprocessed/PhysioNet.pt")