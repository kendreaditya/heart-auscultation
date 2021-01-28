import sys
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from Preprocessor import Preprocessor

"""
Dataset Snapshot:

training-a
training-b
training-c
training-d
training-e
training-f
validation

"""
class PhysioNet(Preprocessor):
    def __init__(self):
        super().__init__()
        self.dataset_dir = ["./data/PhysioNet-2016/training-a/", "./data/PhysioNet-2016/training-b/",
                            "./data/PhysioNet-2016/training-c/", "./data/PhysioNet-2016/training-d/",
                            "./data/PhysioNet-2016/training-e/", "./data/PhysioNet-2016/training-f/"]

        self.lbls = {"normal": 0, "abnormal": 1}
        self.data = []
        self.data_lbls = []
       
    def traverseDataset(self, location):

        for dir in tqdm(self.dataset_dir):
            references = np.genfromtxt(f"{dir}REFERENCE.csv", delimiter=',', dtype=str)
            for record in references:
                data_lbl = self.lbls["abnormal"] if record[1]=="1" else self.lbls["normal"]

                metadata = np.genfromtxt(f"{dir}{record[0]}.hea", delimiter="\n", dtype=str)

                raw_signal = self.getAudioSignal(f"{dir}{record[0]}.wav")
                segmented_signal = self.signalPreprocess(raw_signal, length=5, sampleRate=500, includeLast=False)

                for segment in segmented_signal:
                    self.data.append(segment)
                    self.data_lbls.append(data_lbl)
        
        self.data = torch.Tensor(self.data)
        self.data_lbls = torch.Tensor(self.data_lbls).long()
        print(self.data.shape)
        print(self.data_lbls.shape)


        torch.save({'data': self.data, 'labels': self.data_lbls}, location)

    def signalPreprocess(self, data, **kargs):
        segmented_signal = self.timeSegmentation(data, length=kargs["length"], sampleRate=kargs["sampleRate"], includeLast=kargs["includeLast"])
        return segmented_signal
        
dataset = PhysioNet()
dataset.traverseDataset("./data/preprocessed/PhysioNet.pt")