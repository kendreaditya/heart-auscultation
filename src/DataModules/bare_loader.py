import os
import torch
import pytorch_lightning as pl
from tqdm import tqdm
from torch.utils.data import random_split, DataLoader, TensorDataset
from torchsampler import ImbalancedDatasetSampler
from scipy.io import wavfile
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift
from scipy import signal
import numpy as np


import matplotlib.pyplot as plt


class Loader(pl.LightningDataModule):
    def __init__(self, data_path, sample_length=2500, dataset_split=[0.7, .1, .2], file_extention="wav", freq=500, augmentation_factor=1):
        super(Loader, self).__init__()
        self.augmentation_factor = augmentation_factor
        self.freq = freq
        self.sample_length = sample_length
        self.dataset_split = dataset_split
        self.data_path = data_path
        self.data = torch.load(self.data_path)
        self.dataset_lbls = {key: i for i, key in enumerate(sorted(list(set(self.data["labels"]))))}

        self.x = []
        self.y = []

        self.augment = Compose([
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
            TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
            PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
            Shift(min_fraction=-0.5, max_fraction=0.5, p=0.5),
        ])

    def prepare_data(self):
        print("Preparing data...")
        print("Done Preparing data!")

    def setup(self, stage=None):
        print("Starting Preprocessing...")
        for audio, lbl in tqdm(zip(self.data["data"], self.data["labels"])):
            self.x.append(audio.numpy())
            self.y.append(lbl)
        # Sparce Split Dataset
        self.traning_set, self.validation_set, self.testing_set = self.split_dataset(self.x, self.y, self.dataset_split)
        print("Done Preprocessing!")

    def resample(self, data, freq, target_freq):
        if freq == target_freq:
            return freq, data

        secs = len(data) / freq
        num_samples = int(secs * target_freq)
        data = signal.resample(data, num_samples)
        return freq, data

    def split_data(self, data, sample_length):
        # implament spltting basedon s1 and s2
        data_splits = []

        for i in range(0, len(data), sample_length):
            if len(data[i:i + sample_length]) == sample_length:
                data_splits.append(data[i:i + sample_length])
        return data_splits

    def split_dataset(self, x, y, ratio, stratified=True):
        """
        testing set -> get full length via ratio*len(y)
        according get the type number of labels types
        1/2 = number of labels for type for testing set
        """
        dataset = list(map(list, zip(x, y)))

        if stratified == False:
            return None

        label_maped = {}

        for datum in dataset:
            lbl = str(datum[1])
            if lbl not in label_maped.keys():
                label_maped[lbl] = list()
            label_maped[lbl].append(datum)

        idx_ratio = ratio.copy()
        idx_ratio.insert(0, 0)
        idx_ratio = np.array(idx_ratio)

        split_idx = {lbl: np.floor(idx_ratio * len(label_maped[lbl])).astype(int) for lbl in label_maped.keys()}

        split_x, split_y = [[] for _ in ratio], [[] for _ in ratio]

        for set_n in range(len(ratio)):
            for lbl in label_maped.keys():
                start, end = sum(split_idx[lbl][:set_n + 1]), sum(split_idx[lbl][:set_n + 2])
                split_lbl = label_maped[lbl][start:end]

                [split_x[set_n].append(i[0]) for i in split_lbl]
                [split_y[set_n].append(i[1]) for i in split_lbl]

        return list(map(list, zip(split_x, split_y)))

    def train_dataloader(self):
        print("Started data agumentation...")
        for _ in range(self.augmentation_factor):
            for sample, lbl in tqdm(list(map(list, zip(self.traning_set[0], self.traning_set[1])))):
                augmented_sample = self.augment(samples=sample, sample_rate=self.freq)
                self.traning_set[0].append(augmented_sample.tolist())
                self.traning_set[1].append(lbl)
        print("Finshed data agumentation!")
        self.traning_set = [torch.tensor(self.traning_set[0]).float(
        ).reshape(-1, 1, self.sample_length), torch.tensor(self.traning_set[1]).long()]

        dataset = TensorDataset(self.traning_set[0], self.traning_set[1])
        data_loader = DataLoader(dataset, batch_size=128, sampler=ImbalancedDatasetSampler(
            dataset, callback_get_label=lambda dataset, idx: int(dataset[idx][1])))
        return data_loader

    def val_dataloader(self):
        self.validation_set = [torch.tensor(self.validation_set[0]).float().reshape(-1, 1, self.sample_length),
                               torch.tensor(self.validation_set[1]).long()]

        dataset = TensorDataset(self.validation_set[0], self.validation_set[1])
        data_loader = DataLoader(dataset, batch_size=128)
        return data_loader

    def test_dataloader(self):
        self.testing_set = [torch.tensor(self.testing_set[0]).float().reshape(-1, 1, self.sample_length),
                            torch.tensor(self.testing_set[1]).long()]

        dataset = TensorDataset(self.testing_set[0], self.testing_set[1])
        data_loader = DataLoader(dataset, batch_size=128)
        return data_loader


if __name__ == "__main__":
    datamodule = Loader(
        "K:\OneDrive - Cumberland Valley School District\Education\Activates\Science Fair\PCG-Science-Fair\PCG-arrhythmia-detection\data\preprocessed\PASCAL.pt")
    datamodule.prepare_data()
    datamodule.setup()
    loader = datamodule.train_dataloader()
