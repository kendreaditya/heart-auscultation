# MITHSDB dataset from PhysioNet 2016
import os
import torch
import pytorch_lightning as pl
from tqdm import tqdm
from torch.utils.data import random_split, DataLoader, TensorDataset
from torchsampler import ImbalancedDatasetSampler
from scipy.io import wavfile
from scipy import signal
import numpy as np
import wfdb


class physionet(pl.LightningDataModule):
    def __init__(self, data_path, sample_shape=(2500), dataset_split=[0.7, .1, .2], freq=500):
        super(physionet, self).__init__()
        self.freq = freq
        self.sample_shape = sample_shape
        self.dataset_split = dataset_split

        self.dataset_dir = [f"{data_path}/training-a/"]

        self.dataset_lbls = {"normal": 0, "abnormal": 1}

        self.data = []
        self.x = []
        self.y = []

    def prepare_data(self):
        print("Preparing data...")
        for path in self.dataset_dir:
            records = np.genfromtxt(f"{path}REFERENCE.csv", delimiter=',', dtype=str)
            for record in records:
                fn, lbl = record
                lbl = 0 if int(lbl) < 0 else int(lbl)

                rdrecord = wfdb.rdrecord(f'{path}{fn}')
                if rdrecord.p_signal.shape[-1] == 1:
                    continue
                PCG, ECG = np.transpose(rdrecord.p_signal)

                info = wfdb.rdsamp(f'{path}{fn}')
                freq = info[1]["fs"]

                self.data.append([freq, PCG, ECG])
        print("Done Preparing data!")

    def setup(self, stage=None):
        print("Starting Preprocessing...")
        for freq, PCG, ECG in tqdm(self.data):

            # Resample to freq
            _, PCG = self.resample(PCG, freq, self.freq)
            _, ECG = self.resample(ECG, freq, self.freq)

            # Standardize
            PCG = (np.array(PCG) - np.mean(PCG)) / np.std(PCG)
            ECG = (np.array(ECG) - np.mean(ECG)) / np.std(ECG)

            # Splitting
            PCG_splits = self.split_data(PCG, self.sample_shape)
            ECG_splits = self.split_data(ECG, self.sample_shape)

            for PCG, ECG in zip(PCG_splits, ECG_splits):
                self.x.append([PCG])
                self.y.append([ECG])

        # Sparce Split Dataset
        self.traning_set, self.validation_set, self.testing_set = self.split_dataset(self.x, self.y, self.dataset_split)
        self.traning_set = [torch.tensor(self.traning_set[0]).float(), torch.tensor(self.traning_set[1]).long()]
        self.validation_set = [torch.tensor(self.validation_set[0]).float(),
                               torch.tensor(self.validation_set[1]).long()]
        self.testing_set = [torch.tensor(self.testing_set[0]).float(), torch.tensor(self.testing_set[1]).long()]
        print("Done Preprocessing!")

    def resample(self, data, freq, target_freq):
        if freq == target_freq:
            return freq, data

        secs = len(data) / freq
        num_samples = int(secs * target_freq)
        data = signal.resample(data, num_samples)
        return freq, data

    def split_data(self, data, sample_shape):
        # implament spltting basedon s1 and s2
        data_splits = []

        for i in range(0, len(data), sample_shape):
            if len(data[i:i + sample_shape]) == sample_shape:
                data_splits.append(data[i:i + sample_shape])
        return data_splits

    def split_dataset(self, x, y, ratio, stratified=True):
        """
        testing set -> get full length via ratio*len(y)
        according get the type number of labels types
        1/2 = number of labels for type for testing set
        """
        dataset = list(map(list, zip(x, y)))
        split_x, split_y = [[] for _ in ratio], [[] for _ in ratio]

        if stratified == False:

            np.random.shuffle(dataset)
            split_idx = ratio.copy()
            split_idx.insert(0, 0)
            split_idx = np.array(split_idx)
            split_idx = np.floor(len(dataset) * split_idx).astype(int)

            for set_n in range(len(ratio)):
                start, end = sum(split_idx[:set_n + 1]), sum(split_idx[:set_n + 2])
                split_set = dataset[start:end]

                [split_x[set_n].append(i[0]) for i in split_set]
                [split_y[set_n].append(i[1]) for i in split_set]
            return list(map(list, zip(split_x, split_y)))

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

        for set_n in range(len(ratio)):
            for lbl in label_maped.keys():
                start, end = sum(split_idx[lbl][:set_n + 1]), sum(split_idx[lbl][:set_n + 2])
                split_lbl = label_maped[lbl][start:end]

                [split_x[set_n].append(i[0]) for i in split_lbl]
                [split_y[set_n].append(i[1]) for i in split_lbl]

        return list(map(list, zip(split_x, split_y)))

    def train_dataloader(self):
        dataset = TensorDataset(self.traning_set[0], self.traning_set[1])
        data_loader = DataLoader(dataset, batch_size=128)
        return data_loader

    def val_dataloader(self):
        dataset = TensorDataset(self.validation_set[0], self.validation_set[1])
        data_loader = DataLoader(dataset, batch_size=128, shuffle=True)
        return data_loader

    def test_dataloader(self):
        dataset = TensorDataset(self.testing_set[0], self.testing_set[1])
        data_loader = DataLoader(dataset, batch_size=128, shuffle=True)
        return data_loader


if __name__ == "__main__":
    datamodule = physionet(
        "K:\OneDrive - Cumberland Valley School District\Education\Activates\Science Fair\PCG-Science-Fair\PCG-arrhythmia-detection\data\PhysioNet-2016")
    datamodule.setup()
    datamodule.prepare_data()
    datamodule.train_dataloader()
