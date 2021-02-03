import sys
import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn import preprocessing, metrics, model_selection
from tqdm import tqdm
from scipy.io import wavfile
from scipy import signal
import pywt
import matplotlib.pyplot as plt
from torchsampler import ImbalancedDatasetSampler

# Extends skitlearn class?


class Preprocessor():
    def __init__(self):
        pass

    def getAudioSignal(self, file, targetSamplingRate=500):
        sampleRate, data = wavfile.read(file)

        if sampleRate != targetSamplingRate:
            secs = len(data)/sampleRate
            num_samples = int(secs*targetSamplingRate)
            data = signal.resample(data, num_samples)

        return data

    def getFiles(self, dir, fileExtention="wav"):
        return [fn for fn in os.listdir(dir) if fileExtention in fn]

    def timeSegmentation(self, data, length, sampleRate=500, includeLast=False):
        length_samples = length*sampleRate
        segmented_data = []

        if includeLast:
            data_length = len(data)
        else:
            data_length = len(data)-length_samples

        for i in range(0, data_length, length_samples):
            segmented_data.append(data[i:i+length_samples])
        return segmented_data

    def standardization(self, data):
        return (data - torch.mean(data))/torch.std(data)

    def waveletDenoise(self, s, threshold=5, type='db10', level=4):
        coeffs = pywt.wavedec(s, type, level=level)

        # Applying threshold
        for x in range(len(coeffs)):
            coeffs[x] = pywt.threshold(coeffs[x], threshold, 'soft')

        # Reconstruing denoise signal (IDWT)
        reconstruction = pywt.waverec(coeffs, type)
        return reconstruction

    def savgolDenoise(self, data, window=10, order=None):
        return torch.from_numpy(signal.savogal_filter(data, window, order))

    def combineDatasets(self, dataset_path):
        data, labels = [], []
        for dir in dataset_path:
            dataset = torch.load(dir)
            data.append(dataset["data"])
            labels.append(dataset["labels"])

        data = torch.cat(data, dim=0)
        labels = torch.cat(labels, dim=0)
        return data, labels

    def toTensorDatasets(self, data, labels, split_ratio, **kwargs):
        data_splits = []
        labels_splits = []

        temp_data, temp_labels = data, labels

        for i in range(len(split_ratio)-1):
            splits = [split_ratio[i], sum(split_ratio[i+1:])]
            splits = [1/(sum(splits)/splits[0]), 1/(sum(splits)/splits[1])]

            x_split_1, x_split_2, y_split_1, y_split_2 = model_selection.train_test_split(
                temp_data, temp_labels, train_size=splits[0], test_size=split_ratio[1], shuffle=False)

            data_splits.append(x_split_1)
            labels_splits.append(y_split_1)

            if i == len(split_ratio)-2:
                data_splits.append(x_split_2)
                labels_splits.append(y_split_2)

            temp_data, temp_labels = x_split_2, y_split_2

        tensorDatasets = []

        for x, y in zip(data_splits, labels_splits):
            dataset = TensorDataset(x, y.long())
            tensorDatasets.append(dataset)

        return tensorDatasets

    def dataloaders(self, datasets, **kwargs):
        dataloaders = []
        for dataset in datasets:
            dataloaders.append(DataLoader(
                dataset, sampler=ImbalancedDatasetSampler(dataset), batch_size=kwargs['batch_size']))
        return dataloaders
