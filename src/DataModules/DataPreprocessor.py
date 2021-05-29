import sys
import os
import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import TensorDataset, DataLoader
from sklearn import preprocessing, metrics, model_selection
from tqdm import tqdm
from scipy.io import wavfile
from scipy import signal
import pywt
import matplotlib.pyplot as plt
from torchsampler import ImbalancedDatasetSampler


class PreprocessorDataModule(pl.LightningDataModule):
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

    def crossValidationSplit(self, dataset, split_ratio):
        split_ratio = [round(ratio*len(dataset)) for ratio in split_ratio]

        if sum(split_ratio) != len(dataset):
            split_ratio[0] += sum(split_ratio)-len(dataset)

        return torch.utils.data.random_split(dataset, split_ratio)

    def toTensorDatasets(self, x, y):
        tensorDatasets = []

        dataset = TensorDataset(x, y.long())
        tensorDatasets.append(dataset)

        return tensorDatasets
