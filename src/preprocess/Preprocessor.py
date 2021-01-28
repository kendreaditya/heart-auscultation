import sys
import os
import numpy as np
import torch
from tqdm import tqdm
from scipy.io import wavfile
from scipy import signal
import pywt
import matplotlib.pyplot as plt


# Extends skitlearn class?
class Preprocessor():
    def __init__(self):
        pass

    def getAudioSignal(self, file, targetSamplingRate=500):
        sampleRate, data = wavfile.read(file)

        if sampleRate!=targetSamplingRate:
            secs = len(data)/sampleRate
            num_samples = int(secs*targetSamplingRate)
            data = signal.resample(data, num_samples)
        
        return data
    
    def getFiles(self, dir, fileExtention="wav"):
        return [fn for fn in os.listdir(dir) if fileExtention in fn]
    
    def denoise(self, s, threshold=5, type='db10', level=4):
        coeffs = pywt.wavedec(s, type, level=level)

        # Applying threshold
        for x in range(len(coeffs)):
            coeffs[x] = pywt.threshold(coeffs[x], threshold, 'soft')

        # Reconstruing denoise signal (IDWT)
        reconstruction = pywt.waverec(coeffs, type)
        return reconstruction

    def ShannonEnergy(self, data):
        Es = data.copy()

        for i in range(len(Es)):
            Es[i] = (Es[i]**2) * np.log(Es[i]**2)

        return Es

    def findPeaks(self, data):
        peaks, meta = signal.find_peaks(data, height=np.nanmean(data)+np.nanstd(data))

        delta = []
        for i in range(0,len(peaks)-1):
            delta.append(np.abs(peaks[i]-peaks[i+1]))
        mean = np.mean(delta)

        peaks, meta = signal.find_peaks(data, height=np.nanmean(data)+np.nanstd(data), distance=mean)

        return peaks

    def peakSegmentation(self, data, peaks, margin=200):
        segmentation = []

        for i in range(0,len(peaks)):
            local_mean = peaks[i]

            if local_mean > margin:
                idxR, idxL = int(local_mean-margin), int(local_mean+margin)
            else:
                idxR, idxL = 0, 2*margin
            segmentation.append(data[idxR:idxL])
        return segmentation
    
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