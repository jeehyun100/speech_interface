#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import librosa
import numpy as np
from torch.utils.data import Dataset
import sys
import scipy
import random

class SpeakerDataset(Dataset):
    def __init__(self, data_list, data_path='', nfft=512, win_len_time=0.02, hop_len_time=0.01, fs = 16000,
                 feature_type='mel', n_coeff=64, fr_len=150, random_window=False):
        # Read txt file.
        with open(data_list,'r') as f:
            self.fileset = f.read().split('\n')[:-1] 
        self.data_path = data_path
        self.feature_type = feature_type
        self.win_sample = int(win_len_time*fs)
        self.hop_sample = int(hop_len_time*fs)
        self.nfft = nfft
        self.fs = fs
        self.n_coeff = n_coeff
        self.eps = np.array(sys.float_info.epsilon)
        self.max_fr = fr_len
        self.total_n = len(self.fileset)
        self.currenct_n = 1
        self.random_window = random_window

    def __len__(self):
        return len(self.fileset)
    
    def __getitem__(self, idx):
        token = self.fileset[idx].split(' ')
        path = token[1]
        label = int(token[0])
        sig, fs = librosa.load(self.data_path+path, sr=self.fs)
        if self.random_window:
            if(self.currenct_n > self.total_n ):
                win_length_r = int(random.uniform(100, self.nfft))
                self.win_sample = win_length_r
                print("Current_n/Total_n {0} / {1}".format(self.currenct_n, self.total_n))
                self.currenct_n = 1
                print("window length {0}".format(win_length_r))

        #print("global window length {0}".format(self.win_sample))
        feature = self.get_feature(sig)

        self.currenct_n += 1
        return feature, label
    
    # Combine features depending on the context window setup.
    def context_window(self, feature, left, right):
        context_feature = []
        for i in range(left, len(feature)-right):
            feat = np.concatenate((feature[i-left:i+right]),axis=-1)
            context_feature.append(feat)
        context_feature = np.vstack(context_feature)
        return context_feature            
            
    # Compute input features and concatenate depending on the type of context window.
    def get_feature(self,sig):
        stft = librosa.core.stft(sig, n_fft=self.nfft, hop_length=self.hop_sample,
                                 win_length=self.win_sample)
        feature = abs(stft).transpose()
        if self.feature_type == 'mel':
            mel_fb = librosa.filters.mel(self.fs, n_fft=self.nfft, n_mels=self.n_coeff)
            power = feature**2
            feature = np.matmul(power, mel_fb.transpose())
            feature = 10*np.log10(feature+self.eps)
            delta = librosa.feature.delta(feature)
            delta2 = librosa.feature.delta(feature, order=2)
            feature = np.concatenate((feature, delta, delta2), axis=-1)
        elif self.feature_type == 'mfcc':
            mel_fb = librosa.filters.mel(self.fs, n_fft=self.nfft, n_mels=self.n_coeff)
            power = feature**2
            feature = np.matmul(power, mel_fb.transpose())
            feature = 10*np.log10(feature+self.eps)
            feature = scipy.fftpack.dct(feature, axis=-1, norm='ortho')
            delta = librosa.feature.delta(feature)
            delta2 = librosa.feature.delta(feature, order=2)
            feature = np.concatenate((feature, delta, delta2), axis=-1)
            
        feature = self.context_window(feature=feature, left=5, right=5)
        return feature