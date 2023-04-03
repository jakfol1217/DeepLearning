import pandas as pd
import numpy as np
from sklearn import preprocessing
from torch.utils.data import DataLoader, Dataset
import librosa
import glob
import torch
from pydub import AudioSegment

import random

name_dict = {'yes': 0,
             'no': 1,
             'up': 2,
             'down': 3,
             'left': 4,
             'right': 5,
             'on': 6,
             'off': 7,
             'stop': 8,
             'go': 9,
             'silence':10}
#unknown = 11
def get_audio_datasets(path=''):
    if len(path) > 0 and path[-1] != '/':
        path += '/'
    # Get full dataset
    files = glob.glob(path + '.data-audioset/train/audio/*/*.wav')
    full_data = pd.DataFrame(pd.Series(files).apply(lambda x: str.replace(x, '\\', '/')))
    full_data['label'] = full_data[0].apply(lambda x: x.split('/')[3])
    full_data = full_data.rename({0: 'file'}, axis=1)
    full_data = full_data[full_data['label'] != '_background_noise_']
    # Add noise
    files = glob.glob('.data-audioset/train/audio/_background_noise_/*.wav')
    names = []
    for file in files:
        audio = AudioSegment.from_wav(file)
        for i in range(len(audio) // 1000):
            name = str.split(file, '.wav')[0] + '_' + str(i) + '.wav'
            audio_cut = audio[1000 * i:1000 * (i + 1)]
            audio_cut.export(name, format="wav")
            names.append(name)
    df_silence = pd.DataFrame({'file': names,
                               'label': ['silence' for i in range(len(names))]})

    full_data = pd.concat([full_data, df_silence])
    full_data = full_data.sample(frac=1).reset_index(drop=True)
    full_data['label'] = full_data['label'].map(name_dict).fillna(10)
    # Get test and valid datasets
    test_data = (pd.read_csv(path + '.data-audioset/train/testing_list.txt', header=None)
                 .apply(lambda x: '.data-audioset/train/audio/' + x))
    valid_data = (pd.read_csv(path + '.data-audioset/train/validation_list.txt', header=None)
                  .apply(lambda x: '.data-audioset/train/audio/' + x))
    test_data = pd.concat([test_data, pd.Series(names[12])])
    valid_data = pd.concat([valid_data, pd.Series(names[-12])])
    test_data = full_data[full_data['file'].isin(test_data[0])].reset_index()
    valid_data = full_data[full_data['file'].isin(valid_data[0])].reset_index()
    # Get train dataset
    train_data = full_data[~full_data['file'].isin(pd.concat([test_data['file'], valid_data['file']]))]
    train_dataset = AudioDataset(train_data)
    valid_dataset = AudioDataset(valid_data)
    test_dataset = AudioDataset(test_data)
    return train_dataset, test_dataset, valid_dataset

def load_audio_dataloaders_validation(path='', bs=16):
    dataset_train, dataset_test, dataset_val = get_audio_datasets(path=path)
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=bs)
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=bs)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=bs)
    return dataloader_train, dataloader_test, dataloader_val
class DataPrep:
    # By default, the data have 16000 sampling rate and are monochannel
    @staticmethod
    def load(path, sr, mono=True):
        data, sr = librosa.load(path, sr=sr, mono=mono)
        return data, sr

    @staticmethod
    def resize(data, max_size):
        data_len = data.shape[0]
        if data_len > max_size:
            data = data[:max_size]
        elif data_len < max_size:
            pad_start_len = random.randint(0, max_size - data_len)
            pad_end_len = max_size - data_len - pad_start_len

            # Pad with 0s
            pad_start = np.zeros(pad_start_len)
            pad_end = np.zeros(pad_end_len)

            data = np.concatenate((pad_start, data, pad_end))
        return data
    @staticmethod
    def spectogram_mfcc(data, sr, n_fft):
        mfcc = librosa.feature.mfcc(y=data, sr=sr, n_fft=n_fft)
        mfcc = librosa.amplitude_to_db(mfcc)
        return mfcc
    @staticmethod
    def scale_spec(spec):
        spec = preprocessing.scale(spec, axis=1)
        return spec

    @staticmethod
    def spectro_gram(data, sr, n_mels, n_fft):

        spec = librosa.feature.melspectrogram(y=data, sr=sr, n_fft=n_fft, n_mels=n_mels)

        spec = librosa.amplitude_to_db(spec)
        return spec

    # This class will be expanded with additional transforms
    # TODO: https://towardsdatascience.com/preprocess-audio-data-with-the-signal-envelope-499e6072108

class AudioDataset(Dataset):
    def __init__(self, data, mfcc=True, scale=False):
        self.data = data
        self.mfcc = mfcc
        self.scale = scale
        self.sr = 16_000
        self.n_fft = 512
        self.n_mels = 256
        self.max_size = 16_000

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        file, sr = DataPrep.load(self.data['file'].iloc[item], self.sr)
        file = DataPrep.resize(file, self.max_size)
        if self.mfcc:
            spec = DataPrep.spectogram_mfcc(file, sr, self.n_fft)
        else:
            spec = DataPrep.spectro_gram(file, sr, self.n_mels, self.n_fft)
        if self.scale:
            spec = DataPrep.scale_spec(spec)
        return torch.from_numpy(spec), self.data['label'].iloc[item]

