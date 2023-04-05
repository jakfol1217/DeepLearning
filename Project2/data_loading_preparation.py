import pandas as pd
import numpy as np
from sklearn import preprocessing
from torch.utils.data import DataLoader, Dataset
import librosa
import glob
import torch
import os
import re
from pydub import AudioSegment

import random

random.seed(1217)  # <- remember that

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
             'silence': 10}


# unknown = 11


def split_silence_to_chunks(path=''):
    if len(path) > 0 and path[-1] != '/':
        path += '/'
    files = glob.glob(path + '.data-audioset/train/audio/_background_noise_/*.wav')

    if len(files) > 6:  # files have already been split
        names = glob.glob(path + '.data-audioset/train/audio/_background_noise_/*_split.wav')
        names = [str.replace(name, '\\', '/') for name in names]
        df_silence = pd.DataFrame({'file': names,
                                   'label': ['silence' for i in range(len(names))]})
        return df_silence, names

    names = []
    for file in files:
        audio = AudioSegment.from_wav(file)
        for i in range(len(audio) // 1000):
            name = str.split(file, '.wav')[0] + '_' + str(i) + '_split.wav'
            audio_cut = audio[1000 * i:1000 * (i + 1)]
            audio_cut.export(name, format="wav")
            names.append(name)
    names = [str.replace(name, '\\', '/') for name in names]
    df_silence = pd.DataFrame({'file': names,
                               'label': ['silence' for i in range(len(names))]})
    return df_silence, names


# limit_11 -- percentage of 11-th class that is to remain in datasets (values from 0 to 1)
def get_audio_datasets(path='', limit_11=0.5):
    if len(path) > 0 and path[-1] != '/':
        path += '/'
    # GET FULL DATASET
    files = glob.glob(path + '.data-audioset/train/audio/*/*.wav')
    full_data = pd.DataFrame(pd.Series(files).apply(lambda x: str.replace(x, '\\', '/')))
    full_data['label'] = full_data[0].apply(lambda x: x.split('/')[3])
    full_data = full_data.rename({0: 'file'}, axis=1)
    full_data = full_data[full_data['label'] != '_background_noise_']
    # ADD NOISE
    df_silence, names = split_silence_to_chunks(path)
    full_data = pd.concat([full_data, df_silence])

    # ASSIGN LABELS TO CLASSES
    full_data['label'] = full_data['label'].map(name_dict).fillna(11)
    # Limiting the number of "11" class
    idx = full_data['label'] == 11
    full_data = full_data[[i if i and random.uniform(0, 1) < limit_11 else not i for i in idx]]
    full_data = full_data.sample(frac=1).reset_index(drop=True)
    # GET TEST AND VALIDATION DATASETS
    test_data = (pd.read_csv(path + '.data-audioset/train/testing_list.txt', header=None)
                 .apply(lambda x: '.data-audioset/train/audio/' + x))
    valid_data = (pd.read_csv(path + '.data-audioset/train/validation_list.txt', header=None)
                  .apply(lambda x: '.data-audioset/train/audio/' + x))
    test_data = pd.concat([test_data, pd.Series([names[i] for i in [12, 120, 170]])])
    valid_data = pd.concat([valid_data, pd.Series([names[i] for i in [-12, -120, -170]])])
    test_data = full_data[full_data['file'].isin(test_data[0])].reset_index()
    valid_data = full_data[full_data['file'].isin(valid_data[0])].reset_index()
    # GET TRAIN DATASET
    train_data = full_data[~full_data['file'].isin(pd.concat([test_data['file'], valid_data['file']]))]
    # CREATE DATASETS
    train_dataset = AudioDataset(train_data, 'train')
    valid_dataset = AudioDataset(valid_data, 'val')
    test_dataset = AudioDataset(test_data, 'test')
    # CACHING MANAGEMENT
    # If cached folder do not exist, create it
    for dats in ['train', 'val', 'test']:
        if not os.path.exists('.data-audioset/cached_' + dats):
            os.makedirs('.data-audioset/cached_' + dats)
        files = glob.glob('.data-audioset/cached_' + dats + '/*')
        for f in files:
            os.remove(f)
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
    def load_cached(path, item, string):
        cached_path = str.split(path, '.wav')
        cached_path = cached_path[0] + f'_cached_' + str(item)  # adding item num because not all names are unique
        cached_path = re.sub('train/audio/.*/', f'cached_{string}/', cached_path)
        try:
            spec = torch.load(cached_path)
        except:
            return None
        return spec

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
    def __init__(self, data, cache_str, mfcc=True, scale=False):
        self.data = data
        self.mfcc = mfcc
        self.scale = scale
        self.sr = 16_000
        self.n_fft = 512
        self.n_mels = 256
        self.max_size = 16_000
        self.cache_str = cache_str

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        path, sr = self.data['file'].iloc[item], self.sr
        # Loading cached file (tensor file)
        spec = DataPrep.load_cached(path, item, self.cache_str)
        if spec is not None:
            return spec, self.data['label'].iloc[item]

        file, sr = DataPrep.load(path, sr)
        file = DataPrep.resize(file, self.max_size)
        if self.mfcc:
            spec = DataPrep.spectogram_mfcc(file, sr, self.n_fft)
        else:
            spec = DataPrep.spectro_gram(file, sr, self.n_mels, self.n_fft)
        if self.scale:
            spec = DataPrep.scale_spec(spec)
        spec = torch.from_numpy(spec).float()
        cached_path = str.split(path, '.wav')
        cached_path = cached_path[0] + f'_cached_' + str(item)  # adding item num because not all names are unique
        cached_path = re.sub('train/audio/.*/', f'cached_{self.cache_str}/', cached_path)
        torch.save(spec, cached_path)
        return spec, self.data['label'].iloc[item]