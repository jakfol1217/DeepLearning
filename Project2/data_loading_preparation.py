import copy

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


class DataPrep:
    # By default, the data have 16000 sampling rate and are monochannel
    @staticmethod
    def load(path, sr, mono=True):
        data, sr = librosa.load(path, sr=sr, mono=mono)
        return data, sr

    @staticmethod
    def load_cached(path, item, string, aug=""):
        cached_path = str.split(path, '.wav')
        cached_path = cached_path[0] + f'_cached_' + str(item)  # adding item num because not all names are unique
        cached_path = re.sub('train/audio/.*/', f'cached_{string}/', cached_path)
        try:
            spec = torch.load(cached_path)
        except:
            return None
        return spec

    @staticmethod
    def save_cache(path, spec, string):
        cached_path = str.split(path, '.wav')
        cached_path = cached_path[0] + f'_cached_' + \
                      str.split(path, '/')[3]  # adding class name because not all file names are unique
        cached_path = re.sub('train/audio/.*/', f'cached_{string}/', cached_path)
        torch.save(spec, cached_path)

    @staticmethod
    def save_cache_aug(path, spec, string, aug):
        cached_path = str.split(path, '.wav')
        cached_path = cached_path[0] + f'_cached_' + \
                      str.split(path, '/')[3] + "_" + aug  # adding class name because not all file names are unique
        cached_path = re.sub('train/audio/.*/', f'cached_{string}/', cached_path)
        torch.save(spec, cached_path)

    # -----------------PREPROCESSING-----------------
    # ----------------- data - means audio signal in time domain, that is, what is read from .wav file -----------------
    # ----------------- spec - means the spetogram constructed from data -----------------
    # ----------------- sr - sample rate, according to readme- 16_000 -----------------
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

    # -----------------AUGMENTATION-----------------
    @staticmethod
    def time_shift(data, limit, sr=None):
        sig_len = data.shape[0]
        shift_amt = int(random.random() * limit * sig_len)
        return np.roll(data, shift_amt)

    @staticmethod
    def pitch_shift(data, factor, sr=16_000):
        augmented_data = librosa.effects.pitch_shift(y=data, sr=sr, n_steps=factor)
        return augmented_data

    @staticmethod
    def speed_shift(data, rate, sr=None):
        augmented_data = librosa.effects.time_stretch(y=data, rate=rate)
        augmented_data = librosa.core.resample(y=augmented_data, orig_sr=len(augmented_data), target_sr=16000)
        return augmented_data

    @staticmethod
    def freq_mask(spec, freq_limit=3, num_masks=1):
        spec_aug = copy.deepcopy(spec)
        num_mel_channels = spec_aug.shape[0]
        mn = spec_aug.mean()
        for i in range(num_masks):
            window = random.randrange(1, freq_limit)
            win_end = random.randrange(window, num_mel_channels)
            spec_aug[max(0, win_end - window):win_end] = mn
        return spec_aug

    @staticmethod
    def time_mask(spec, time_limit=5, num_masks=1):
        spec_aug = copy.deepcopy(spec)
        length = spec_aug.shape[1]
        mn = spec_aug.mean()
        for i in range(num_masks):
            window = random.randrange(1, time_limit)
            win_end = random.randrange(window, length)
            spec_aug[:, max(0, win_end - window):win_end] = mn
        return spec_aug

    # -----------------CREATE SPECTOGRAM-----------------
    @staticmethod
    def spectogram_mfcc(data, sr, n_fft):
        mfcc = librosa.feature.mfcc(y=data, sr=sr, n_fft=n_fft)
        return mfcc

    @staticmethod
    def spec_to_db(spec):
        spec = librosa.amplitude_to_db(spec)
        return spec

    @staticmethod
    def scale_spec(spec):
        spec = preprocessing.scale(spec, axis=1)
        return spec

    @staticmethod
    def spectro_gram(data, sr, n_mels, n_fft):

        spec = librosa.feature.melspectrogram(y=data, sr=sr, n_fft=n_fft, n_mels=n_mels)

        spec = librosa.amplitude_to_db(spec)
        return spec


# -------------------------DEFAULT PARAMS-------------------------
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

default_params = {
    'mfcc': True,
    'scale': False,
    'to_db': True,
    'sr': 16_000,
    'n_fft': 512,
    'n_mels': 256,
    'max_size': 16_000
}

default_transforms = {
    'transform_time': None,
    'transform_spec': None
}


# -------------------------TRANSFORMATION CLASSES-------------------------
class FreqMask(object):
    def __init__(self, num_masks=1, freq_limit=3, p=0.5):
        self.num_masks = num_masks
        self.limit = freq_limit
        self.p = p

    def __call__(self, spec):
        spec_aug = copy.deepcopy(spec)
        if random.uniform(0, 1) <= self.p:
            spec_aug = DataPrep.freq_mask(spec_aug, self.limit, self.num_masks)
        return spec_aug


class TimeMask(object):
    def __init__(self, num_masks=1, time_limit=5, p=0.5):
        self.num_masks = num_masks
        self.limit = time_limit
        self.p = p

    def __call__(self, spec):
        spec_aug = copy.deepcopy(spec)
        if random.uniform(0, 1) <= self.p:
            spec_aug = DataPrep.time_mask(spec_aug, self.limit, self.num_masks)
        return spec_aug


class TimeShift(object):
    def __init__(self, rate=5, p=0.5):
        self.rate = rate
        self.p = p

    def __call__(self, data):
        data_aug = copy.deepcopy(data)
        data_aug = DataPrep.time_shift(data_aug, self.rate)
        return data_aug


class PitchShift(object):
    def __init__(self, rate=5, p=0.5):
        self.rate = rate
        self.p = p

    def __call__(self, data):
        data_aug = copy.deepcopy(data)
        data_aug = DataPrep.pitch_shift(data_aug, self.rate)
        return data_aug


class SpeedShift(object):
    def __init__(self, rate, p=0.5):
        self.rate = rate
        self.p = p

    def __call__(self, data):
        data_aug = copy.deepcopy(data)
        data_aug = DataPrep.speed_shift(data_aug, self.rate)
        return data_aug

# ALL THE POSSIBLE AUGMENTS AS A REFERENCE
#augments = {
#    'transform_time': [TimeShift(5), PitchShift(5), SpeedShift(5)],
#    'transform_spec': [FreqMask(5), TimeMask(5)]
#}


# -------------------------END OF TRANSFORMATION CLASSES-------------------------

def split_silence_to_chunks(path=''):
    """
    split background noise files into chunks for "noise" class
    :param path: non-absolute path to audio files
    :return: split files names in form of dataframe
    """
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


def get_audio_datasets(path='', limit_11=0.5, preprocess_params=default_params, transforms=default_transforms):
    """
    create datasets from audio files
    :param path: non-absolute path to audio files
    :param limit_11: percentage of "unknown" class to include in datasets (values from 0 to 1)
    :param preprocess_params: preprocessing parameters
    :param transforms: augmentations for training dataset
    :return: train, test and validation datasets
    """
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
    full_data = full_data.sample(frac=1, random_state=2137).reset_index(drop=True)
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
    train_dataset = AudioDataset(train_data, 'train', **preprocess_params, **transforms)
    valid_dataset = AudioDataset(valid_data, 'val', **preprocess_params)
    test_dataset = AudioDataset(test_data, 'test', **preprocess_params)
    # CACHING MANAGEMENT
    # If cached folder does not exist, create it
    for dats in ['train', 'val', 'test']:
        if not os.path.exists('.data-audioset/cached_' + dats):
            os.makedirs('.data-audioset/cached_' + dats)
        # files = glob.glob('.data-audioset/cached_' + dats + '/*') #emptying cache <- disabled
        # for f in files: # IT IS ADVISED TO ENABLE EMPTYING CACHE WHEN PREPROCESSING PARAMETERS ARE CHANGED
        #    os.remove(f)
    return train_dataset, test_dataset, valid_dataset


def load_audio_dataloaders_validation(path='', bs=16, limit_11=0.5, preprocess_params=default_params,
                                      transforms=default_transforms):
    dataset_train, dataset_test, dataset_val = get_audio_datasets(path=path, limit_11=limit_11,
                                                                  preprocess_params=preprocess_params,
                                                                  transforms=transforms)
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=bs)
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=bs)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=bs)
    return dataloader_train, dataloader_test, dataloader_val


# DEFINING SOME ADITIONAL DATALOADER GETTERS
def load_audio_dataloaders_timeshift(path='', bs=16, limit_11=0.5, rate=5):
    # rate <- parameter for the time shift
    augments = {
        'transform_time': TimeShift(rate),
        'transform_spec': None
    }
    return load_audio_dataloaders_validation(path=path, bs=bs, limit_11=limit_11,
                                             preprocess_params=default_params,
                                             transforms=augments)


def load_audio_dataloaders_pitchshift(path='', bs=16, limit_11=0.5, rate=5):
    # rate <- parameter for the pitch shift
    augments = {
        'transform_time': PitchShift(rate),
        'transform_spec': None
    }
    return load_audio_dataloaders_validation(path=path, bs=bs, limit_11=limit_11,
                                             preprocess_params=default_params,
                                             transforms=augments)


def load_audio_dataloaders_speedshift(path='', bs=16, limit_11=0.5, rate=5):
    # rate <- parameter for the speed shift
    augments = {
        'transform_time': SpeedShift(rate),
        'transform_spec': None
    }
    return load_audio_dataloaders_validation(path=path, bs=bs, limit_11=limit_11,
                                             preprocess_params=default_params,
                                             transforms=augments)


def load_audio_dataloaders_timemask(path='', bs=16, limit_11=0.5, num_masks=1, limit=5):
    # num_masks , time_limit <- parameters for time masking
    augments = {
        'transform_time': None,
        'transform_spec': TimeMask(num_masks, time_limit=limit)
    }
    return load_audio_dataloaders_validation(path=path, bs=bs, limit_11=limit_11,
                                             preprocess_params=default_params,
                                             transforms=augments)


def load_audio_dataloaders_freqmask(path='', bs=16, limit_11=0.5, num_masks=1, limit=5):
    # num_masks , time_limit <- parameters for frequency masking
    augments = {
        'transform_time': None,
        'transform_spec': FreqMask(num_masks, freq_limit=limit)
    }
    return load_audio_dataloaders_validation(path=path, bs=bs, limit_11=limit_11,
                                             preprocess_params=default_params,
                                             transforms=augments)


def load_audio_dataloaders_timeshift_freqmask(path='', bs=16, limit_11=0.5, num_masks=1, limit=5, rate=5):
    augments = {
        'transform_time': TimeShift(rate=rate),
        'transform_spec': FreqMask(num_masks, freq_limit=limit)
    }
    return load_audio_dataloaders_validation(path=path, bs=bs, limit_11=limit_11,
                                             preprocess_params=default_params,
                                             transforms=augments)

# -------------------------END OF CUSTOM DATALOADER GETTERS------------------------

def cache_all(preprocess_params=default_params, transforms=default_transforms):
    """
    Cache all dataset entries to speed up learning process
    :param preprocess_params: parameters for preprocessing
    :param transforms: augmentations for training dataset (time domain augmentations are also cached)
    """
    train_dataset, test_dataset, valid_dataset = get_audio_datasets(limit_11=1,
                                                                    preprocess_params=preprocess_params,
                                                                    transforms=transforms)
    # simply access all the files, they will all get cached
    print("Processing train dataset:")
    for a in range(len(train_dataset)):
        train_dataset[a]
        print(f"Processed {a + 1} out of {len(train_dataset)}", end="\r")
    print("Train dataset processed")
    print("Processing test dataset:")
    for a in range(len(test_dataset)):
        test_dataset[a]
        print(f"Processed {a + 1} out of {len(test_dataset)}", end="\r")
    print("Test dataset processed")
    print("Processing validation dataset:")
    for a in range(len(valid_dataset)):
        valid_dataset[a]
        print(f"Processed {a + 1} out of {len(valid_dataset)}", end="\r")
    print("Validation dataset processed")


# AUDIO DATASET CLASS
class AudioDataset(Dataset):
    def __init__(self, data, cache_str, mfcc=True, scale=False, to_db=True, transform_time=None, transform_spec=None,
                 sr=16_000, n_fft=512,
                 n_mels=256,
                 max_size=16_000):
        self.data = data
        # SPECTOGRAM OPTIONS
        self.mfcc = mfcc
        self.scale = scale
        self.to_db = to_db

        # AUDIO PARAMETERS
        self.sr = sr
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.max_size = max_size
        # CACHE PATH
        self.cache_str = cache_str
        # TRANSFORMATIONS
        self.transform_time = transform_time
        self.transform_spec = transform_spec
        if isinstance(self.transform_time, list):
            self.p = self.transform_time[0].p
        elif self.transform_time is not None:
            self.p = self.transform_time.p
        else:
            self.p = 0.5

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        path, sr = self.data['file'].iloc[item], self.sr
        lbl = self.data['label'].iloc[item]
        # LOADING CACHED FILE (TENSOR FILE)
        spec = DataPrep.load_cached(path, str.split(path, '/')[3], self.cache_str)
        if spec is not None:
            if lbl != 11 and self.transform_time is not None and random.uniform(0, 1) <= self.p:
                spec_aug = DataPrep.load_cached(path, str.split(path, '/')[3], self.cache_str, 'aug')
                spec = spec_aug
            # SPECTOGRAM AUGMENTATION
            spec_aug = self._augment_spec(spec)
            return spec_aug, self.data['label'].iloc[item]
        # IF NOT CACHED, FILE PREPARATION
        file, sr = DataPrep.load(path, sr)
        # TIME AUGMENTATION CACHING
        if lbl != 11 and self.transform_time is not None:
            file_aug, name = self._augment_time(file)
            if len(name) > 0:
                spec_aug = self._preprocess(file_aug)
                DataPrep.save_cache_aug(path, spec_aug, self.cache_str, name)
        spec = self._preprocess(file)
        # SAVE SPEC TO CACHE
        DataPrep.save_cache(path, spec, self.cache_str)
        # SPECTOGRAM AUGMENTATION
        spec_aug = self._augment_spec(spec)
        return spec_aug, self.data['label'].iloc[item]

    def _preprocess(self, data):
        sr = self.sr
        file = DataPrep.resize(data, self.max_size)
        if self.mfcc:
            spec = DataPrep.spectogram_mfcc(file, sr, self.n_fft)
        else:
            spec = DataPrep.spectro_gram(file, sr, self.n_mels, self.n_fft)
        if self.to_db:
            spec = DataPrep.spec_to_db(spec)
        if self.scale:
            spec = DataPrep.scale_spec(spec)
        spec = torch.from_numpy(spec).float()
        return spec

    def _augment_spec(self, spec):
        if self.transform_spec is None:
            return spec
        spec_aug = copy.deepcopy(spec)
        if isinstance(self.transform_spec, list):
            for aug in self.transform_spec:
                spec_aug = aug(spec_aug)
        else:
            spec_aug = self.transform_spec(spec_aug)
        return spec_aug

    def _augment_time(self, data):
        if self.transform_time is None:
            return data, ""
        data_aug = copy.deepcopy(data)
        if isinstance(self.transform_time, list):
            for aug in self.transform_time:
                data_aug = aug(data_aug)
        else:
            data_aug = self.transform_time(data_aug)
        return data_aug, "aug"
