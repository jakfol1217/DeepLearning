import copy
import os
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torch
import torchvision
import csv
from random import choices, sample
from torch.utils.data import Subset, ConcatDataset
import numpy as np

# Data loading and augmentation

name_dict = {'airplane': 0,
             'automobile': 1,
             'bird': 2,
             'cat': 3,
             'deer': 4,
             'dog': 5,
             'frog': 6,
             'horse': 7,
             'ship': 8,
             'truck': 9}


def return_cifar10_labels(path):
    with open(path) as csvfile:
        csvreader = csv.reader(csvfile)
        filelist = list(csvreader)
    for filename in filelist[1:]:
        label = name_dict.get(filename[1])
        filename[1] = label
    return filelist[1:]


class Cifar10Dataset(Dataset):
    def __init__(self, path, labels_path=None, transform=None):
        self.path = path
        self.names = os.listdir(path)
        self.labels = None
        self.transform = transform
        if labels_path is not None:
            self.labels = return_cifar10_labels(labels_path)

    def __getitem__(self, item):
        if self.transform is not None:
            if self.labels is not None:
                return self.transform(Image.open(self.path + '/' + self.names[item])), self.labels[item][1]
            return self.transform(Image.open(self.path + '/' + self.names[item]))
        if self.labels is not None:
            return Image.open(self.path + '/' + self.names[item]), self.labels[item][1]
        return Image.open(self.path + '/' + self.names[item])

    def __len__(self):
        return len(self.names)


global_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


def _split_train_val(dataset, size, transform, transform_test, bs):
    size_train = size * len(dataset)
    size_val = len(dataset) - size_train
    dataset_train, dataset_val = torch.utils.data.random_split(dataset, [int(size_train), int(size_val)])
    dataset_train.dataset = copy.deepcopy(dataset)
    dataset_train.dataset.transform = transform
    dataset_val.dataset.transform = transform_test
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=bs)
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=bs)
    return dataloader_train, dataloader_val


def load_cifar10_dataloaders_validation(transform=global_transform, bs=16):
    transform_test = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = torchvision.datasets.CIFAR10(".data", download=True)
    dataloader_train, dataloader_val = _split_train_val(dataset, 0.9, transform, transform_test, bs)
    dataset_test = torchvision.datasets.CIFAR10(".data", download=True, train=False, transform=transform_test)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=bs)
    return dataloader_train, dataloader_test, dataloader_val


# Kaggle loaders

def load_cifar10_train_dataloaders_validation_kaggle(path='.data-cifar/train', label_path='.data-cifar/trainLabels.csv',
                                                     transform=global_transform, bs=16):
    transform_test = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = Cifar10Dataset(path, label_path, transform)
    dataloader_train, dataloader_val = _split_train_val(dataset, 0.9, transform, transform_test, bs)
    return dataloader_train, dataloader_val


def load_cifar10_test_dataloader_kaggle(path='.data-cifar/test', bs=16):
    transform_test = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = Cifar10Dataset(path, transform=transform_test)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=bs)
    return dataloader


# DATA AUGMENTATION
# Images are 32 x 32 - there is no point in resizing them

# ---------------------READY FUNCTIONS ----------------------------------
class GaussianNoise(object):
    def __init__(self, variance):
        self.var = np.sqrt(variance)

    def __call__(self, img):
        size = img.size()
        img = img + (torch.randn(size) * self.var)
        return img


def augmented_cifar10_dataset_randomflip(bs=16):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return load_cifar10_dataloaders_validation(transform, bs)


def augmented_cifar10_dataset_randomrotate(rotate, bs=16):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomRotation(rotate),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return load_cifar10_dataloaders_validation(transform, bs)


def augmented_cifar10_dataset_randomcrop(size, bs=16):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    return load_cifar10_dataloaders_validation(transform, bs)


def augmented_cifar10_dataset_randomflip_rotate(rotate, bs=16):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomRotation(rotate),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return load_cifar10_dataloaders_validation(transform, bs)


def augmented_cifar10_dataset_gauss_noise(variance, bs=16):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        GaussianNoise(variance)
    ])
    return load_cifar10_dataloaders_validation(transform, bs)


def augmented_cifar10_dataset_randomapply(rotate, bs=16):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomApply(transforms=[torchvision.transforms.RandomHorizontalFlip(),
                                                       torchvision.transforms.RandomRotation(rotate)], p=0.5),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return load_cifar10_dataloaders_validation(transform, bs)
