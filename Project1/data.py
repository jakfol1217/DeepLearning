import os
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torch
import torchvision
import csv

# Data loading and augmentation

name_dict={'airplane':0,
           'automobile':1,
           'bird':2,
           'cat':3,
           'deer':4,
           'dog':5,
           'frog':6,
           'horse':7,
           'ship':8,
           'truck':9}

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


def load_cifar10_dataloaders():
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset_train = torchvision.datasets.CIFAR10(".data", download=True, transform=transform)
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=16)
    dataset_test = torchvision.datasets.CIFAR10(".data", download=True, train=False, transform=transform)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=16)
    return dataloader_train, dataloader_test

def load_cifar10_dataloaders_validation():
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = torchvision.datasets.CIFAR10(".data", download=True, transform=transform)
    size_train = 0.9 * len(dataset)
    size_val = len(dataset) - size_train
    dataset_train, dataset_val = torch.utils.data.random_split(dataset, [int(size_train), int(size_val)])
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=16)
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=16)
    dataset_test = torchvision.datasets.CIFAR10(".data", download=True, train=False, transform=transform)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=16)
    return dataloader_train, dataloader_test, dataloader_val

# Kaggle loaders

def load_cifar10_train_dataloader_kaggle(path='.data-cifar/train', label_path='.data-cifar/trainLabels.csv'):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = Cifar10Dataset(path, label_path, transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16)
    return dataloader

def load_cifar10_train_dataloaders_validation_kaggle(path='.data-cifar/train', label_path='.data-cifar/trainLabels.csv'):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = Cifar10Dataset(path, label_path, transform)
    size_train = 0.9 * len(dataset)
    size_val = len(dataset) - size_train
    dataset_train, dataset_val = torch.utils.data.random_split(dataset, [int(size_train), int(size_val)])
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=16)
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=16)
    return dataloader_train, dataloader_val

def load_cifar10_test_dataloader_kaggle(path='.data-cifar/test'):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = Cifar10Dataset(path, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16)
    return dataloader

