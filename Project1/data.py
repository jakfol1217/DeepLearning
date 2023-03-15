import os
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torch
import torchvision
import csv
from random import choices, sample
from torch.utils.data import Subset, ConcatDataset

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

transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

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


def load_cifar10_dataloaders(transform=transform):
    dataset_train = torchvision.datasets.CIFAR10(".data", download=True, transform=transform)
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=16)
    dataset_test = torchvision.datasets.CIFAR10(".data", download=True, train=False, transform=transform)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=16)
    return dataloader_train, dataloader_test

def load_cifar10_dataloaders_validation(transform=transform):
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

def load_cifar10_train_dataloader_kaggle(path='.data-cifar/train', label_path='.data-cifar/trainLabels.csv', transform=transform):
    dataset = Cifar10Dataset(path, label_path, transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16)
    return dataloader

def load_cifar10_train_dataloaders_validation_kaggle(path='.data-cifar/train', label_path='.data-cifar/trainLabels.csv', transform=transform):
    dataset = Cifar10Dataset(path, label_path, transform)
    size_train = 0.9 * len(dataset)
    size_val = len(dataset) - size_train
    dataset_train, dataset_val = torch.utils.data.random_split(dataset, [int(size_train), int(size_val)])
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=16)
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=16)
    return dataloader_train, dataloader_val

def load_cifar10_test_dataloader_kaggle(path='.data-cifar/test', transform=transform):
    dataset = Cifar10Dataset(path, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16)
    return dataloader

# DATA AUGMENTATION
# Images are 32 x 32 - there is no point in resizing them

# PERCENTAGE IS INT- 2 MEANS 200% OF DATASET SIZE (minus 10% of validation) IS AUGMENTED DATA
# ---------------------READY FUNCTIONS ----------------------------------
def augmented_cifar10_dataset_randomflip(aug_percentage):
    transform = torchvision.transforms.RandomHorizontalFlip()
    return augmented_cifar10_dataset(aug_percentage, transform)

def augmented_cifar10_dataset_randomrotate(aug_percentage, rotate):
    transform = torchvision.transforms.RandomRotation(rotate)
    return augmented_cifar10_dataset(aug_percentage, transform)

def augmented_cifar10_dataset_randomflip_rotate(aug_percentage, rotate):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomRotation(rotate)
    ])
    return augmented_cifar10_dataset(aug_percentage, transform)

def augmented_cifar10_dataset_gaussianblur(aug_percentage, kernel_size, sigma):
    transform = torchvision.transforms.GaussianBlur(kernel_size, sigma)
    return augmented_cifar10_dataset(aug_percentage, transform)

def augmented_cifar10_dataset_full(aug_percentage, rotate, kernel_size, sigma):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomRotation(rotate),
        torchvision.transforms.GaussianBlur(kernel_size, sigma)
    ])
    return augmented_cifar10_dataset(aug_percentage, transform)


# ---------------------DEFINE-YOUR-OWN-TRANSFORMATION FUNCTION ----------------------------------
def return_class_indexes(dataset):
    class_indexes = [[] for i in range(10)]
    for i in range(len(dataset)):
        class_indexes[dataset[i][1]].append(i)
    return class_indexes
def augmented_cifar10_dataset(aug_percentage, aug_transform):
    # normal transform and augmentation transforms
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    transform_full = torchvision.transforms.Compose([
        aug_transform, transform
    ])
    # getting cifar 10 dataset
    dataset = torchvision.datasets.CIFAR10(".data", download=True)
    dataset_test = torchvision.datasets.CIFAR10(".data", download=True, train=False, transform=transform)
    # get indexes for each class
    class_indexes = return_class_indexes(dataset)
    # get train and val datasets
    indexes_val = []
    for i in range(len(class_indexes)):
        indexes_val += sample(class_indexes[i], k=int(len(class_indexes[i]) * 0.1))
    indexes_train = [i for i in range(len(dataset)) if i not in indexes_val]
    dataset_val = Subset(dataset, indexes_val)
    dataset_train = Subset(dataset, indexes_train)
    # get subset of dataset (can be bigger than dataset itself, also classes are still balanced)
    class_indexes = return_class_indexes(dataset_train)
    indexes = []
    for i in range(len(class_indexes)):
        indexes += choices(class_indexes[i], k=int(len(class_indexes[i]) * aug_percentage))
    dataset_augmented = Subset(dataset, indexes)
    # run transforms
    dataset_train.dataset.transform = transform
    dataset_val.dataset.transform = transform
    dataset_augmented.dataset.transform = transform_full
    dataset_full = ConcatDataset((dataset_train, dataset_augmented))
    # create dataloaders (train, test and validation)
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=16)
    dataloader_full = torch.utils.data.DataLoader(dataset_full, batch_size=16, shuffle=True)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=16)
    return dataloader_full, dataloader_test, dataloader_val



