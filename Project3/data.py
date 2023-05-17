import pandas as pd
import numpy as np
import random
import torchvision.datasets
import torchvision.transforms
import torch

DATA_PATH = '.data/data0/lsun/bedroom'
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
IMAGE_SIZE = 64
SEED = 1217
random.seed(SEED)
torch.manual_seed(SEED)


def load_dataset(transform):
    dataset = torchvision.datasets.ImageFolder(root=DATA_PATH,
                                   transform=transform)
    return dataset


def load_dataloaders(transform, bs=64):
    dataset = load_dataset(transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=bs,
                                             shuffle=True)
    return dataloader


# main function to load data
def load_dataloader_preprocess(bs=64):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(IMAGE_SIZE),
        torchvision.transforms.CenterCrop(IMAGE_SIZE),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return load_dataloaders(transform, bs)
