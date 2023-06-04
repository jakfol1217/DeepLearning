import pandas as pd
import numpy as np
import random
import torchvision.datasets
import torchvision.transforms
import torch

DATA_PATH = '.data/data0/lsun/bedroom'
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
SEED = 1217
random.seed(SEED)
torch.manual_seed(SEED)


def load_dataset(transform, path):
    dataset = torchvision.datasets.ImageFolder(root=path,
                                   transform=transform)
    return dataset


def load_dataloaders(path, transform, bs=64):
    dataset = load_dataset(transform, path)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=bs,
                                             shuffle=True)
    return dataloader


# main function to load data
def load_dataloader_preprocess(image_size=64, bs=64, path=DATA_PATH):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(image_size),
        torchvision.transforms.CenterCrop(image_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return load_dataloaders(path, transform, bs)

