import torch
import torchvision

# Data loading and augmentation

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
    size_train = 0.9*len(dataset)
    size_val = len(dataset) - size_train
    dataset_train, dataset_val = torch.utils.data.random_split(dataset, [int(size_train), int(size_val)])
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=16)
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=16)
    dataset_test = torchvision.datasets.CIFAR10(".data", download=True, train=False, transform=transform)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=16)
    return dataloader_train, dataloader_test, dataloader_val
