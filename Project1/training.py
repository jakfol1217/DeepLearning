import torch
import torch.nn as nn
from architecture import training_func, vgg11_bn
from data import load_cifar10_dataloaders_validation

train, test, val = load_cifar10_dataloaders_validation()

model = vgg11_bn('cpu', pretrained=False)

criterion = torch.nn.CrossEntropyLoss()
cifar10_optim = torch.optim.Adam(model.parameters(), lr=1e-3)

training_func(model, cifar10_optim, criterion, train, val, 3)