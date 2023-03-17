import torch
import torch.nn as nn


# File based on https://github.com/huyvnphan/PyTorch_CIFAR10/
class VGG(nn.Module):
    def __init__(self, features, num_classes=10):
        super(VGG, self).__init__()
        self.features = features

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Linear(512 * 1 * 1, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

 # Create convolution part of VGG11_bn architecture
def make_vgg11_bn_layers(cfg = None):
    if cfg == None:
        cfg = [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"]
    layers = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

 # Create VGG11_bn model
def vgg11_bn(device="cpu", pretrained=True, num_classes=10):
    weights_path = ".weights/state_dicts/vgg11_bn.pt"
    model = VGG(make_vgg11_bn_layers(), num_classes=num_classes)
    if pretrained:
        state_dict = torch.load(
            weights_path, map_location=device
        )
        model.load_state_dict(state_dict)
    return model


def eval_accuracy(model, dataloader, training_device='cpu'):
    with torch.no_grad():
        model.to(training_device)
        correct = 0
        all_so_far = 0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(training_device), labels.to(training_device)
            pred = torch.argmax(model(inputs), dim=1)

            all_so_far += labels.size().numel()
            correct += torch.sum(pred.eq(labels))
    return correct/all_so_far

# Returns list of list as in sample submission
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

def predict_kaggle_test(model, dataloader, training_device='cpu'):
    with torch.no_grad():
        model.to(training_device)
        labels=[["id", "label"]]
        for number, inputs in enumerate(dataloader):
            inputs = inputs.to(training_device)
            pred = torch.argmax(model(inputs), dim=1)
            val = [number, list(name_dict.keys())[pred.item()]]
            labels.append(val)
    return labels

def training_func(model, optimizer, criterion, dataloader_train, dataloader_val, max_epochs, training_device='cpu', *_args, **_kwargs):
    model.train()
    model.to(training_device)
    #torch.cuda.empty_cache()
    for epoch in range(max_epochs):
        for inputs, labels in dataloader_train:
            inputs, labels = inputs.to(training_device), labels.to(training_device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
        print("Epoch: {}, Accuracy on validation set: {}".format(epoch, eval_accuracy(model, dataloader_val, training_device)))
    #torch.cuda.empty_cache()