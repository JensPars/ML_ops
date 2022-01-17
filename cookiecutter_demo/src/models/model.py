def model():
    import torchvision.models as models
    import torch.nn as nn
    resnet18 = models.resnet18()
    resnet18.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    resnet18.fc = nn.Linear(in_features=512, out_features=10, bias=True)
    return resnet18




