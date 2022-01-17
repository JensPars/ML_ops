from src.models.model import *
from src.data.dataset import Dataset
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
import torch

trainset = Dataset()
trainloader = DataLoader(trainset, batch_size=16, shuffle=True)
mod = model()
criterion = CrossEntropyLoss()
optimizer = optim.Adam(mod.parameters(), lr=0.0001)

running_loss = 0
try:
    for images, labels in trainloader:
        out = mod(images.unsqueeze(1).type(torch.float32))
        loss = criterion(out,labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        running_loss += loss.item()
        break
    print(f"Training loss: {running_loss/len(trainloader)}")
except:
    'Training Failed'