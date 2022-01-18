import torch
from torchvision import models

model = models.resnet18(pretrained=True)
script_model = torch.jit.script(model)
#script_model.save('deployable_model.pt')
data = torch.rand(10,3,16,16)
assert torch.all(model(data) == script_model(data)) 