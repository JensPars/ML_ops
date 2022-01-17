from tests import _PATH_DATA, _PROJECT_ROOT
import torch
#from src.data.dataset import *
from src.data.dataset import *
from src.models.model import *

dataset = Dataset()
assert len(dataset) == 60000
## Checking input dims
assert dataset[0][0].shape == (28,28)
## Checking unique classes
assert len(dataset.classes()) == 10
## Checking output dims
assert model()(torch.zeros(3,1,28,28)).shape == (3,10)
