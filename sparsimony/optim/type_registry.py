import torch.nn as nn

TYPE_REGISTRY = {nn.Linear: "weight", nn.Conv2d: "weight"}
