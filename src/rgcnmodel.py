import torch
import torch.nn as nn
import torch.functional as F
from .modules import RGCNConv

class RGCNModel(nn.Module):
    def __init__(self,config):
        self.rgcn = config
    def forward(self):
        pass
