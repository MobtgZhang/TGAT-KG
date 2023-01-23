import torch
import torch.nn as nn

from .utils import to_var
class FBetaModel(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(predict,target):
        pass
class PreCallModel(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,predict,target):
        pass
class ReCallModel(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,predict,target):
        pass
class AccModel(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,predict,target):
        pass

def evaluate_model(model,loss_fn,graph,data_loader,device):
    model.eval()
    for item in data_loader:
        to_var(item,device)
        head,rel,tail,target = item
        logits = model(head,rel,tail,graph.edge_index)
        loss = loss_fn(logits,target)
        
