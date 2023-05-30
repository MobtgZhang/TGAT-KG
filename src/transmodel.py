import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import TransE,TransD,TransH,TransR

class TransModel(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.num_ents = config.num_ents
        self.num_rels = config.num_rels
        self.sigmoid = config.sigmoid
        print(config.get_attrs())
        trans_dict = {
            "TransE":TransE,
            "TransD":TransD,
            "TransH":TransH,
            "TransR":TransR,
        }
        self.model = trans_dict[config.model_name](**config.get_attrs())
        self.lin = nn.Linear(self.model.out_dim,2)
    def forward(self,head,rel,tail):
        o_emb = self.model(head,rel,tail)
        logits = self.lin(o_emb)
        if self.sigmoid:
            return torch.sigmoid(logits)
        else:
            return F.softmax(logits,dim=-1)
            
