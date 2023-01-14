import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv

from .appnp import PRbinaryHop
from .fagcn import FAPagateNet

class FNNNet(nn.Module):
    def __init__(self,in_dim,hid_dim,out_dim):
        super(FNNNet,self).__init__()
        self.lin = nn.Linear(in_dim,hid_dim)
        self.op = nn.Linear(hid_dim,out_dim)
    def forward(self,inpx):
        hid = F.relu(self.lin(inpx))
        return self.op(hid)
class KGTConv(nn.Module):
    def __init__(self,config):
        super(KGTConv,self).__init__()
        self.num_ents = config.num_ents
        self.num_rels = config.num_rels
        self.emb_dim = config.emb_dim
        self.k_fa = config.k_fa
        self.fa_rho = config.fa_rho
        self.fa_alpha = config.fa_alpha
        self.fa_dropout = config.fa_dropout

        self.k_pr = config.k_pr
        self.pr_alpha = config.pr_alpha
        self.pr_beta = config.pr_beta
        self.pr_dropout = config.pr_dropout
        self.ent_emb = nn.Embedding(config.num_ents,config.emb_dim)
        self.fa_net = FAPagateNet(config.k_fa,config.fa_rho,config.fa_alpha,config.fa_dropout)
        self.rgcn = RGCNConv(config.emb_dim, config.emb_dim,config.num_rels)
        self.appnet = PRbinaryHop(config.k_pr,config.pr_alpha,config.pr_beta,config.pr_dropout)
    def forward(self,head,rel,tail,edge_index):
        h_emb = self.ent_emb(head)
        t_emb = self.ent_emb(tail)
        h_emb = self.appnet(h_emb,edge_index)
        t_emb = self.appnet(h_emb,edge_index,backward=True)
        print(h_emb.shape,t_emb.shape)
    def graph_forward(self,x,edge_type,edge_index):
        x_emb = self.ent_emb(x)
        print(x_emb.shape,edge_index.shape)
        #e_fea = self.fa_net(x_emb,edge_index)
        #app_fea = self.appnet(x_emb,edge_index)
        x_emb = self.rgcn(x_emb,edge_index,edge_type)
        print(x_emb.shape)
        
