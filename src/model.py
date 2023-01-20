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
        self.rel_emb = nn.Embedding(config.num_rels,config.emb_dim)
        self.fa_net = FAPagateNet(config.k_fa,config.fa_rho,config.fa_alpha,config.fa_dropout)
        self.rgcn = RGCNConv(config.emb_dim, config.emb_dim,config.num_rels)
        self.appnet = PRbinaryHop(config.k_pr,config.pr_alpha,config.pr_beta,config.pr_dropout)
        # the feaures
        self.rg_feature = nn.Parameter(torch.Tensor(size=(config.num_ents,config.emb_dim)),requires_grad=True)
        self.ap_feature = nn.Parameter(torch.Tensor(size=(config.num_ents,config.emb_dim)),requires_grad=True)
        self.fa_feature = nn.Parameter(torch.Tensor(size=(config.num_ents,config.emb_dim)),requires_grad=True)
        # the mlp layer
        tmp_emb = (config.emb_dim*2)*3+config.emb_dim
        self.fnn = FNNNet(tmp_emb,config.emb_dim,2)
    def transE_score(self,head,rel,tail):
        h_emb = self.ent_emb(head)
        r_emb = self.rel_emb(rel)
        t_emb = self.ent_emb(tail)
        s_emb = h_emb + r_emb - t_emb
        return torch.sigmoid(s_emb)
    def forward(self,head,rel,tail):
        """
        head:(batch_size,)
        tail:(batch_size,)
        """
        rg_emb = torch.cat([self.rg_feature[head],self.rg_feature[tail]],dim=1)
        ap_emb = torch.cat([self.ap_feature[head],self.ap_feature[tail]],dim=1)
        fa_emb = torch.cat([self.fa_feature[head],self.fa_feature[tail]],dim=1)
        s_emb = self.transE_score(head,rel,tail)
        o_emb = torch.cat([rg_emb,ap_emb,fa_emb,s_emb],dim=1)
        logits = self.fnn(o_emb)
        return F.softmax(logits)
    def graph_forward(self,x,edge_index,edge_type):
        x_emb = self.ent_emb(x)
        self.fa_feature.data = self.fa_net(x_emb,edge_index)
        self.ap_feature.data = self.appnet(x_emb,edge_index)
        self.rg_feature.data = self.rgcn(x_emb,edge_index,edge_type)
        
