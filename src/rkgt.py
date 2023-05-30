import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import RGCNConv

from .modules import PRbinaryHop,PRGATConv,BbAPPNP,GatedGraphConv
from .modules import TransD,TransE,TransH,TransR

class FNNNet(nn.Module):
    def __init__(self,in_dim,hid_dim,out_dim):
        super(FNNNet,self).__init__()
        self.lin = nn.Linear(in_dim,hid_dim)
        self.op = nn.Linear(hid_dim,out_dim)
    def forward(self,inpx):
        hid = F.relu(self.lin(inpx))
        return self.op(hid)
class FuseUnit(nn.Module):
    def __init__(self,in_dim,hid_dim,out_dim):
        super(FuseUnit,self).__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.fn1 = nn.Linear(hid_dim,hid_dim)
        self.fn2 = nn.Linear(in_dim,hid_dim)
        self.fn3 = nn.Linear(hid_dim,out_dim)
    def forward(self,combin_emb,s_emb):
        # inputs = torch.cat([rg_emb,ap_emb,gn_emb],dim=-1)
        combine_emb = self.fn2(combin_emb)
        trans_emb = self.fn1(s_emb)
        query = combine_emb.unsqueeze(-1)
        key = trans_emb.unsqueeze(1)
        value = s_emb.unsqueeze(-1)
        mat_kq = 1.0/math.sqrt(self.hid_dim)*torch.bmm(query,key)
        out = F.relu(torch.matmul(mat_kq,value).squeeze())
        out = self.fn3(out)
        return out
class FuseUnit2(nn.Module):
    def __init__(self,in_dim,hid_dim,out_dim):
        super(FuseUnit2,self).__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.fn1 = nn.Linear(hid_dim,hid_dim)
        self.fn2 = nn.Linear(in_dim,hid_dim)
        self.fn3 = nn.Linear(hid_dim,out_dim)
    def forward(self,combin_emb,s_emb):
        c_emb = self.fn2(combin_emb)
        t_emb = self.fn1(s_emb)
        
        query = c_emb.unsqueeze(-1)
        key = t_emb.unsqueeze(1)
        value = s_emb.unsqueeze(-1)
        mat_kq = 1.0/math.sqrt(self.hid_dim)*torch.bmm(query,key)
        mat_kq = F.softmax(mat_kq,dim=-1)
        out = F.relu(torch.matmul(mat_kq,value).squeeze())
        out = self.fn3(out)
        return out
class RKGTConv(nn.Module):
    def __init__(self,config):
        super(RKGTConv,self).__init__()
        assert config.model_type in ["pr","prgat","prdrop"]
        # assert config.trans_type in ["TransR","TransE","TransH","TransA","TransD"]
        assert config.method_type in ["fnn","att","fnn2"]
        model_dict = {
            "TransE":TransE,
            "TransR":TransR,
            "TransH":TransH,
            "TransD":TransD
        }
        
        self.num_ents = config.num_ents
        self.num_rels = config.num_rels
        self.emb_dim = config.emb_dim
        self.trans_type = config.trans_type
        self.method_type = config.method_type
        self.sigmoid = config.sigmoid
        
        self.k_pr = config.k_pr
        self.pr_alpha = config.pr_alpha
        self.pr_beta = config.pr_beta
        self.pr_dropout = config.pr_dropout
        self.transScore = model_dict[config.trans_type](config.num_ents,config.num_rels,config.emb_dim)
        # This is used for RGCN model to evaluate the relation and head,tail.
        self.rgcn = RGCNConv(config.emb_dim, config.emb_dim,config.num_rels,config.num_bases)
        # This is used for PRGATConv model to evaluate the head and tail.
        if config.model_type == "prgat":
            self.appnet = PRGATConv(in_channels=config.emb_dim,out_channels=config.out_dim,heads=config.heads,
                                alpha=config.pr_alpha,beta=config.pr_beta,dropout=config.pr_dropout,k_loops=config.k_pr)
            ap_dim = config.heads*config.out_dim
            mlp_dim = (config.emb_dim*2)*2+ap_dim*2+config.emb_dim
            #print(mlp_dim,config.emb_dim*2,(config.heads*config.out_dim)*2,config.emb_dim)
        elif config.model_type == "pr":
            self.appnet = PRbinaryHop(config.k_pr,config.pr_alpha,config.pr_beta,config.pr_dropout)
            ap_dim = config.emb_dim*2
            mlp_dim = (config.emb_dim*2)*2+ap_dim+config.emb_dim
        elif config.model_type == "prdrop":
            self.appnet = BbAPPNP(K=config.k_pr,alpha=config.pr_alpha)
            ap_dim = config.emb_dim*2
            mlp_dim = config.emb_dim*2*2+ap_dim+config.emb_dim
        else:
            raise ValueError("Error for the model type: %s"%str(config.model_type))
        
        # This is the route for the GGNN model
        self.ggnn = GatedGraphConv(config.emb_dim,num_layers=config.num_layers)
        # the feaures
        self.rg_feature = nn.Parameter(torch.Tensor(size=(config.num_ents,config.emb_dim)),requires_grad=True)
        self.ap_feature = nn.Parameter(torch.Tensor(size=(config.num_ents,config.emb_dim)),requires_grad=True)
        self.gn_feature = nn.Parameter(torch.Tensor(size=(config.num_ents,config.emb_dim)),requires_grad=True)
        self.rel_feature = nn.Parameter(torch.Tensor(size=(config.num_rels,config.emb_dim)),requires_grad=True)
        # the mlp layer
        if config.method_type == "fnn":
            self.fnn = FNNNet(mlp_dim,config.emb_dim,2)
        elif config.method_type == "att1":
            self.fnn = FuseUnit2(mlp_dim-config.emb_dim,config.emb_dim,2)
        elif config.method_type == "att":
            self.fnn = FuseUnit(mlp_dim-config.emb_dim,config.emb_dim,2)
        
    def forward(self,head,rel,tail):
        """
        head:(batch_size,)
        tail:(batch_size,)
        """
        rg_emb = torch.cat([self.rg_feature[head],self.rg_feature[tail]],dim=1) 
        ap_emb = torch.cat([self.ap_feature[head],self.ap_feature[tail]],dim=1)
        gn_emb = torch.cat([self.gn_feature[head],self.gn_feature[tail]],dim=1) 
        s_emb = torch.sigmoid(self.transScore(head,rel,tail))
        
        if self.method_type == "fnn":
            print(rg_emb.shape,ap_emb.shape,gn_emb.shape,s_emb.shape)
            exit()
            o_emb = torch.cat([rg_emb,ap_emb,gn_emb,s_emb],dim=1)
            logits = self.fnn(o_emb)
        elif self.method_type == "att1":
            c_emb = torch.cat([rg_emb,ap_emb,gn_emb],dim=-1)
            logits = self.fnn(c_emb,s_emb)
        elif self.method_type == "att":
            c_emb = torch.cat([rg_emb,ap_emb,gn_emb],dim=-1)
            logits = self.fnn(c_emb,s_emb)
        # print(out.shape)
        # exit()
        # o_emb = torch.cat([rg_emb,ap_emb,gn_emb,s_emb],dim=1)
        # logits = self.fnn(o_emb)
        if self.sigmoid:
            return torch.sigmoid(logits)
        else:
            return F.softmax(logits,dim=-1)
    def graph_forward(self,x,edge_index,edge_type):
        x_emb = self.transScore.ent_emb(x)
        self.ap_feature.data = self.appnet(x_emb,edge_index)
        self.rg_feature.data = self.rgcn(x_emb,edge_index,edge_type)
        self.gn_feature.data = self.ggnn(x_emb,edge_index)

