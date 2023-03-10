import torch
import torch.nn as nn
import torch.nn.functional as F
from .rgcn import RGCNConv

from .appnp import PRbinaryHop
from .attpr import PRGATConv
from .bpappnp import BbAPPNP
from .ggnn import GatedGraphConv
from .trans import TransA,TransD,TransE,TransH,TransR

class FNNNet(nn.Module):
    def __init__(self,in_dim,hid_dim,out_dim):
        super(FNNNet,self).__init__()
        self.lin = nn.Linear(in_dim,hid_dim)
        self.op = nn.Linear(hid_dim,out_dim)
    def forward(self,inpx):
        hid = F.relu(self.lin(inpx))
        return self.op(hid)
class MixKGATConv(nn.Module):
    def __init__(self,config):
        super(MixKGATConv,self).__init__()
        assert config.model_type in ["pr","prgat","prdrop"]
        assert config.trans_type in ["TransR","TransE","TransH","TransA","TransD"]
        model_dict = {
            "TransE":TransE,
            "TransR":TransR,
            "TransH":TransH,
            "TransA":TransA,
            "TransD":TransD
        }
        
        self.num_ents = config.num_ents
        self.num_rels = config.num_rels
        self.emb_dim = config.emb_dim
        self.trans_type = config.trans_type
        
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
            mlp_dim = (config.emb_dim*2)*2+(config.heads*config.out_dim)*2+config.emb_dim
            #print(mlp_dim,config.emb_dim*2,(config.heads*config.out_dim)*2,config.emb_dim)
        elif config.model_type == "pr":
            self.appnet = PRbinaryHop(config.k_pr,config.pr_alpha,config.pr_beta,config.pr_dropout)
            mlp_dim = (config.emb_dim*2)*3+config.emb_dim
        elif config.model_type == "prdrop":
            self.appnet = BbAPPNP(K=config.k_pr,alpha=config.pr_alpha)
            mlp_dim = config.emb_dim*2*3+config.emb_dim
        else:
            raise ValueError("Error for the model type: %s"%str(config.model_type))
        # This is the route for the GGNN model
        self.ggnn = GatedGraphConv(config.emb_dim,num_layers=config.num_layers)
        # the feaures
        self.rg_feature = nn.Parameter(torch.Tensor(size=(config.num_ents,config.emb_dim)),requires_grad=True)
        self.ap_feature = nn.Parameter(torch.Tensor(size=(config.num_ents,config.emb_dim)),requires_grad=True)
        self.gn_feature = nn.Parameter(torch.Tensor(size=(config.num_ents,config.emb_dim)),requires_grad=True)
        # the mlp layer
        self.fnn = FNNNet(mlp_dim,config.emb_dim,2)
    def forward(self,head,rel,tail):
        """
        head:(batch_size,)
        tail:(batch_size,)
        """
        rg_emb = torch.cat([self.rg_feature[head],self.rg_feature[tail]],dim=1) 
        ap_emb = torch.cat([self.ap_feature[head],self.ap_feature[tail]],dim=1) 
        gn_emb = torch.cat([self.gn_feature[head],self.gn_feature[tail]],dim=1) 
        s_emb = torch.sigmoid(self.transScore(head,rel,tail))
        o_emb = torch.cat([rg_emb,ap_emb,s_emb,gn_emb],dim=1)
        #print(o_emb.shape,rg_emb.shape,ap_emb.shape,s_emb.shape,gn_emb.shape)
        logits = self.fnn(o_emb)
        return F.softmax(logits,dim=-1)
    def graph_forward(self,x,edge_index,edge_type):
        x_emb = self.transScore.ent_emb(x)
        self.ap_feature.data = self.appnet(x_emb,edge_index)
        self.rg_feature.data = self.rgcn(x_emb,edge_index,edge_type)
        self.gn_feature.data = self.ggnn(x_emb,edge_index)

