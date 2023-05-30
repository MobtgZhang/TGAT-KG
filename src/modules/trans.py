import torch
import torch.nn as nn
import torch.nn.functional as F

class TransE(nn.Module):
    # config.num_ents,config.num_rels,config.emb_dim,
    def __init__(self,num_ents,num_rels,emb_dim, margin=1.0, l_regular=2,**kwargs):
        super(TransE,self).__init__()
        self.num_rels = num_ents
        self.num_rels = num_rels
        self.emb_dim = emb_dim
        self.margin = margin
        self.l_regular = l_regular
        self.ent_emb = nn.Embedding(num_ents,emb_dim)
        self.rel_emb = nn.Embedding(num_rels,emb_dim)
        self.out_dim = emb_dim
    def forward(self,head, relation, tail):
        head = self.ent_emb(head)
        tail = self.ent_emb(tail)
        relation = torch.squeeze(self.rel_emb(relation), dim=1)
        score = head+relation-tail
        return score

class TransD(nn.Module):
    def __init__(self,num_ents,num_rels,ent_dim,rel_dim=None,margin=1.0,**kwargs):
        super(TransD,self).__init__()
        self.num_ents = num_ents
        self.ent_dim = ent_dim
        self.num_rels = num_rels
        rel_dim = rel_dim if rel_dim is not None else ent_dim
        self.rel_dim = rel_dim
        self.margin = margin

        self.ent_emb = nn.Embedding(num_ents,ent_dim)
        self.ent_map_emb = nn.Embedding(num_ents,ent_dim)
        self.rel_emb = nn.Embedding(num_rels,rel_dim)
        self.rel_map_emb = nn.Embedding(num_rels,rel_dim)
        self.out_dim = rel_dim
    def forward(self,head, relation, tail):
        headp = self.ent_map_emb(head) # (B, En)
        head = self.ent_emb(head) # (B, En)
        tailp = self.ent_map_emb(tail) # (B, En)
        tail = self.ent_emb(tail) # (B, En)
        relationp = self.rel_map_emb(relation) # (B, Rn)
        relation = self.rel_emb(relation) # (B, Rn)

        relationp = torch.unsqueeze(relationp, dim=2)   # (B, Em, 1)
        headp = torch.unsqueeze(headp, dim=1)           # (B, 1, En)
        tailp = torch.unsqueeze(tailp, dim=1)           # (B, 1, En)
        I_mat = torch.eye(self.rel_dim,self.ent_dim).to(head.device) # (Er,Et)
        Mrh = torch.matmul(relationp,headp) + I_mat 
        Mrt = torch.matmul(relationp,tailp) + I_mat
        head = torch.unsqueeze(head, dim=2)
        tail = torch.unsqueeze(tail, dim=2)
        
        head = torch.squeeze(torch.matmul(Mrh, head), dim=2)   # (B, Em, 1) -> (B, Em)
        tail = torch.squeeze(torch.matmul(Mrt, tail), dim=2)   # (B, Em, 1) -> (B, Em)
        score = head+relation-tail
        return score
class TransR(nn.Module):
    def __init__(self,num_ents,num_rels,emb_dim,margin=1.0,l_regular=2,**kwargs):
        super(TransR, self).__init__()
        assert l_regular in [1,2]
        self.num_ents = num_ents
        self.num_rels = num_rels
        self.emb_dim = emb_dim
        self.margin = margin
        self.l_regular = l_regular
        self.ent_emb = nn.Embedding(num_ents,emb_dim)
        self.rel_emb = nn.Embedding(num_rels,emb_dim)
        self.transfer = nn.Parameter(torch.rand(size=(emb_dim,emb_dim)),requires_grad=True)
        self.distfn = nn.PairwiseDistance(l_regular)
        self.out_dim = emb_dim
    def forward(self,head, relation, tail):
        # Step2
        head_m = self.ent_emb(head)
        rel_m = self.rel_emb(relation)
        tail_m = self.ent_emb(tail)
        head_m = torch.matmul(head_m,self.transfer)
        tail_m = torch.matmul(tail_m,self.transfer)
        # Step3 and Step4
        score = head_m+rel_m-tail_m
        return score
    
class TransH(nn.Module):
    def __init__(self,num_ents,num_rels,emb_dim,margin=1.0,l_regular=2,v_c=1.0,eps=0.001,**kwargs):
        super(TransH,self).__init__()
        self.num_ents = num_ents
        self.num_rels = num_rels
        self.margin = margin
        self.l_regular = l_regular
        self.v_c = v_c
        self.eps = eps

        self.ent_emb = nn.Embedding(num_ents,emb_dim)
        self.rel_emb = nn.Embedding(num_rels,emb_dim)
        self.rel_hyper = nn.Embedding(num_rels,emb_dim)
        self.out_dim = emb_dim
    def forward(self,head, relation, tail):
        # Step2
        head_emb = self.ent_emb(head)
        rel_hyper = self.rel_hyper(relation)
        rel_emb = self.rel_emb(relation)
        tail_emb = self.ent_emb(tail)
        # Step3
        head_emb = head_emb - rel_hyper * torch.sum(head_emb * rel_hyper, dim=1, keepdim=True)
        tail_emb = tail_emb - rel_hyper * torch.sum(tail_emb * rel_hyper, dim=1, keepdim=True)
        # Step4
        return head_emb+rel_emb-tail_emb
