import torch
import torch.nn as nn

class TransE(nn.Module):
    def __init__(self,ent_size,rel_size,emb_dim, margin=1.0, l_regular=2):
        super(TransE,self).__init__()
        self.ent_size = ent_size
        self.rel_size = rel_size
        self.emb_dim = emb_dim
        self.margin = margin
        self.l_regular = l_regular
        self.ent_emb = nn.Embedding(ent_size,emb_dim)
        self.rel_emb = nn.Embedding(rel_size,emb_dim)
    def forward(self,head, relation, tail):
        head = torch.squeeze(self.ent_emb(head), dim=1)
        tail = torch.squeeze(self.ent_emb(tail), dim=1)
        relation = torch.squeeze(self.rel_emb(relation), dim=1)
        score = head+relation-tail
        return score

class TransD(nn.Module):
    def __init__(self,ent_size,rel_size,ent_dim,rel_dim=None,margin=1.0):
        super(TransD,self).__init__()
        self.ent_size = ent_size
        self.ent_dim = ent_dim
        self.rel_size = rel_size
        rel_dim = rel_dim if rel_dim is not None else ent_dim
        self.rel_dim = rel_dim
        self.margin = margin

        self.ent_emb = nn.Embedding(ent_size,ent_dim)
        self.ent_map_emb = nn.Embedding(ent_size,ent_dim)
        self.rel_emb = nn.Embedding(rel_size,rel_dim)
        self.rel_map_emb = nn.Embedding(rel_size,rel_dim)
    def forward(self,head, relation, tail):
        headp = torch.squeeze(self.ent_map_emb(head), dim=1)   # (B, 1, En) -> (B, En)
        head = torch.squeeze(self.ent_emb(head), dim=1)       # (B, 1, En) -> (B, En)
        tailp = torch.squeeze(self.ent_map_emb(tail), dim=1)   # (B, 1, En) -> (B, En)
        tail = torch.squeeze(self.ent_emb(tail), dim=1)       # (B, 1, En)  -> (B, En)
        relationp = torch.squeeze(self.rel_map_emb(relation), dim=1) # (B, 1, Em) -> (B, Em)
        relation = torch.squeeze(self.rel_emb(relation), dim=1)     # (B, 1, Em) -> (B, Em)

        relationp = torch.unsqueeze(relationp, dim=2)   # (B, Em, 1)
        headp = torch.unsqueeze(headp, dim=1)           # (B, 1, En)
        tailp = torch.unsqueeze(tailp, dim=1)           # (B, 1, En)

        I_mat = torch.eye(self.rel_dim,self.ent_dim).to(head.device)
        Mrh = torch.matmul(relationp,headp) + I_mat
        Mrt = torch.matmul(relationp,tailp) + I_mat

        head = torch.unsqueeze(head, dim=2)
        tail = torch.unsqueeze(tail, dim=2)
        head = torch.squeeze(torch.matmul(Mrh, head), dim=2)   # (B, Em, 1) -> (B, Em)
        tail = torch.squeeze(torch.matmul(Mrt, tail), dim=2)   # (B, Em, 1) -> (B, Em)
        score = head+relation-tail
        return score
class TransR(nn.Module):
    def __init__(self,ent_size,rel_size,emb_dim,margin=1.0,l_regular=2):
        super(TransR, self).__init__()
        assert l_regular in [1,2]
        self.ent_size = ent_size
        self.rel_size = rel_size
        self.emb_dim = emb_dim
        self.margin = margin
        self.l_regular = l_regular
        self.ent_emb = nn.Embedding(ent_size,emb_dim)
        self.rel_emb = nn.Embedding(rel_size,emb_dim)
        self.transfer = nn.Parameter(torch.rand(size=(emb_dim,emb_dim)),requires_grad=True)
        self.distfn = nn.PairwiseDistance(l_regular)
    def forward(self,head, relation, tail):
        # Step2
        head_m = torch.squeeze(self.ent_emb(head),dim=1)
        rel_m = torch.squeeze(self.rel_emb(relation),dim=1)
        tail_m = torch.squeeze(self.ent_emb(tail),dim=1)
        head_m = torch.matmul(head_m,self.transfer)
        tail_m = torch.matmul(tail_m,self.transfer)
        # Step3 and Step4
        score = head_m+rel_m-tail_m
        return score
    
class TransH(nn.Module):
    def __init__(self,ent_size,rel_size,emb_dim,margin=1.0,l_regular=2,v_c=1.0,eps=0.001):
        super(TransH,self).__init__()
        self.ent_size = ent_size
        self.rel_size = rel_size
        self.margin = margin
        self.l_regular = l_regular
        self.v_c = v_c
        self.eps = eps

        self.ent_emb = nn.Embedding(ent_size,emb_dim)
        self.rel_emb = nn.Embedding(rel_size,emb_dim)
        self.rel_hyper = nn.Embedding(rel_size,emb_dim)
    def forward(self,head, relation, tail):
        # Step2
        head_emb = torch.squeeze(self.ent_emb(head), dim=1)
        rel_hyper = torch.squeeze(self.rel_hyper(relation), dim=1)
        rel_emb = torch.squeeze(self.rel_emb(relation), dim=1)
        tail_emb = torch.squeeze(self.ent_emb(tail), dim=1)
        # Step3
        head_emb = head_emb - rel_hyper * torch.sum(head_emb * rel_hyper, dim=1, keepdim=True)
        tail_emb = tail_emb - rel_hyper * torch.sum(tail_emb * rel_hyper, dim=1, keepdim=True)
        # Step4
        return head_emb+rel_emb-tail_emb
