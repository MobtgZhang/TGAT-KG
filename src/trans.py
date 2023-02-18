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
    def forward(self,in_triple):
        head, relation, tail = torch.chunk(input=in_triple,chunks=3,dim=1)
        head = torch.squeeze(self.ent_emb(head), dim=1)
        tail = torch.squeeze(self.ent_emb(tail), dim=1)
        relation = torch.squeeze(self.rel_emb(relation), dim=1)
        score = head+relation-tail
        return score

class TransD(nn.Module):
    def __init__(self,ent_size,rel_size,rel_dim,ent_dim,margin=1.0):
        super(TransD,self).__init__()
        self.ent_size = ent_size
        self.ent_dim = ent_dim
        self.rel_size = rel_size
        self.rel_dim = rel_dim
        self.margin = margin

        self.ent_emb = nn.Embedding(ent_size,ent_dim)
        self.ent_map_emb = nn.Embedding(ent_size,ent_dim)
        self.rel_emb = nn.Embedding(rel_size,rel_dim)
        self.rel_map_emb = nn.Embedding(rel_size,rel_dim)
    def forward(self,in_triple):
        head, relation, tail = torch.chunk(in_triple,chunks=3,dim=1)
        headp = torch.squeeze(self.ent_map_emb(head), dim=1)   # (B, 1, En) -> (B, En)
        head = torch.squeeze(self.ent_emb(head), dim=1)       # (B, 1, En) -> (B, En)
        tailp = torch.squeeze(self.ent_map_emb(tail), dim=1)   # (B, 1, En) -> (B, En)
        tail = torch.squeeze(self.ent_emb(tail), dim=1)       # (B, 1, En)  -> (B, En)
        relationp = torch.squeeze(self.rel_map_emb(relation), dim=1) # (B, 1, Em) -> (B, Em)
        relation = torch.squeeze(self.rel_emb(relation), dim=1)     # (B, 1, Em) -> (B, Em)

        relationp = torch.unsqueeze(relationp, dim=2)   # (B, Em, 1)
        headp = torch.unsqueeze(headp, dim=1)           # (B, 1, En)
        tailp = torch.unsqueeze(tailp, dim=1)           # (B, 1, En)

        I_mat = torch.eye(self.rel_dim,self.ent_dim).to(in_triple.device)
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
    def forward(self,in_triple):
        # Step1
        head,relation,tail = torch.chunk(input=in_triple,chunks=3,dim=1)
        # Step2
        head_m = torch.squeeze(self.ent_emb(head),dim=1)
        rel_m = torch.squeeze(self.rel_emb(relation),dim=1)
        tail_m = torch.squeeze(self.ent_emb(tail),dim=1)
        head_m = torch.matmul(head_m,self.transfer)
        tail_m = torch.matmul(tail_m,self.transfer)
        # Step3 and Step4
        score = head_m+rel_m-tail_m
        return score

class TransA(nn.Module):
    def __init__(self,ent_size,rel_size,emb_dim,margin=1.0):
        super(TransA,self).__init__()
        self.ent_size = ent_size
        self.rel_size = rel_size
        self.emb_dim = emb_dim
        self.margin = margin
        
        self.ent_emb = nn.Embedding(ent_size,emb_dim)
        self.rel_emb = nn.Embedding(rel_size,emb_dim)
        self.rel_w = nn.Parameter(torch.zeros(size=(rel_size,emb_dim,emb_dim)),requires_grad=True)
    def calculate_wr(self,pos_x,neg_x):
        pos_head, pos_rel, pos_tail = torch.chunk(input=pos_x,chunks=3,dim=1)
        neg_head, neg_rel, neg_tail = torch.chunk(input=neg_x,chunks=3,dim=1)
        pos_headm, pos_relm, pos_tailm = self.ent_emb(pos_head), self.rel_emb(pos_rel),self.ent_emb(pos_tail)
        neg_headm, neg_relm, neg_tailm = self.ent_emb(neg_head),self.rel_emb(neg_rel),self.ent_emb(neg_tail)
        error_pos = torch.abs(pos_headm + pos_relm - pos_tailm)
        error_neg = torch.abs(neg_headm + neg_relm - neg_tailm)
        del pos_headm, pos_relm, pos_tailm, neg_headm, neg_relm, neg_tailm
        self.rel_w[pos_rel] += torch.sum(torch.matmul(error_neg.permute((0, 2, 1)), error_neg), dim=0) - \
                           torch.sum(torch.matmul(error_pos.permute((0, 2, 1)), error_pos), dim=0)
    def forward(self,in_triple):
        head, relation, tail = torch.chunk(input=in_triple,chunks=3,dim=1)        
        head_emb = torch.squeeze(self.ent_emb(head), dim=1) 
        rel_emb = torch.squeeze(self.rel_emb(relation), dim=1)
        tail_emb = torch.squeeze(self.ent_emb(tail), dim=1)

        rel_wr = self.rel_w[relation]
        # (B, E) -> (B, 1, E) * (B, E, E) * (B, E, 1) -> (B, 1, 1) -> (B, )
        error = torch.unsqueeze(torch.abs(head_emb+rel_emb-tail_emb), dim=1)
        error = torch.matmul(torch.matmul(error, torch.unsqueeze(rel_wr, dim=0)), error.permute((0, 2, 1)))
        return torch.squeeze(error)

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
    def forward(self,in_triple):
        # Step1
        head, relation, tail = torch.chunk(in_triple,chunks=3,dim=1)
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

class KG2E(nn.Module):
    def __init__(self, ent_size, rel_size, emb_dim, margin=1.0, sim="KL", vmin=0.03, vmax=3.0):
        super(KG2E, self).__init__()
        assert (sim in ["KL", "EL"])
        self.margin = margin
        self.sim = sim
        self.ent_size = ent_size
        self.rel_size = rel_size
        self.emb_dim = emb_dim
        self.vmin = vmin
        self.vmax = vmax

        # Embeddings represent the mean vector of entity and relation
        # Covars represent the covariance vector of entity and relation
        self.ent_emb = nn.Embedding(ent_size,emb_dim)
        self.ent_covar = nn.Embedding(ent_size,emb_dim)
        self.rel_emb = nn.Embedding(rel_size,emb_dim)
        self.rel_covar = nn.Embedding(rel_size,emb_dim)

    def KL_score(self,errorv,errorm,relationv,relationm):
        # Calculate KL(e, r)
        losep1 = torch.sum(errorv/relationv, dim=1)
        losep2 = torch.sum((relationm-errorm)**2 /relationv, dim=1)
        KLer = (losep1 + losep2 - self.emb_dim) / 2

        # Calculate KL(r, e)
        losep1 = torch.sum(relationv/errorv, dim=1)
        losep2 = torch.sum((errorm - relationm) ** 2 / errorv, dim=1)
        KLre = (losep1 + losep2 - self.emb_dim) / 2
        return (KLer + KLre) / 2
    def EL_score(self,errorv,errorm,relationv,relationm):
        losep1 = torch.sum((errorm - relationm) ** 2 / (errorv + relationv), dim=1)
        losep2 = torch.sum(torch.log(errorv+relationv), dim=1)
        return (losep1 + losep2) / 2
    def forward(self,in_triple):
        head, relation, tail = torch.chunk(input=in_triple,chunks=3,dim=1)

        headm = torch.squeeze(self.ent_emb(head), dim=1)
        headv = torch.squeeze(self.ent_covar(head), dim=1)
        tailm = torch.squeeze(self.ent_emb(tail), dim=1)
        tailv = torch.squeeze(self.ent_covar(tail), dim=1)
        relationm = torch.squeeze(self.rel_emb(relation), dim=1)
        relationv = torch.squeeze(self.rel_covar(relation), dim=1)
        errorm = tailm - headm
        errorv = tailv + headv
        if self.sim == "KL":
            return self.KL_score(relationm=relationm, relationv=relationv, errorm=errorm, errorv=errorv)
        elif self.sim == "EL":
            return self.EL_score(relationm=relationm, relationv=relationv, errorm=errorm, errorv=errorv)

