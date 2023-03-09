import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Relation(nn.Module):
    def __init__(self, in_features, out_features, ablation=0):
        super(Relation, self).__init__()
        
        self.gamma1_1 = nn.Linear(in_features, out_features, bias=False)
        self.gamma1_2 = nn.Linear(out_features, in_features, bias=False)

        self.gamma2_1 = nn.Linear(in_features, out_features, bias=False)
        self.gamma2_2 = nn.Linear(out_features, in_features, bias=False)

        self.beta1_1 = nn.Linear(in_features, out_features, bias=False)
        self.beta1_2 = nn.Linear(out_features, in_features, bias=False)

        self.beta2_1 = nn.Linear(in_features, out_features, bias=False)
        self.beta2_2 = nn.Linear(out_features, in_features, bias=False)

        self.r = nn.Parameter(torch.FloatTensor(1, in_features))

        self.ablation = ablation
        self.elu = nn.ELU()
        self.lrelu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()
        self.reset_parameter()
    def reset_parameter(self):
        stdv = 1. / math.sqrt(self.r.size(1))
        self.r.data.uniform_(-stdv, stdv)
    def forward(self, ft, neighbor):

        if self.ablation == 3:
            self.m = ft + self.r - neighbor
        else:

            gamma1 = self.gamma1_2(self.gamma1_1(ft))
            gamma2 = self.gamma2_2(self.gamma2_1(neighbor))
            gamma = self.lrelu(gamma1 + gamma2) + 1.0 

            beta1 = self.beta1_2(self.beta1_1(ft)) 
            beta2 = self.beta2_2(self.beta2_1(neighbor))
            beta = self.lrelu(beta1 + beta2) 

            self.r_v = gamma * self.r + beta
            self.m = ft + self.r_v - neighbor
            
        return F.normalize(self.m)

class Generator(nn.Module):
    def __init__(self, in_features, std, ablation):
        super(Generator, self).__init__()

        self.g = nn.Linear(in_features, in_features, bias=True)
        self.std = std
        self.ablation = ablation

    def forward(self, ft):
        #h_s = ft
        if self.training:
            if self.ablation == 2:
                mean = torch.zeros(ft.shape, device='cuda')
                ft = torch.normal(mean, 1.)
            else:
                ft = torch.normal(ft, self.std)
        h_s = F.elu(self.g(ft)) 
        
        return h_s

class TransSAGE(nn.Module):
    def __init__(self, nfeat, nhid, g_sigma,device,ablation=0, nheads=3, dropout=0.5, concat=True):
        super(TransSAGE, self).__init__()

        self.device = device
        self.ablation = ablation
        self.r = Relation(nfeat, ablation)
        self.g = Generator(nfeat, g_sigma, ablation)       
        self.weight = nn.Linear(nfeat, nhid, bias=False)

    def forward(self, x, adj, head):
        
        mean = F.normalize(adj, p=1, dim=1)
        neighbor = torch.mm(mean,x)
        output = self.r(x, neighbor)

        if head or self.ablation == 2:            
            ft_input = self.weight(x)
            ft_neighbor = self.weight(neighbor)
            h_k = torch.cat([ft_input, ft_neighbor], dim=1)

        else:
            if self.ablation == 1:
                h_s = self.g(output)
            else:
                h_s = output
            
            norm = torch.sum(adj, 1, keepdim=True) + 1
            neighbor = neighbor + h_s / norm
            ft_input = self.weight(x)
            ft_neighbor = self.weight(neighbor)
            h_k = torch.cat([ft_input, ft_neighbor], dim=1)

        return h_k, output 
