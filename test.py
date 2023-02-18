import torch
from src.trans import TransA,TransD,TransE,TransH,TransR,KG2E
def main():
    batch_size = 16
    ent_num = 500
    ent_dim = 50
    rel_dim = 30
    rel_num = 54
    h = torch.randint(0,ent_num,size=(batch_size,1))
    r = torch.randint(0,rel_num,size=(batch_size,1))
    t = torch.randint(0,ent_num,size=(batch_size,1))
    in_triple = torch.cat([h,r,t],dim=1)
    model_d = TransD(ent_num,rel_num,rel_dim=rel_dim,ent_dim=ent_dim)
    model_e = TransE(ent_num,rel_num,emb_dim=ent_dim)
    model_h = TransH(ent_num,rel_num,ent_dim)
    model_r = TransR(ent_num,rel_num,ent_dim)
    model_kl = KG2E(ent_num,rel_num,ent_dim,sim="KL")
    model_el = KG2E(ent_num,rel_num,ent_dim,sim="EL")
    outs_e = model_e(in_triple)
    outs_d = model_d(in_triple)
    outs_h = model_h(in_triple)
    outs_r = model_r(in_triple)
    outs_kl = model_kl(in_triple)
    outs_el = model_el(in_triple)
    print(outs_e.shape,outs_d.shape,outs_h.shape,outs_r.shape,outs_kl.shape,outs_el.shape)
if __name__ == "__main__":
    main()

