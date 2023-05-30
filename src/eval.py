import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score,f1_score
from tqdm import tqdm
from .utils import to_var

def evaluate_model(model,loss_fn,data_loader,name,device):
    model.eval()

    all_loss = 0.0
    time_bar = tqdm(enumerate(data_loader),total=len(data_loader),leave = True)
    y_pred = []
    y_true = []
    with torch.no_grad():
        for idx,item in time_bar:
            to_var(item,device)
            head,rel,tail,targets = item
            logits = model(head,rel,tail)
            predicts = torch.argmax(logits,dim=-1)
            loss = loss_fn(logits,targets)
            all_loss += loss.item()
            y_pred.append(predicts)
            y_true.append(targets)
            time_bar.set_description("%s epoch %d"%(name,idx))
    y_pred = torch.cat(y_pred).detach().cpu().numpy()
    y_true = torch.cat(y_true).detach().cpu().numpy()
    all_f1 = f1_score(y_true, y_pred)
    all_emval = accuracy_score(y_true,y_pred)
    all_loss /= len(data_loader)
    data_dict = {
        "f1":np.around(all_f1,decimals=4),
        "acc":np.around(all_emval,decimals=4),
        "loss":np.around(all_loss,decimals=4)
    }
    return data_dict

