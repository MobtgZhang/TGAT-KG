from .utils import to_var

def evaluate_model(model,loss_fn,graph,data_loader,device):
    model.eval()
    for item in data_loader:
        to_var(item,device)
        head,rel,tail,target = item
        logits = model(head,rel,tail,graph.edge_index)
        loss = loss_fn(logits,target)
        
