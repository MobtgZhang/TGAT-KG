import os
import time
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.optim as optim

from torch.utils.data import DataLoader
from torch_geometric.data import Data

from config import check_args,get_args,load_config
from src.data import Dictionary
from src.utils import build_graph,load_dataset,to_var
from src.model import KGTConv

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    result_dir = os.path.join(args.result_dir,args.dataset)
    # create dataset 
    load_train_file = os.path.join(result_dir,"train2id.txt")
    load_valid_file = os.path.join(result_dir,"valid2id.txt")
    ent_filename = os.path.join(result_dir,"entities.txt")
    rel_filename = os.path.join(result_dir,"relations.txt")
    ent_dict = Dictionary.load(ent_filename)
    rel_dict = Dictionary.load(rel_filename)
    train_set = load_dataset(load_train_file)
    valid_set = load_dataset(load_valid_file)
    graph = build_graph(result_dir,tags_list=["train","valid"])
    # create the graph
    train_loader = DataLoader(train_set,batch_size=args.batch_size,shuffle=True)
    test_loader = DataLoader(valid_set,batch_size=args.batch_size,shuffle=True)
    # load config and create model
    config_file = os.path.join(args.config_dir,args.model_name+".yaml")
    config = load_config(config_file)
    config.num_ents = len(ent_dict)
    config.num_rels = len(rel_dict)
    model = KGTConv(config).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=args.learning_rate)
    for epoch in range(args.epoches):
        model.train()
        graph.edge_index = graph.edge_index.to(device)
        graph.x = graph.x.to(device)
        model.graph_forward(graph.x,graph.edge_index)
        for item in train_loader:
            optimizer.zero_ragd()
            to_var(item,device)
            head,rel,tail,target = item
            logits = model(head,rel,tail,graph.edge_index)
            loss = loss_fn(logits,target)
            loss.backward()
            optimizer.step()
        evaluate_model(model)

if __name__ == "__main__":
    args = get_args()
    check_args(args)
    # First step, create a logger
    logger = logging.getLogger()
    # The log level switch
    logger.setLevel(logging.INFO)
    # Second step, create a handler,which is used to write to logging file.
    args.time_step_str = time.strftime("%Y%m%d%H%M",time.localtime(time.time()))
    log_file = os.path.join(args.log_dir,args.time_step_str+".log")
    # Third, define the output formatter
    format_str = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh = logging.FileHandler(log_file, mode='w')
    # the log's switch of the output log file
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(format_str)
    ch = logging.StreamHandler()
    ch.setFormatter(format_str)
    # Fourth, add the logger into handler
    logger.addHandler(fh)
    logger.addHandler(ch)
    main(args)
