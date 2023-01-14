import os
import time
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch_geometric.data import Data

from config import check_args,get_args,load_config
from src.data import Dictionary
from src.utils import build_graph,load_dataset
from src.model import KGTConv

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    result_dir = os.path.join(args.result_dir,args.dataset)
    # create dataset 
    load_train_file = os.path.join(result_dir,"train2id.txt")
    load_valid_file = os.path.join(result_dir,"valid2id.txt")
    train_set = load_dataset(load_train_file)
    valid_set = load_dataset(load_valid_file)
    # create the graph
    train_loader = DataLoader(train_set,batch_size=args.batch_size,shuffle=True)
    test_loader = DataLoader(valid_set,batch_size=args.batch_size,shuffle=True)
    # load config and create model
    config_file = os.path.join(args.config_dir,args.model_name+".yaml")
    config = load_config(config_file)
    model = KGTConv(config)
    loss = nn.CrossEntropyLoss()
    for epoch in range(args.epoches):
        for item in train_loader:
            head,rel,tail,target = item
            exit()
    #graph = build_graph(result_dir,["train","valid"])
    #print(graph)
    
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
