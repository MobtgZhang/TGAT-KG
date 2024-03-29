import os
import time
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader

from config import check_args,get_args,load_config,get_model
from src.data import Dictionary,DataSaver
from src.utils import build_graph,load_dataset,to_var
from src.eval import evaluate_model

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info("The parser args is %s"%str(args))
    result_dir = os.path.join(args.result_dir,args.dataset)
    log_dir = os.path.join(args.log_dir,args.dataset,args.model_name)
    # create dataset 
    load_train_file = os.path.join(result_dir,"train2id.txt")
    load_valid_file = os.path.join(result_dir,"valid2id.txt")
    load_test_file = os.path.join(result_dir,"test2id.txt")
    ent_filename = os.path.join(result_dir,"entities.txt")
    rel_filename = os.path.join(result_dir,"relations.txt")
    ent_dict = Dictionary.load(ent_filename)
    rel_dict = Dictionary.load(rel_filename)
    train_set = load_dataset(load_train_file)
    valid_set = load_dataset(load_valid_file)
    test_set = load_dataset(load_test_file)
    if "Trans" in args.model_name:
        trans_flag = True
    else:
        trans_flag = False
        graph = build_graph(result_dir,tags_list=["train","valid"])
    # create the graph
    train_loader = DataLoader(train_set,batch_size=args.batch_size,shuffle=True)
    valid_loader = DataLoader(valid_set,batch_size=args.batch_size,shuffle=True)
    test_loader = DataLoader(test_set,batch_size=args.batch_size,shuffle=True)
    # load config and create model
    config_file = os.path.join(args.config_dir,args.model_name+".yaml")
    config = load_config(config_file)
    logger.info("The model args is %s"%str(config))
    config.add_attrs("num_ents",len(ent_dict))
    config.add_attrs("num_rels",len(rel_dict))
    config.add_attrs("model_name",args.model_name)
    config.add_attrs("sigmoid",args.sigmoid)
    model = get_model(config)
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)
    save_valid_file = os.path.join(log_dir,args.time_step_str + "-valid.csv")
    save_test_file = os.path.join(log_dir,args.time_step_str + "-test.csv")
    valid_saver = DataSaver(save_valid_file)
    test_saver = DataSaver(save_test_file)
    for epoch in tqdm(range(args.epoches),desc="training %s dataset and %s model"%(args.dataset,args.model_name)):
        model.train()
        if not trans_flag:
            graph.edge_index = graph.edge_index.to(device)
            graph.x = graph.x.to(device)
            graph.edge_type = graph.edge_type.to(device)
            model.graph_forward(graph.x,graph.edge_index,graph.edge_type)
        loss_avg = 0.0
        for item in train_loader:
            optimizer.zero_grad()
            to_var(item,device)
            head,rel,tail,target = item
            logits = model(head,rel,tail)
            loss = loss_fn(logits,target)
            loss_avg += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=args.max_norm)
            optimizer.step()
        scheduler.step()
        loss_avg /= len(train_loader)
        # the valid results
        valid_saver.start()
        valid_dict = evaluate_model(model,loss_fn,valid_loader,"valid set",device)
        valid_saver.add(valid_dict)
        # the test results
        test_saver.start()
        test_dict = evaluate_model(model,loss_fn,test_loader,"test set",device)
        test_saver.add(test_dict)
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        # print the results to the console
        logger.info("train loss average:%0.4f"%loss_avg)        
        logger.info("test set\tF1-score:%0.4f,accuracy:%0.4f,loss:%0.4f,lr:%0.4f"%(test_dict["f1"],test_dict["acc"],test_dict["loss"],lr))
        logger.info("valid set\tF1-score:%0.4f,accuracy:%0.4f,loss:%0.4f,lr:%0.4f"%(valid_dict["f1"],valid_dict["acc"],valid_dict["loss"],lr))
        
if __name__ == "__main__":
    args = get_args()
    check_args(args)
    # First step, create a logger
    logger = logging.getLogger()
    # The log level switch
    logger.setLevel(logging.INFO)
    # Second step, create a handler,which is used to write to logging file.
    args.time_step_str = time.strftime("%Y%m%d%H%M",time.localtime(time.time()))
    log_dir = os.path.join(args.log_dir,args.dataset,args.model_name)
    log_file = os.path.join(log_dir,args.time_step_str+".log")
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
