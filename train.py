import os
import time
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader

from config import check_args,get_KGATConv_args,Configuration
from src.data import Dictionary,DataSaver
from src.utils import build_graph,load_dataset,to_var
from src.eval import evaluate_model
from src.kgtconv import KGATConv

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
    # others 
    # test
    load_test__rt_file = os.path.join(result_dir,"test2id__rt.txt")
    load_test_h_t_file = os.path.join(result_dir,"test2id_h_t.txt")
    load_test_hr__file = os.path.join(result_dir,"test2id_hr_.txt")
    # valid
    load_valid__rt_file = os.path.join(result_dir,"valid2id__rt.txt")
    load_valid_h_t_file = os.path.join(result_dir,"valid2id_h_t.txt")
    load_valid_hr__file = os.path.join(result_dir,"valid2id_hr_.txt")
    # load test
    test__rt_set = load_dataset(load_test__rt_file)
    test_h_t_set = load_dataset(load_test_h_t_file)
    test_hr__set = load_dataset(load_test_hr__file)
    # load valid
    valid__rt_set = load_dataset(load_valid__rt_file)
    valid_h_t_set = load_dataset(load_valid_h_t_file)
    valid_hr__set = load_dataset(load_valid_hr__file)
    if "Trans" in args.model_name:
        trans_flag = True
    else:
        trans_flag = False
        graph = build_graph(result_dir,tags_list=["train","valid","test"])
    # create the graph
    train_loader = DataLoader(train_set,batch_size=args.batch_size,shuffle=True)
    valid_loader = DataLoader(valid_set,batch_size=args.batch_size,shuffle=True)
    test_loader = DataLoader(test_set,batch_size=args.batch_size,shuffle=True)
    # other dataloader
    test__rt_loader = DataLoader(test__rt_set,batch_size=args.batch_size,shuffle=True)
    test_h_t_loader = DataLoader(test_h_t_set,batch_size=args.batch_size,shuffle=True)
    test_hr__loader = DataLoader(test_hr__set,batch_size=args.batch_size,shuffle=True)
    valid__rt_loader = DataLoader(valid__rt_set,batch_size=args.batch_size,shuffle=True)
    valid_h_t_loader = DataLoader(valid_h_t_set,batch_size=args.batch_size,shuffle=True)
    valid_hr__loader = DataLoader(valid_hr__set,batch_size=args.batch_size,shuffle=True) 
    # load config and create model
    config = Configuration()
    config.add_attrs("num_ents",len(ent_dict))
    config.add_attrs("num_rels",len(rel_dict))
    for key,value in args._get_kwargs():
        config.add_attrs(key,value)
    logger.info("The model args is %s"%str(config))
    model = KGATConv(config)
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=args.learning_rate)
    save_valid_file = os.path.join(log_dir,args.time_step_str + "-valid.csv")
    save_test_file = os.path.join(log_dir,args.time_step_str + "-test.csv")
    valid_saver = DataSaver(save_valid_file)
    test_saver = DataSaver(save_test_file)
    test__rt_saver = DataSaver(os.path.join(log_dir,args.time_step_str + "-test__rt.csv"))
    test_h_t_saver = DataSaver(os.path.join(log_dir,args.time_step_str + "-test_h_t.csv"))
    test_hr__saver = DataSaver(os.path.join(log_dir,args.time_step_str + "-test_hr_.csv"))
    valid__rt_saver = DataSaver(os.path.join(log_dir,args.time_step_str + "-valid__rt.csv"))
    valid_h_t_saver = DataSaver(os.path.join(log_dir,args.time_step_str + "-valid_h_t.csv"))
    valid_hr__saver = DataSaver(os.path.join(log_dir,args.time_step_str + "-valid_hr_.csv"))
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
        loss_avg /= len(train_loader)
        # the valid results
        valid_saver.start()
        valid_dict = evaluate_model(model,loss_fn,valid_loader,"valid set",device)
        valid_saver.add(valid_dict)
        # the test results
        test_saver.start()
        test_dict = evaluate_model(model,loss_fn,test_loader,"test set",device)
        test_saver.add(test_dict)
        # others
        # test ?,r,t
        test__rt_saver.start()
        test__rt_dict = evaluate_model(model,loss_fn,test__rt_loader,"test__rt set",device)
        test__rt_saver.add(test__rt_dict)
        # test h,?,t
        test_h_t_saver.start()
        test_h_t_dict = evaluate_model(model,loss_fn,test_h_t_loader,"test_h_t set",device)
        test_h_t_saver.add(test_h_t_dict)
        # test h,r,?
        test_hr__saver.start()
        test_hr__dict = evaluate_model(model,loss_fn,test_hr__loader,"test_hr__ set",device)
        test_hr__saver.add(test_hr__dict)
        # valid ?,r,t
        valid__rt_saver.start()
        valid__rt_dict = evaluate_model(model,loss_fn,valid__rt_loader,"valid__rt set",device)
        valid__rt_saver.add(valid__rt_dict)
        # valid h,?,t
        valid_h_t_saver.start()
        valid_h_t_dict = evaluate_model(model,loss_fn,valid_h_t_loader,"valid_h_t set",device)
        valid_h_t_saver.add(valid_h_t_dict)
        # valid h,r,?
        valid_hr__saver.start()
        valid_hr__dict = evaluate_model(model,loss_fn,valid_hr__loader,"valid_hr__ set",device)
        valid_hr__saver.add(valid_hr__dict)
        # print the results to the console
        logger.info("train loss average:%0.4f"%loss_avg)        
        logger.info("test set\tF1-score:%0.4f,accuracy:%0.4f,loss:%0.4f"%(test_dict["f1"],test_dict["acc"],test_dict["loss"]))
        logger.info("valid set\tF1-score:%0.4f,accuracy:%0.4f,loss:%0.4f"%(valid_dict["f1"],valid_dict["acc"],valid_dict["loss"]))
        
if __name__ == "__main__":
    args = get_KGATConv_args()
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

