import os
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir",default="./result",type=str)
    parser.add_argument("--log-dir",default="./log",type=str)
    parser.add_argument("--data-dir",default="./data",type=str)
    parser.add_argument("--config-dir",default="./config",type=str)
    parser.add_argument("--batch-size",default=20,type=int)
    parser.add_argument("--dataset",default="fb15k",type=str)
    args = parser.parse_args()
    return args
def check_build_args(args):
    result_dir = os.path.join(args.result_dir,args.dataset)
    data_dir = os.path.join(args.data_dir,args.dataset)
    log_dir = os.path.join(args.log_dir,args.dataset)
    assert os.path.exists(data_dir)    
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

def check_args(args):
    result_dir = os.path.join(args.result_dir,args.dataset)
    data_dir = os.path.join(args.data_dir,args.dataset)
    log_dir = os.path.join(args.log_dir,args.dataset)
    assert os.path.exists(data_dir)    
    assert os.path.exists(result_dir)
    assert os.path.exists(args.config_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
def load_config(config_dir):
    pass
