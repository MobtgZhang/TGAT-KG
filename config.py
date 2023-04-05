import os
import argparse
import yaml
from src.kgtconv import KGTConv
from src.mixkgconv import MixKGATConv
from src.newconv import NewKGATConv

from src.trans import TransD,TransE,TransH,TransR

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir",default="./result",type=str)
    parser.add_argument("--log-dir",default="./log",type=str)
    parser.add_argument("--data-dir",default="./data",type=str)
    parser.add_argument("--config-dir",default="./config",type=str)
    parser.add_argument("--batch-size",default=2048,type=int)
    parser.add_argument("--dataset",default="fb15k",type=str)
    parser.add_argument("--model-name",default="KGATConv",type=str)
    parser.add_argument("--epoches",default=40,type=int)
    parser.add_argument("--max-norm",default=5.0,type=float)
    parser.add_argument("--learning-rate",default=0.01,type=float)
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
class Configuration:
    def __init__(self,data_dict):
        for key in data_dict:
            if not hasattr(self,key):
                setattr(self,key,data_dict[key])
        self.atts = [key for key in data_dict]
    def get_strs(self,):
        line = ""
        for key in self.atts:
            line += "%s=%s, "%(key,getattr(self,key))
        return line
    def __str__(self):
        return self.get_strs()
    def __repr__(self):
        return self.get_strs()
def load_config(config_file):
    with open(config_file,'r',encoding='utf-8') as rfp:
        tmp_dict = yaml.load(rfp,Loader=yaml.FullLoader)
        config = Configuration(tmp_dict)
    return config
def get_model(args):
    load_config_file = os.path.join(args.config_dir,args.model_name+".yaml")
    config = load_config(load_config)
    model_dict = {
        "MixKGATConv":MixKGATConv(config),
        "KGATConv":KGTConv(config),
        "NewKGATConv":NewKGATConv(config)
    }
    return model_dict[args.model_name]
