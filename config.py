import os
import argparse
import yaml
from src.kgtconv import KGATConv
from src.transmodel import TransModel
from src.rgcnmodel import RGCNModel

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
    parser.add_argument("--sigmoid",action="store_true")
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
    log_dir = os.path.join(args.log_dir,args.dataset,args.model_name)
    assert os.path.exists(data_dir)    
    assert os.path.exists(result_dir)
    assert os.path.exists(args.config_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
class Configuration:
    def __init__(self,data_dict=None):
        if data_dict is not None:
            for key in data_dict:
                if not hasattr(self,key):
                    setattr(self,key,data_dict[key])
            self.attrs = [key for key in data_dict]
        else:
            self.attrs = []
    def add_attrs(self,key,value):
        self.attrs.append(key)
        if not hasattr(self,key):
            setattr(self,key,value)
    def get_attrs(self):
        re_dict = {}
        for key in self.attrs:
            re_dict[key] = getattr(self,key)
        return re_dict
    def get_strs(self,):
        line = ""
        for key in self.attrs:
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
def get_model(config):
    model_dict = {
        "TransE":TransModel,
        "TransD":TransModel,
        "TransH":TransModel,
        "TransR":TransModel,
        "RGCN":RGCNModel,
        "KGATConv":KGATConv
    }
    return model_dict[config.model_name](config)

def get_KGATConv_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir",default="./result",type=str)
    parser.add_argument("--log-dir",default="./log",type=str)
    parser.add_argument("--data-dir",default="./data",type=str)
    parser.add_argument("--config-dir",default="./config",type=str)
    parser.add_argument("--batch-size",default=2048,type=int)
    parser.add_argument("--dataset",default="fb15k",type=str)
    parser.add_argument("--epoches",default=40,type=int)
    parser.add_argument("--max-norm",default=5.0,type=float)
    parser.add_argument("--learning-rate",default=0.02,type=float)
    parser.add_argument("--sigmoid",action="store_true")
    # model parameters
    parser.add_argument("--emb-dim",default=30,type=int)
    parser.add_argument("--k-pr",default=10,type=int)
    parser.add_argument("--alpha",default=0.65,type=float)
    parser.add_argument("--beta",default=0.90,type=float)
    parser.add_argument("--dropout",default=0.20,type=float)
    parser.add_argument("--num-bases",default=10,type=int)
    parser.add_argument("--out-dim",default=10,type=int)
    parser.add_argument("--num-layers",default=10,type=int)
    parser.add_argument("--heads",default=3,type=int)
    parser.add_argument("--model-type",default="prgat",type=str)
    parser.add_argument("--trans-type",default="TransR",type=str)
    parser.add_argument("--method-type",default="fnn",type=str)
    args = parser.parse_args()
    args.model_name = "KGATConv"
    return args
