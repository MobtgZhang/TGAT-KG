import os
import random
import numpy as np
import torch
from torch_geometric.data import Data

from .data import Dictionary
def to_var(item,device):
    for k in range(len(item)):
        item[k] = item[k].to(device)
        
def create_entrels(data_dir,result_dir):
    tags_list = ["train","valid","test"]
    rel_dict = Dictionary()
    ent_dict = Dictionary()
    for tag in tags_list:
        load_filename = os.path.join(data_dir,tag+".txt")
        with open(load_filename,mode="r",encoding="utf-8") as rfp:
            for line in rfp:
                out = line.strip().split("\t")
                ent_dict.add(out[0])
                rel_dict.add(out[1])
                ent_dict.add(out[2])
    ent_filename = os.path.join(result_dir,"entities.txt")
    rel_filename = os.path.join(result_dir,"relations.txt")
    ent_dict.save(ent_filename)
    rel_dict.save(rel_filename)

def create_pos_neg_ids(tag_name,data_dir,result_dir,ent_dict,rel_dict):
    all_triple_list = []
    tp_rel_dict = {}
    save_filename = os.path.join(result_dir,"%s2id.txt"%tag_name)
    load_filename = os.path.join(data_dir,"%s.txt"%tag_name)
    with open(load_filename,mode="r",encoding="utf-8") as rfp:
        for line in rfp:
            words = line.strip().split("\t")
            w_triple = [ent_dict[words[0]],rel_dict[words[1]],ent_dict[words[2]],1]
            all_triple_list.append(w_triple)
            rel = w_triple[1]
            if rel not in tp_rel_dict:
                tp_rel_dict[rel] = {
                    "h":set(),
                    "t":set()
                }
            tp_rel_dict[rel]["h"].add(w_triple[0])
            tp_rel_dict[rel]["t"].add(w_triple[2])
    length = len(all_triple_list)
    if tag_name!="train":
        triple_data_dict = {
            "_rt":[],
            "h_t":[],
            "hr_":[]
        }
    # generate negative triples
    for idx in range(length):
        item = all_triple_list[idx]
        rel = item[1]
        n_h = len(tp_rel_dict[rel]["h"])
        n_t = len(tp_rel_dict[rel]["t"])
        tph = n_t/n_h
        hpt = n_h/n_t
        if tph>hpt:
            # replace the head entity
            old_tail = item[2]
            old_head = item[0]
            while True:
                new_head = random.choice(range(len(ent_dict)))
                if new_head!=item[0]:
                    break
            if tag_name!="train":
                triple_data_dict["_rt"].append([new_head,rel,old_tail,-1])
                triple_data_dict["_rt"].append([old_head,rel,old_tail,1])
            all_triple_list.append([new_head,rel,old_tail,-1])
        else:
            # replace the tail entity
            old_head = item[0]
            old_tail = item[2]
            while True:
                new_tail = random.choice(range(len(ent_dict)))
                if new_tail!=item[2]:
                    break
            if tag_name!="train":
                triple_data_dict["hr_"].append([old_head,rel,new_tail,-1])
                triple_data_dict["hr_"].append([old_head,rel,old_tail,1])
            all_triple_list.append([old_head,rel,new_tail,-1])
        # replace the relation
        old_rel = item[1]
        while True:
            new_rel = random.choice(range(len(rel_dict)))
            if new_rel!=item[1]:
                break
        if tag_name!="train":
            triple_data_dict["h_t"].append([item[0],new_rel,item[2],-1])
            triple_data_dict["h_t"].append([item[0],old_rel,item[2],1])
        all_triple_list.append([item[0],new_rel,item[2],-1])
    # shuffle the list
    random.shuffle(all_triple_list)
    with open(save_filename,mode="w",encoding="utf-8") as wfp:
        for item in all_triple_list:
            str_line = "%s\t%s\t%s\t%s\t\n"%(item[0],item[1],item[2],item[3])
            wfp.write(str_line)
    if tag_name != "train":
        for key in triple_data_dict:
            save_change_filename = os.path.join(result_dir,tag_name+"2id_%s.txt"%key)
            with open(save_change_filename,mode="w",encoding="utf-8") as wfp:
                random.shuffle(triple_data_dict[key])
                for item in triple_data_dict[key]:
                    str_line = "%s\t%s\t%s\t%s\t\n"%(item[0],item[1],item[2],item[3])
                    wfp.write(str_line)
def build_graph(result_dir,tags_list):
    edge_index = []
    edge_attr = []
    node_index = []
    for tag in tags_list:
        load_filename = os.path.join(result_dir,tag+"2id.txt")
        with open(load_filename,mode="r",encoding="utf-8") as rfp:
            for line in rfp:
                out = [int(item) for item in line.strip().split("\t")]
                if out[-1] == 1:
                    head,rel,tail = out[:3]
                    edge_index.append([head,tail])
                    edge_attr.append(rel)
                    node_index.append(head)
                    node_index.append(tail)
                else:
                    pass
    edge_index = torch.tensor(edge_index,dtype=torch.long).T
    edge_attr = torch.tensor(edge_attr,dtype=torch.long)
    node_index = torch.tensor(node_index,dtype=torch.long)
    graph = Data(x=node_index,edge_index=edge_index)
    graph.edge_type = edge_attr
    return graph
def load_dataset(file_name):
    with open(file_name,mode="r",encoding="utf-8") as rfp:
        data_list = []
        for line in rfp:
            out = [int(t) for t in line.strip().split("\t")]
            out[-1] = 0 if out[-1]==-1 else 1
            data_list.append(out)
    return data_list
