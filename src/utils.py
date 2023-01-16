import os
import random
import numpy as np
import torch
from torch_geometric.data import Data

from .data import Dictionary

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

def create_pos_neg_ids(load_filename,save_filename,ent_dict,rel_dict,generate_negs = False):
    all_data_list = []
    if generate_negs:
        tp_rel_dict = {}
    with open(load_filename,mode="r",encoding="utf-8") as rfp:
        for line in rfp:
            words = line.strip().split("\t")
            w_triple = [ent_dict[words[0]],rel_dict[words[1]],ent_dict[words[2]],1]
            all_data_list.append(w_triple)
            if generate_negs:
                rel = w_triple[1]
                if rel not in tp_rel_dict:
                    tp_rel_dict[rel] = {
                        "h":set(),
                        "t":set()
                    }
                tp_rel_dict[rel]["h"].add(w_triple[0])
                tp_rel_dict[rel]["t"].add(w_triple[2])
    if generate_negs:
        length = len(all_data_list)
        # generate negative triples
        for idx in range(length):
            item = all_data_list[idx]
            rel = item[1]
            n_h = len(tp_rel_dict[rel]["h"])
            n_t = len(tp_rel_dict[rel]["t"])
            tph = n_t/n_h
            hpt = n_h/n_t
            if tph>hpt:
                # replace the head entity
                new_tail = item[2]
                while True:
                    new_head = random.choice(range(len(ent_dict)))
                    if new_head!=item[0]:
                        break
            else:
                # replace the tail entity
                new_head = item[0]
                while True:
                    new_tail = random.choice(range(len(ent_dict)))
                    if new_tail!=item[2]:
                        break
            w_triple = [new_head,rel,new_tail,-1]
            all_data_list.append(w_triple)
    # shuffle the list
    random.shuffle(all_data_list)
    with open(save_filename,mode="w",encoding="utf-8") as wfp:
        for item in all_data_list:
            str_line = "%d\t%d\t%d\t%d\t\n"%(item[0],item[1],item[2],item[3])
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

