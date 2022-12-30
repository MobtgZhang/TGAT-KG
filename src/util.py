import os
import random

from .data import Dictionary

def create_entrels(data_dir,result_dir):
    tags_list = ["train","valid","test"]
    entities_dict = Dictionary()
    relations_dict = Dictionary()
    for file_tag in tags_list:
        load_filename = os.path.join(data_dir,"%s.txt"%file_tag)
        with open(load_filename,mode="r",encoding="utf-8") as rfp:
            for line in rfp:
                words = line.strip().split("\t")
                entities_dict.add(words[0])
                entities_dict.add(words[2])
                relations_dict.add(words[1])
    entities_file = os.path.join(result_dir,"entities.txt")
    relations_file = os.path.join(result_dir,"relations.txt")
    entities_dict.save(entities_file)
    relations_dict.save(relations_file)
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
