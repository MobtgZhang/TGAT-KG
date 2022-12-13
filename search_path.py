import os
import time
import argparse
import logging
import numpy as np
logger = logging.getLogger()
from search import read_all_triples
from utils import get_index,load_vec_txt

def cal_rank(paths_list, Ent2V, Rel2V, h, t, r):
    plist =[]

    for path in paths_list:
        sd_r = 0.0
        sd_h = 0.0
        sd_t = 0.0
        for triple in path:
            # print(triple)
            cosV_h = np.dot(Ent2V[int(h)], Ent2V[int(triple[1])]) / (np.linalg.norm(Ent2V[int(h)]) * np.linalg.norm(Ent2V[int(triple[1])]))
            sd_h +=cosV_h
            cosV_t = np.dot(Ent2V[int(t)], Ent2V[int(triple[0])]) / (np.linalg.norm(Ent2V[int(t)]) * np.linalg.norm(Ent2V[int(triple[0])]))
            sd_t +=cosV_t

            cosV_r = np.dot(Rel2V[int(r)], Rel2V[int(triple[2])]) / (np.linalg.norm(Rel2V[int(r)]) * np.linalg.norm(Rel2V[int(triple[2])]))
            sd_r +=cosV_r
        sd_v = (sd_r + sd_h + sd_t) / (3 * len(path))
        plist.append((sd_v, path))
    plist = sorted(plist, key=lambda sp: sp[0], reverse=True)
    return plist

def search_path(core, startnode, data_dict, taillist, Paths, pathlist, depth=5):
    depth -= 1

    if depth <= 0:
        return Paths

    if startnode not in data_dict:
        return Paths

    sequence = data_dict[startnode]
    count = 0
    for key in sequence.keys():

        if key in taillist:
            continue

        for val in sequence.get(key):
            pathlist.append((startnode, key, val))
            taillist.append(key)
            # print('***', pathlist)
            s = tuple(pathlist)
            if (core + '_' + key) not in Paths.keys():
                Paths[core + '_' + key] = [s]
            else:
                Paths[core + '_' + key].append(s)
            pathlist.remove((startnode, key, val))
            taillist.remove(key)
        for val in sequence.get(key):
            taillist.append(key)
            pathlist.append((startnode, key, val))
            Paths = search_path(core, key, data_dict, taillist, Paths, pathlist, depth)
            taillist.remove(key)
            pathlist.remove((startnode, key, val))

    return Paths

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir",default="./result",type=str)
    parser.add_argument("--log-dir",default="./log",type=str)
    parser.add_argument("--data-dir",default="./data",type=str)
    parser.add_argument("--dataset",default="FB15K",type=str)
    args = parser.parse_args()
    return args
def main(args):
    file_entity = os.path.join(args.data_dir,args.dataset,"KBCdataset","entity2id.txt")
    file_relation = os.path.join(args.data_dir,args.dataset,"KBCdataset","relation2id.txt")
    file_train = os.path.join(args.data_dir,args.dataset,"golddataset","train2id.txt")
    file_valid = os.path.join(args.data_dir,args.dataset,"golddataset","valid2id.txt")
    file_test = os.path.join(args.data_dir,args.dataset,"golddataset","test2id.txt")

    file_ent2vec = os.path.join(args.data_dir,args.dataset,"KBCdataset","FB15K_PTransE_Entity2Vec_100.txt")
    file_rel2vec = os.path.join(args.data_dir,args.dataset,"KBCdataset","FB15K_PTransE_Relation2Vec_100.txt")

    data_dict = read_all_triples([file_train, file_test, file_valid])
    logger.info("dict size ( %d )"%(len(data_dict)))
    logger.info("ReadAllTriples is done!")
    rel_vocab, rel_idex_word = get_index(file_relation)
    relvec_k, Rel2V = load_vec_txt(file_rel2vec, rel_vocab,k_num=100)
    ent_vocab, ent_idex_word = get_index(file_entity)
    entvec_k, Ent2V = load_vec_txt(file_ent2vec, ent_vocab,k_num=100)

    file_path = os.path.join(args.result_dir,args.dataset,"Path_4")
    files_list = os.listdir(file_path)
    files_indexes_list = set([filename.split("_")[0] for filename in files_list])
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    
    line_dict = {}
    headlist = []

    files_root_path = os.path.join(args.data_dir,args.dataset,"KBCdataset")

    for filename in ["conf_train2id.txt","conf_test2id.txt"]:
        load_file_name = os.path.join(files_root_path,filename)
        with open(load_file_name,mode="r",encoding="utf-8") as rfp:
            for linet in rfp:
                tmp_list = linet.strip().split('\t')

                if tmp_list[0]+'_'+tmp_list[1] in line_dict.keys():
                    if (tmp_list[0],tmp_list[1],tmp_list[2]) not in line_dict[tmp_list[0]+'_'+tmp_list[1]]:
                        line_dict[tmp_list[0] + '_' + tmp_list[1]].append((tmp_list[0],tmp_list[1],tmp_list[2]))
                else:
                    line_dict[tmp_list[0] + '_' + tmp_list[1]] = [(tmp_list[0],tmp_list[1],tmp_list[2])]
                if int(tmp_list[0]) not in headlist:
                    headlist.append(int(tmp_list[0]))
    
    for i in range(2000, 2500):#561, 2500  2750, 5000
        if i in headlist:
            startnode = str(i)
            all_paths = {}
            pathlist = []
            taillist = [startnode]
            if str(i) in files_indexes_list:
                continue
            all_paths = search_path(startnode, startnode, data_dict, taillist, all_paths, pathlist, 4)

            for head in all_paths:
                if head in line_dict:
                    for tri in line_dict[head]:
                        logger.info('------------------'+str(i)+'--------------  %s'%str(tri))
                        save_file_name = os.path.join(file_path,tri[0] + '_' + tri[1] + '_' + tri[2] +'.txt')
                        if os.path.exists(save_file_name):
                            continue
                        else:
                            p_rank_list = cal_rank(all_paths[head], Ent2V, Rel2V, tri[0], tri[1], tri[2])
                            with open(save_file_name,'w') as wfp:
                                for num, ps in enumerate(p_rank_list):
                                    if num > 50:
                                        break
                                    if ps[1] == ((tri[0], tri[1], tri[2]),):
                                        continue
                                    for tri in ps[1]:
                                        wfp.write('('+tri[0]+', '+tri[1]+', '+tri[2]+')'+'\t')
                                    wfp.write(str(ps[0]) + '\n')

if __name__ == "__main__":
    args = get_args()
    # First step, create a logger
    logger = logging.getLogger()
    # The log level switch
    logger.setLevel(logging.INFO)
    # Second step, create a handler,which is used to write to logging file.
    time_step_str = time.strftime("%Y%m%d%H%M",time.localtime(time.time()))
    log_path = args.log_dir
    log_file = os.path.join(args.log_dir,time_step_str+".log")

    fh = logging.FileHandler(log_file, mode='w')
    # the log's switch of the output log file
    fh.setLevel(logging.DEBUG)
    # Third, define the output formatter
    format_str = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(format_str)
    ch = logging.StreamHandler()
    ch.setFormatter(format_str)
    # Fourth, add the logger into handler
    logger.addHandler(fh)
    logger.addHandler(ch)
    main(args)




