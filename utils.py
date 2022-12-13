import os
import logging
import numpy as np

logger = logging.getLogger()
def get_dataset(hrt_filename):
    train_triples = []
    train_confidence = []
    with open(hrt_filename,mode="r",encoding="utf-8") as rfp:
        for line in rfp:
            raw_triple = line.strip().split("\t")
            raw_triple = [int(item) for item in raw_triple]
            train_triples.append([raw_triple[0],raw_triple[1],raw_triple[2],raw_triple[3]])
            if raw_triple[3] == "1":
                train_confidence.append([0,1])
            else:
                train_confidence.append([1,0])
    return train_triples,train_confidence

def load_vec_txt(vec_file,vocab,k_num):
    w2v_dict = {}
    idx_mat = np.zeros(shape=(len(vocab)+2,k_num))
    unknown_tokens = 0
    with open(vec_file,mode="r",encoding="utf-8") as rfp:
        for line in rfp:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:],dtype=np.float32)
            w2v_dict[word] = coefs
    for word in vocab:
        if word not in w2v_dict:
            w2v_dict[word] = np.random.uniform(-0.25,0.25,k_num)
            unknown_tokens += 1
            idx_mat[vocab[word]] = w2v_dict[word]
        else:
            idx_mat[vocab[word]] = w2v_dict[word]
    logger.info("Number of unknow tokens in dictionary is %d"%unknown_tokens)
    return k_num,idx_mat

def get_index(index_filename):
    source_vocab = {}
    source_idx_word = {}
    with open(index_filename,mode="r",encoding="utf-8") as rfp:
        for line in rfp:
            sourc = line.strip().split("\t")
            if not sourc[0] in source_vocab:
                source_vocab[sourc[0]] = int(sourc[1])
                source_idx_word[int(sourc[1])] = sourc[0]
    return source_vocab,source_idx_word

def get_trans_features(entity_rank_path):
    feature_dict = {}
    files_list = os.listdir(entity_rank_path)
    idx = 1
    for filename in files_list:
        idx += 1
        tmp_filename = os.path.join(entity_rank_path,filename)
        nodes_dict = {}
        with open(tmp_filename,mode="r",encoding="utf-8") as rfp:
            for line in rfp:
                nodes = line.strip().split("\t")
                nodes_dict[int(nodes[0])] = [float(nodes[1]), float(nodes[2]), float(nodes[3]), float(nodes[4]), float(nodes[5]), float(nodes[6])]
        feature_dict[int(os.path.splitext(filename)[0])] = nodes_dict
    return feature_dict

def get_rrank_features(dict_features,examples_list):
    features = []
    for idx,example in enumerate(examples_list):
        if example[1] in dict_features[example[0]]:
            features.append(dict_features[example[0]][example[1]])
        else:
            features.append([0.0, 0.0, 0.0, 0.0, 0.0, 10000.0])
    return features
def get_path_index(load_file_path, max_p, triples, topk):
    train_path_h = []
    train_path_t = []
    train_path_r = []

    for baset in triples:
        ph = []
        pt = []
        pr = []
        length = 0
        file_name_str = os.path.join(load_file_path,str(baset[0])+'_'+ str(baset[1])+'_'+ str(baset[2])+'.txt')
        if os.path.exists(file_name_str):
            with open(file_name_str,mode="r",encoding="utf-8") as rfp:
                lines = rfp.readlines()
                if len(lines)>=(topk+1):
                    tmp_triple = lines[topk].strip().split("\t")
                    for t in range(0,len(tmp_triple)-1):
                        idx = tmp_triple[t].strip("(").strip(")").split(", ")
                        ph.append(int(idx[0]))
                        pt.append(int(idx[1]))
                        pr.append(int(idx[2]))
                    length = len(tmp_triple) -1 
            for k in range(0,max_p-length):
                ph.append(0)
                pr.append(0)
                pt.append(0)
            train_path_h.append(ph)
            train_path_r.append(pr)
            train_path_t.append(pt)
        else:
            logger.info("Not find the file %s"%file_name_str)
            exit()
    return train_path_h,train_path_t,train_path_r
                


