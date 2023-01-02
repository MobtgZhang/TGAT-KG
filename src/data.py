import logging
import networkx as nx

logger = logging.getLogger()
# read triples of the dataset
def read_all_triples(file_list):
    data_dict = {}
    for file_name  in file_list:
        with open(file_name,mode="r",encoding="utf-8") as rfp:
            for line in rfp:
                key_list = line.strip().split("\t")
                h_key,r_key,t_key,score = key_list
                if score == "1":
                    if h_key in data_dict:
                        if r_key in data_dict[h_key]:
                            data_dict[h_key][r_key].append(t_key.strip())
                        else:
                            data_dict[h_key][r_key] = [t_key.strip()]
                    else:
                        data_dict[h_key] = {r_key:[t_key.strip()]}
                else:
                    continue
    return data_dict
# DFS method to search the graph
def GraphDFS(data_dict,dg,node,depth):
    depth -= 1
    if depth <0:
        return dg
    if node not in data_dict:
        return dg
    sequence = data_dict[node]
    count = 0
    for key in sequence:
        # added relation node
        if not dg.has_node(key):
            dg.add_node(key)
        # added head entity to relation edge
        if not dg.has_edge(node,key):
            dg.add_edge(node, key, weight=len(sequence[key]))
            count += len(sequence[key])
        else:
            continue
        dg = GraphDFS(data_dict,dg,key,depth)
    for n in dg.neighbors(node):
        attrs = {(node,n):{"weight":float(dg.edges[node,n]["weight"]/max(count,1))}}
        dg.update(edges=attrs)
    return dg
# the dictionary of the relation and entity
class Dictionary:
    def __init__(self,unk_token= "UNK_VAL"):
        self.words2id = {unk_token:0}
        self.id2words = [unk_token]
        self.start_id = -1
    def add(self,word):
        if word not in self.words2id:
            self.words2id[word] = len(self.words2id)
            self.id2words.append(word)
    def __getitem__(self,key):
        if type(key) == str:
            return self.words2id.get(key,0)
        elif type(key) == int:
            return self.id2words[key]
        else:
            raise TypeError("The key type %s is unknown."%str(type(key)))
    def __next__(self):
        if self.start_id>=len(self.id2words)-1:
            self.start_id = -1
            raise StopIteration()
        else:
            self.start_id += 1
            return self.id2words[self.start_id]
    def __iter__(self):
        return self
    def __repr__(self):
        re_str = "Dictionary (%d)"%len(self.id2words)
        return re_str
    def __str__(self):
        re_str = "Dictionary (%d)"%len(self.id2words)
        return re_str
    def __len__(self):
        return len(self.id2words)
    @staticmethod
    def load(load_file):
        tmp_dict = Dictionary()
        words2id = {}
        id2words = []
        with open(load_file,mode="r",encoding="utf-8") as rfp:
            for line in rfp:
                idx,key = line.strip().split("\t")
                id2words.append(key)
                words2id[key] = int(idx)
        tmp_dict.id2words = id2words
        tmp_dict.words2id = words2id
        return tmp_dict
    def save(self,save_file):
        with open(save_file,mode="w",encoding="utf-8") as wfp:
            for idx,key in enumerate(self.id2words):
                write_line = "%d\t%s\n"%(idx,key)
                wfp.write(write_line)

