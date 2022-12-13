import os
import time
import logging
import argparse
import networkx as nx
import threading

logger = logging.getLogger()
class DataThread(threading.Thread):
    def __init__(self,file_subGraphs,data_dict,items_list,threadid,print_flag=False):
        super(DataThread,self).__init__()
        self.print_flag = print_flag
        self.items_list = items_list
        self.file_subGraphs = file_subGraphs
        self.data_dict = data_dict
        self.threadid = threadid
        self.spend_time = 0
    def run(self):
        start_t = time.time()
        for line in self.items_list:
            words_list = line.strip().split("\t")
            node0 = words_list[1]
            save_file_name = os.path.join(self.file_subGraphs,node0 + ".txt")            
            if os.path.exists(save_file_name):
                pass
            else:
                dg = nx.DiGraph()
                dg.add_node(node0)
                dg = GraphDFS(self.data_dict,dg,node0,depth=4)
                with open(save_file_name,mode="w",encoding="utf-8") as wfp:
                    node_line = ""
                    for nodek in dg.nodes:
                        node_line = node_line + nodek + "\t"
                    wfp.write(node_line+"\n")
                    for e in dg.edges:
                        wfp.write(e[0] + "\t" + e[1] + "\t" + str(dg.edges[e[0],e[1]]['weight'])+'\n')
                if self.print_flag:
                    logger.info("saved file :(%s) cost time is %0.4f"%(save_file_name,time.time()-start_t))
        end_t = time.time()
        self.spend_time = end_t - start_t
                

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",default="./data",type=str)
    parser.add_argument("--result-dir",default="./result",type=str)
    parser.add_argument("--log-dir",default="./log",type=str)
    parser.add_argument("--workers",default=100,type=int)
    args = parser.parse_args()
    return args
def create_dataset(data_dict,file_entity,file_subGraphs,workers = 10):
    with open(file_entity,mode="r",encoding="utf-8") as rfp:
        lines = rfp.readlines()
        bt_len = len(lines)//workers
        all_theads_list = []
        for idx in range(workers):
            items_list = lines[idx*bt_len:(idx+1)*bt_len]
            threadid = "b-%d"%idx
            tmp_thread = DataThread(file_subGraphs,data_dict,items_list,threadid)
            all_theads_list.append(tmp_thread)
        for idx in range(workers):
            all_theads_list[idx].start()
        for idx in range(workers):
            all_theads_list[idx].join()
            logger.info("Thread name :[%s] cost time is %0.4f"%(all_theads_list[idx].threadid,all_theads_list[idx].spend_time))
def GraphDFS(data_dict,dg,node,depth):
    depth -= 1
    if depth <0:
        return dg
    if node not in data_dict:
        return dg
    sequence = data_dict[node]
    count = 0
    for key in sequence:
        if not dg.has_node(key):
            dg.add_node(key)
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
def read_all_triples(file_list):
    data_dict = {}
    for file_name  in file_list:
        with open(file_name,mode="r",encoding="utf-8") as rfp:
            for line in rfp:
                key_list = line.split(" ")
                if key_list[0] in data_dict:
                    if key_list[1] in data_dict[key_list[0]]:
                        data_dict[key_list[0]][key_list[1]].append(key_list[2].strip())
                    else:
                        data_dict[key_list[0]][key_list[1]] = [key_list[2].strip()]
                else:
                    data_dict[key_list[0]] = {key_list[1]:[key_list[2].strip()]}
    return data_dict

def main(args):
    file_entity = os.path.join(args.data_dir,"FB15K","KBCdataset","entity2id.txt")
    file_train = os.path.join(args.data_dir,"FB15K","golddataset","train2id.txt")
    file_test = os.path.join(args.data_dir,"FB15K","golddataset","test2id.txt")
    file_valid = os.path.join(args.data_dir,"FB15K","golddataset","valid2id.txt")
    file_subGraphs = os.path.join(args.result_dir,"FB15K","subGraphs_4")
    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)
    if not os.path.exists(file_subGraphs):
        os.makedirs(file_subGraphs)
    file_path = [file_train,file_valid,file_test]
    data_dict = read_all_triples(file_path)
    logger.info("data dictionary size:( %d )"%len(data_dict))
    create_dataset(data_dict,file_entity,file_subGraphs,args.workers)

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


