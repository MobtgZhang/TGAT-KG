import os
import time
import logging
import threading

import networkx as nx
from .data import GraphDFS

logger = logging.getLogger()

# the muti-processing for the dataset
class DataThread(threading.Thread):
    def __init__(self,subgraphs_path,data_dict,items_list,depth = 4,threadid=None,print_flag=False):
        super(DataThread,self).__init__()
        self.depth = depth
        self.print_flag = print_flag
        self.items_list = items_list
        self.subgraphs_path = subgraphs_path
        self.data_dict = data_dict
        import uuid
        self.threadid = threadid if threadid is not None else str(uuid.uuid1()).upper()
        self.spend_time = 0
    def run(self):
        start_t = time.time()
        for line in self.items_list:
            words_list = line.strip().split("\t")
            start_node = words_list[0]
            save_file_name = os.path.join(self.subgraphs_path,start_node + ".txt")            
            if os.path.exists(save_file_name):
                pass
            else:
                dg = nx.DiGraph()
                dg.add_node(start_node)
                dg = GraphDFS(self.data_dict,dg,start_node,depth=self.depth)
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

