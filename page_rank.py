import os
import time
import argparse
import logging
import threading
import networkx as nx
from tqdm import tqdm
logger = logging.getLogger()

class PageRankDataThread(threading.Thread):
    def __init__(self,file_subGraphs,file_entityRank,files_list,threadid,print_flag=False):
        super(PageRankDataThread,self).__init__()
        self.print_flag = print_flag
        self.files_list = files_list
        self.file_entityRank = file_entityRank
        self.file_subGraphs = file_subGraphs
        self.threadid = threadid
        self.spend_time = 0
    def run(self):
        start_t = time.time()
        for filename in self.files_list:
            load_file_name = os.path.join(self.file_subGraphs,filename)
            core_node = os.path.splitext(filename)[0]
            save_file_name = os.path.join(self.file_entityRank,core_node+".txt")
            if os.path.exists(save_file_name):
                pass
            else:
                with open(load_file_name,mode="r",encoding="utf-8") as rfp:
                    dg = nx.DiGraph()
                    if self.print_flag:   
                        logger.info("core node:( %s )"%core_node)
                    for idx,line in enumerate(rfp.readlines()):
                        if idx == 0:
                            tmp_list = line.strip().split("\t")
                            for n in tmp_list:
                                dg.add_node(n.strip())
                        else:
                            triple_list = line.strip().split("\t")
                            dg.add_edge(triple_list[0],triple_list[1],weight=triple_list[2])
                if self.print_flag:
                    logger.info("dg size: (%d)"%(len(dg.nodes)))
                pr = PRGraph(dg,core_node)
                page_rank = pr.page_rank()

                
                if not os.path.exists(save_file_name):
                    with open(save_file_name,mode="w",encoding="utf-8") as wfp:
                        for key in page_rank:
                            wfp.write(key+"\t"+str(page_rank[key])+"\n")
        end_t = time.time()
        self.spend_time = end_t - start_t


class PRGraph:
    """
    计算一张图当中的PR值
    """
    def __init__(self,dg,core_node):
        self.damping_factor = 0.85 # 阻尼系数
        self.max_iterations = 500 # 最大迭代次数
        self.min_delta = 10e-7 # 确定迭代是否结束的参数
        self.core_node = core_node
        self.graph = dg
    def page_rank(self,print_flag=False):
        count = 0
        for node in self.graph.nodes:
            if len(list(self.graph.neighbors(node))) == 0:
                count += 1
                self.graph.add_edge(node, node, weight=0.5)
                self.graph.add_edge(node,self.core_node, weight=0.5)
        if print_flag:
            logger.info("count is (%d)"%count)
        nodes = self.graph.nodes
        graph_size = len(nodes)
        if graph_size == 0:
            return {}
        page_rank = dict.fromkeys(nodes,0.0)
        page_rank[self.core_node] = 1.0
        damping_value = (1.0-self.damping_factor)/graph_size
        if print_flag:
            logger.info("start iterating ......")
        flag = False
        for idx in range(self.max_iterations):
            change = 0
            for node in nodes:
                rank = 0
                for incident_page in self.graph.in_edges(node):
                    tmp_node = incident_page[0]
                    if count == 0:
                        rank += self.damping_factor*page_rank[tmp_node]*float(self.graph.edges[incident_page]["weight"])
                    else:
                        rank += self.damping_factor*page_rank[tmp_node]/count*float(self.graph.edges[incident_page]["weight"])
                rank += damping_value
                change += abs(page_rank[node]-rank)
                page_rank[node] = rank
            if change < self.min_delta:
                flag = True
                break
        if flag == False:
            logger.info("core node:(%d) finished out of %s iterations!" % (self.core_node,self.max_iterations))
        return page_rank

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir",default="./log",type=str)
    parser.add_argument("--result-dir",default="./result",type=str)
    parser.add_argument("--workers",default=100,type=int)
    args = parser.parse_args()
    return args
def main(args):
    file_subGraphs = os.path.join(args.result_dir,"FB15K","subGraphs_4")
    assert os.path.exists(file_subGraphs)
    file_entityRank = os.path.join(args.result_dir,"FB15K","entityRank_4")
    if not os.path.exists(file_entityRank):
        os.makedirs(file_entityRank)
    files_list = os.listdir(file_subGraphs)
    bt_len = len(files_list)//args.workers
    all_theads_list = []
    for idx in range(args.workers):
        items_list = files_list[idx*bt_len:(idx+1)*bt_len]
        threadid = "b-%d"%idx
        tmp_thread = PageRankDataThread(file_subGraphs,file_entityRank,items_list,threadid,False)
        all_theads_list.append(tmp_thread)
    for idx in range(args.workers):
        all_theads_list[idx].start()
    for idx in range(args.workers):
        all_theads_list[idx].join()
        logger.info("Thread name :[%s] cost time is %0.4f"%(all_theads_list[idx].threadid,all_theads_list[idx].spend_time))
    
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


