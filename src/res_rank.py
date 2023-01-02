import os
import time
import threading
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
    # calculate the number of tail entities of the graph
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
# calculate the PR value of the graph
class PRGraph:
    def __init__(self,dg,core_node):
        self.damping_factor = 0.15 # The damping coefficient
        self.max_iterations = 500 # The maximum iterations
        self.min_delta = 10e-7 # The parameter to determine whether the iteration ends
        self.core_node = core_node
        self.graph = dg
    def page_rank(self,print_flag=False):
        # calculate the node in-degree
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
                    rank += self.damping_factor*page_rank[tmp_node]/max(count,1)*float(self.graph.edges[incident_page]["weight"])
                rank += damping_value
                change += abs(page_rank[node]-rank)
                page_rank[node] = rank
            if change < self.min_delta:
                flag = True
                break
        if not flag:
            logger.info("core node:(%d) finished out of %s iterations!" % (self.core_node,self.max_iterations))
        return page_rank

# the multi-processing page rank algorithm
class PageRankDataThread(threading.Thread):
    def __init__(self,data_dict,entityRank_path,items_list,depth=4,threadid=None,print_flag=False):
        super(PageRankDataThread,self).__init__()
        self.data_dict = data_dict
        self.entityRank_path = entityRank_path
        self.items_list = items_list
        self.depth = depth
        import uuid
        self.threadid = str(uuid.uuid1()).upper() if threadid is None else threadid
        self.print_flag = print_flag
        self.spend_time = 0
    def run(self):
        start_t = time.time()
        for line in self.items_list:
            core_node,_ = line.strip().split("\t")
            save_file_name = os.path.join(self.entityRank_path,core_node+".txt")
            if os.path.exists(save_file_name):
                pass
            else:
                dg = nx.DiGraph()
                if self.print_flag:   
                    logger.info("core node:( %s )"%core_node)
                dg.add_node(core_node)
                dg = GraphDFS(self.data_dict,dg,core_node,depth=self.depth)
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
def create_page_rank(data_dict,entities_file,pagerank_path,depth=4,workers = 20):
    with open(entities_file,mode="r",encoding="utf-8") as rfp:
        lines = rfp.readlines()
        bt_len = len(lines)//workers
        all_theads_list = []
        for idx in range(workers):
            items_list = lines[idx*bt_len:(idx+1)*bt_len]
            threadid = "b-%d"%idx
            tmp_thread = PageRankDataThread(data_dict,pagerank_path,items_list,depth=depth,threadid=threadid,print_flag=False)            
            all_theads_list.append(tmp_thread)
        for idx in range(workers):
            all_theads_list[idx].start()
        for idx in range(workers):
            all_theads_list[idx].join()
            logger.info("Thread name :[%s] cost time is %0.4f"%(all_theads_list[idx].threadid,all_theads_list[idx].spend_time))
