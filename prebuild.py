import logging

from src.res_rank import DataThread
logger = logging.getLogger()

def create_dataset(data_dict,load_entity_file,subgraphs_path,depth=4,workers = 10):
    with open(load_entity_file,mode="r",encoding="utf-8") as rfp:
        lines = rfp.readlines()
        bt_len = len(lines)//workers
        all_theads_list = []
        for idx in range(workers):
            items_list = lines[idx*bt_len:(idx+1)*bt_len]
            threadid = "b-%d"%idx
            tmp_thread = DataThread(subgraphs_path,data_dict,items_list,depth=depth,threadid=threadid)
            all_theads_list.append(tmp_thread)
        for idx in range(workers):
            all_theads_list[idx].start()
        for idx in range(workers):
            all_theads_list[idx].join()
            logger.info("Thread name :[%s] cost time is %0.4f"%(all_theads_list[idx].threadid,all_theads_list[idx].spend_time))


