import os
import time
import logging

from config import get_args,check_build_args
from src.utils import create_entrels,create_pos_neg_ids
from src.data import Dictionary

def main(args):
    data_dir = os.path.join(args.data_dir,args.dataset)
    result_dir = os.path.join(args.result_dir,args.dataset)
    ent_filename = os.path.join(result_dir,"entities.txt")
    rel_filename = os.path.join(result_dir,"relations.txt")
    # create entities
    if not os.path.exists(ent_filename) or not os.path.exists(rel_filename):
        create_entrels(data_dir,result_dir)
    logger.info("The entities and relations saved in path: %s and %s ."%(ent_filename,rel_filename))
    # load dictionary
    ent_filename = os.path.join(result_dir,"entities.txt")
    ent_dict = Dictionary.load(ent_filename)
    rel_filename = os.path.join(result_dir,"relations.txt")
    rel_dict = Dictionary.load(rel_filename)
    # create dataset
    tags_list = ["train","valid","test"]
    for tag_name in tags_list:
        save_filename = os.path.join(result_dir,"%s2id.txt"%tag_name)
        if not os.path.exists(save_filename):
            load_filename = os.path.join(data_dir,"%s.txt"%tag_name)
            create_pos_neg_ids(load_filename,save_filename,ent_dict,rel_dict,True)
        logger.info("The positive and negative tags saved in path: %s ."%(save_filename))
if __name__ == "__main__":
    args = get_args()
    check_build_args(args)
    # First step, create a logger
    logger = logging.getLogger()
    # The log level switch
    logger.setLevel(logging.INFO)
    # Second step, create a handler,which is used to write to logging file.
    args.time_step_str = time.strftime("%Y%m%d%H%M",time.localtime(time.time()))
    log_file = os.path.join(args.log_dir,args.time_step_str+".log")
    # Third, define the output formatter
    format_str = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh = logging.FileHandler(log_file, mode='w')
    # the log's switch of the output log file
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(format_str)
    ch = logging.StreamHandler()
    ch.setFormatter(format_str)
    # Fourth, add the logger into handler
    logger.addHandler(fh)
    logger.addHandler(ch)
    main(args)
