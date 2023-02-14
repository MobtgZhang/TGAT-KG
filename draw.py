import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir",default="./log",type=str)
    parser.add_argument("--pic-dir",default="./pictures",type=str)
    args = parser.parse_args()
    return args
def main():
    args = get_args()
    if not os.path.exists(args.pic_dir):
        os.mkdir(args.pic_dir)
    for dataset_name in os.listdir(args.log_dir):
        dataset_dir = os.path.join(args.log_dir,dataset_name)
        if os.path.isfile(dataset_dir):
            continue
        for filename in os.listdir(dataset_dir):
            if "test" in filename:
                load_test_filename = os.path.join(dataset_dir,filename)
            if "valid" in filename:
                load_valid_filename = os.path.join(dataset_dir,filename)
        test_dataset = pd.read_csv(load_test_filename)
        valid_dataset = pd.read_csv(load_valid_filename)
        x = len(test_dataset)
        tags_list = ["loss","f1-macro","f1-micro","acc"]
        plt.figure(figsize=(12,10))
        for idx,tag in enumerate(tags_list):
            x = np.linspace(0,len(test_dataset)-1,len(test_dataset))
            y_tp_t = test_dataset[tag].to_numpy()
            y_tp_v = valid_dataset[tag].to_numpy()
            plt.subplot(2,2,idx+1)
            plt.plot(x,y_tp_t,label="test") 
            plt.plot(x,y_tp_v,label="valid")
            plt.xlabel("Epoches")
            plt.ylabel("%s"%tag)
            plt.legend(loc="best")
        plt.suptitle("The model in %s result"%dataset_name)
        save_filename = os.path.join(args.pic_dir,"%s.png"%dataset_name)
        plt.savefig(save_filename)
        plt.close()
if __name__ == "__main__":
    main()


