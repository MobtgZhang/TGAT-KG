import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    result_dir = "./result"
    log_dir = os.path.join("TGAT-KG","log")
    all_dataset = {}
    for dataset_name in os.listdir(log_dir):
        if dataset_name not in all_dataset:
            all_dataset[dataset_name] = {}
            all_dataset[dataset_name]["test"] = {}
            all_dataset[dataset_name]["valid"] = {}
        dataset_dir = os.path.join(log_dir,dataset_name)
        if os.path.isdir(dataset_dir):
            for model_name in os.listdir(dataset_dir):
                model_dir = os.path.join(dataset_dir,model_name)
                if os.path.isdir(model_dir):
                    if model_name not in all_dataset[dataset_name]["test"]:
                        all_dataset[dataset_name]["test"][model_name] = {}
                    if model_name not in all_dataset[dataset_name]["valid"]:
                        all_dataset[dataset_name]["valid"][model_name] = {}
                    load_test_file = [os.path.join(model_dir,filename) for filename in os.listdir(model_dir) if "test" in filename]
                    load_valid_file = [os.path.join(model_dir,filename) for filename in os.listdir(model_dir) if "valid" in filename]
                    if len(load_test_file)!=0:
                        load_test_file = load_test_file[0]
                        all_dataset[dataset_name]["test"][model_name] = pd.read_csv(load_test_file)
                    if len(load_valid_file)!=0:
                        load_valid_file = load_valid_file[0]
                        all_dataset[dataset_name]["valid"][model_name] = pd.read_csv(load_valid_file)
    for dataset_name in all_dataset:
        tmp_save_dir = os.path.join(result_dir,dataset_name)
        if not os.path.exists(tmp_save_dir):
            os.makedirs(tmp_save_dir)
        for data_tag in all_dataset[dataset_name]:
            plt.figure(figsize=(12,10))
            tags_dict = {
                "loss":"loss",
                "f1-macro":"f1-macro",
                "f1-micro":"f1-micro",
                "acc":"accuracy"
            }
            for model_name in all_dataset[dataset_name][data_tag]:
                tmp_dataset = all_dataset[dataset_name][data_tag][model_name]
                for idx,tag_key in enumerate(tags_dict):
                    x = np.linspace(0,len(tmp_dataset)-1,len(tmp_dataset))
                    y_tp_t = tmp_dataset[tag_key].to_numpy()
                    plt.subplot(2,2,idx+1)
                    plt.plot(x,y_tp_t,label=model_name,marker=".")
                    plt.xlabel("Epoches")
                    plt.ylabel("%s value"%tags_dict[tag_key])
                    plt.legend(loc="best")
            plt.suptitle("The model result on %s %s"%(data_tag,dataset_name))
            save_filename = os.path.join(tmp_save_dir,"%s-%s.jpg"%(dataset_name,data_tag))
            plt.savefig(save_filename)
            plt.close()
if __name__ == "__main__":
    main()
