import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    root_dir = "./result"
    dataset_dict = {}
    
    name_list = os.listdir(root_dir)
    for data_name in name_list:
        dataset_dict[data_name] = {}
        all_tags_list = ["train2id","valid2id","test2id"]
        valid_hrt = ["valid2id__rt","valid2id_h_t","valid2id_hr_"]
        test_hrt = ["test2id__rt","test2id_h_t","test2id_hr_"]
        other_tags = ["entities","relations"]
        for tag_name in all_tags_list+valid_hrt+test_hrt+other_tags:
            load_file = os.path.join(root_dir,data_name,"%s.txt"%tag_name)
            with open(load_file,mode="r",encoding="utf-8") as rfp:
                dataset_dict[data_name][tag_name] = len(list(rfp.readlines()))
    headers_list = ["dataset name","relations","entities","triples","train/valid/test","_rt/h_t/hr_(valid)","_rt/h_t/hr_(test)"]
    data_list = []
    for data_name in name_list:
        rel_num = dataset_dict[data_name]["entities"]
        ent_num = dataset_dict[data_name]["relations"]
        test_num = dataset_dict[data_name]["test2id"]
        valid_num = dataset_dict[data_name]["valid2id"]
        train_num = dataset_dict[data_name]["train2id"]
        test__rt_num = dataset_dict[data_name]["test2id__rt"]
        test_h_t_num = dataset_dict[data_name]["test2id_h_t"]
        test_hr__num = dataset_dict[data_name]["test2id_hr_"]
        
        valid__rt_num = dataset_dict[data_name]["valid2id__rt"]
        valid_h_t_num = dataset_dict[data_name]["valid2id_h_t"]
        valid_hr__num = dataset_dict[data_name]["valid2id_hr_"]
        
        all_tri = test_num + valid_num + train_num
        tp_list = [data_name,rel_num,ent_num,all_tri,"%d/%d/%d"%(train_num,valid_num,test_num),"%d/%d/%d"%(valid__rt_num,valid_h_t_num,valid_hr__num),"%d/%d/%d"%(test__rt_num,test_h_t_num,test_hr__num)]
        data_list.append(tp_list)
    save_xlsx_file = "data.xlsx"
    pd.DataFrame(data_list,columns=headers_list).to_excel(save_xlsx_file,index=None)
    # 画出实体个数统计图
    xlabels = list(dataset_dict.keys())
    y1 = [dataset_dict[key]["entities"] for key in dataset_dict]
    y2 = [dataset_dict[key]["relations"] for key in dataset_dict]
    x = np.arange(0,len(xlabels))
    width = 0.7
    for y,name,tpv,color in zip([y1,y2],["entities","relations"],[2000,10],['b','r']):
        plt.figure(figsize=(9,6))

        plt.bar(x,y,width=-width,bottom=0,align='edge',color=color,edgecolor ='#000000',linewidth=1)
        for i,v in enumerate(y): plt.text(i-width/2,v+tpv, str(v), ha='center',fontsize=10)
        plt.xticks(x-width/2, xlabels) # 设置X轴刻度标签
        # 设置轴标签
        plt.xlabel("Dataset Names",size=12)
        plt.ylabel("Number",size=12)
        plt.savefig("%s.jpg"%name)
        plt.close()
    # 画出分布比例图
    x_value = np.arange(len(xlabels))
    valid2id__rt_data = [dataset_dict[key]["valid2id__rt"] for key in dataset_dict]
    valid2id_h_t_data = [dataset_dict[key]["valid2id_h_t"] for key in dataset_dict]
    valid2id_hr__data = [dataset_dict[key]["valid2id_hr_"] for key in dataset_dict]
    test2id__rt_data = [dataset_dict[key]["test2id__rt"] for key in dataset_dict]
    test2id_h_t_data = [dataset_dict[key]["test2id_h_t"] for key in dataset_dict]
    test2id_hr__data = [dataset_dict[key]["test2id_hr_"] for key in dataset_dict]
    # 绘制图形
    # valid dataset
    width=0.33
    tpv = 800
    plt.figure(figsize=(15,12))
    plt.bar(x_value-width, valid2id__rt_data, width=width, color='r', align='center', label=' _rt')
    for i,v in enumerate(valid2id__rt_data): plt.text(i-width,v+tpv, str(v), ha='center',fontsize=6)
    plt.bar(x_value, valid2id_h_t_data, width=width, color='g', align='center', label='h_t')
    for i,v in enumerate(valid2id_h_t_data): plt.text(i,v+tpv, str(v), ha='center',fontsize=6)
    plt.bar(x_value+width, valid2id_hr__data, width=width, color='b', align='center', label='hr_')
    for i,v in enumerate(valid2id_hr__data): plt.text(i+width,v+tpv, str(v), ha='center',fontsize=6)
    # 添加标签
    plt.xlabel("The wrong triple statics in valid datset",size=12)
    plt.ylabel("Number",size=12)
    plt.xticks(x_value, xlabels) # 设置X轴刻度标签
    plt.legend(loc='best')
    plt.savefig("wrong-valid.jpg")
    plt.close()


    width=0.33
    tpv = 800
    plt.figure(figsize=(15,12))
    # test dataset
    plt.bar(x_value-width, test2id__rt_data, width=width, color='r', align='center', label=' _rt')
    for i,v in enumerate(test2id__rt_data): plt.text(i-width,v+tpv, str(v), ha='center',fontsize=6)
    plt.bar(x_value, test2id_h_t_data, width=width, color='g', align='center', label='h_t')
    for i,v in enumerate(test2id_h_t_data): plt.text(i,v+tpv, str(v), ha='center',fontsize=6)
    plt.bar(x_value+width, test2id_hr__data, width=width, color='b', align='center', label='hr_')
    for i,v in enumerate(test2id_hr__data): plt.text(i+width,v+tpv, str(v), ha='center',fontsize=6)
    # 添加标签
    plt.xlabel("The wrong triple statics in test datset",size=12)
    plt.ylabel("Number",size=12)
    plt.xticks(x_value, xlabels) # 设置X轴刻度标签
    plt.legend(loc='best')
    plt.savefig("wrong-test.jpg")
    plt.close()
    
    
    # 画出分布比例图
    x_value = np.arange(len(xlabels))
    train_data = [dataset_dict[key]["train2id"] for key in dataset_dict]
    valid_data = [dataset_dict[key]["valid2id"] for key in dataset_dict]
    test_data = [dataset_dict[key]["test2id"] for key in dataset_dict]
    # 绘制图形
    width=0.33
    tpv = 8000
    plt.figure(figsize=(15,12))
    plt.bar(x_value-width, train_data, width=width, color='r', align='center', label='train')
    for i,v in enumerate(train_data): plt.text(i-width,v+tpv, str(v), ha='center',fontsize=6)
    plt.bar(x_value, valid_data, width=width, color='g', align='center', label='valid')
    for i,v in enumerate(valid_data): plt.text(i,v+tpv, str(v), ha='center',fontsize=6)
    plt.bar(x_value+width, test_data, width=width, color='b', align='center', label='test')
    for i,v in enumerate(test_data): plt.text(i+width,v+tpv, str(v), ha='center',fontsize=6)
    # 添加标签
    plt.xlabel("Dataset Names",size=12)
    plt.ylabel("Number",size=12)
    plt.xticks(x_value, xlabels) # 设置X轴刻度标签
    plt.legend()
    plt.savefig("dataset.jpg")
    plt.close()
    
if __name__ == "__main__":
    main()

