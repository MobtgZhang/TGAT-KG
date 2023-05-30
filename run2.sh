datasets=("YAGO3-10" "fb15k-237" "fb15k" "KG20C" "wn18" "wn18rr")
# dim_nums=(10 15 20 25 30 35 40 45 50)
dim_nums=(30)
alpha_list=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
beta_list=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
layers_list=(20 40 60 80 100 120 140 160 180 200)
batch_size=1024
leanring_rate=0.02
for data_name in ${datasets[@]}
do
    for number in ${dim_nums[@]}
    do  
        for alpha in ${alpha_list[@]}
        do
            python train.py --dataset $data_name --batch-size $batch_size --sigmoid --learning-rate $leanring_rate --alpha $alpha
        done
    done
done
