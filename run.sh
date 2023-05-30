datasets=("fb15k-237" "fb15k" "KG20C" "wn18" "wn18rr" "YAGO3-10")
models_set=("TransD" "KGATConv" "TransE" "TransH" "TransR"  "RGCN")
batch_size=5120
for data_name in ${datasets[@]}
do
    for model_name in ${models_set[@]}
    do
        python main.py --dataset $data_name --batch-size $batch_size --model-name $model_name --sigmoid
    done
done
