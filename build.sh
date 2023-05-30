datasets=("fb15k" "fb15k-237" "KG20C" "wn18" "wn18rr" "YAGO3-10")

for data_name in ${datasets[@]}
do
    python build.py --dataset $data_name
    rm -rf log
done
