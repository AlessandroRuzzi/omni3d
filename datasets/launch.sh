batch_size=1
max_dataset_size=100000
dataset_mode='behave_img'
trunc_thres=0.2
cat='all'

python pre_process_behave.py --batch_size ${batch_size} --dataset_mode ${dataset_mode} --cat ${cat} --max_dataset_size ${max_dataset_size} --trunc_thres ${trunc_thres} 
                --display_freq ${display_freq} --print_freq ${print_freq} --save_epoch_freq ${save_epoch_freq} --debug ${debug}  --nThreads 2