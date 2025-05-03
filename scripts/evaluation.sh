set -x


n_samples=1
DATASET="gsm8k"
data_path=/projectnb/rlhf/mingyuc/TinyZero_old/data/${DATASET}/deepseek_offline_gen_train_${n_samples}_samples.parquet 


python3 -m verl.trainer.main_eval \
    data.path=$data_path 
