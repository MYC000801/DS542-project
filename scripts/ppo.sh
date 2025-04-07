GPUS_PER_NODE=4
BATCH_SIZE=256
MINI_BATCH_SIZE=128 # each batch we do BATCH_SIZE / MINI_BATCH_SIZE updates
MICRO_BATCH_SIZEPER_DEVICE=2 # reduce to reduce memory but slower
MODEL_SIZE="1.5"
MAX_PROMPT_LENGTH=256
MAX_RESPONSE_LENGTH=1024
DATASET="gsm8k"
EPOCH=50

######################
### Set Slurm Job
######################
export VLLM_ATTENTION_BACKEND=XFORMERS

kl_coef=(3e-4)
filter_times=(1)

for filter_time in "${filter_times[@]}"; do
    for kl in "${kl_coef[@]}"; do
        python3 -m verl.trainer.main_ppo \
                    data.train_files=/projectnb/rlhf/mingyuc/TinyZero_old/data/${DATASET}/train.parquet \
                    data.val_files=/projectnb/rlhf/mingyuc/TinyZero_old/data/${DATASET}/test.parquet \
                    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
                    data.max_response_length=${MAX_RESPONSE_LENGTH} \
                    data.train_batch_size=$BATCH_SIZE \
                    data.val_batch_size=500 \
                    data.filter_time=${filter_time} \
                    actor_rollout_ref.model.path=Qwen/Qwen2.5-${MODEL_SIZE}B \
                    actor_rollout_ref.actor.optim.lr=1e-6 \
                    actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE \
                    actor_rollout_ref.actor.ppo_micro_batch_size=$(expr $GPUS_PER_NODE \* $MICRO_BATCH_SIZEPER_DEVICE) \
                    actor_rollout_ref.rollout.log_prob_micro_batch_size=$(expr $GPUS_PER_NODE \* $MICRO_BATCH_SIZEPER_DEVICE) \
                    actor_rollout_ref.rollout.tensor_model_parallel_size=$GPUS_PER_NODE \
                    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
                    actor_rollout_ref.ref.log_prob_micro_batch_size=$(expr $GPUS_PER_NODE \* $MICRO_BATCH_SIZEPER_DEVICE) \
                    critic.optim.lr=1e-5 \
                    critic.model.path=Qwen/Qwen2.5-${MODEL_SIZE}B \
                    critic.ppo_micro_batch_size=$(expr $GPUS_PER_NODE \* $MICRO_BATCH_SIZEPER_DEVICE) \
                    algorithm.kl_ctrl.kl_coef=${kl} \
                    trainer.logger=['wandb'] \
                    +trainer.val_before_train=False \
                    trainer.default_hdfs_dir=null \
                    trainer.n_gpus_per_node=$GPUS_PER_NODE \
                    trainer.nnodes=1 \
                    trainer.save_freq=-1 \
                    trainer.test_freq=50 \
                    trainer.project_name=${DATASET} \
                    trainer.experiment_name=qwen2.5-${MODEL_SIZE}b-ppo-kl-${kl}-regular \
                    trainer.total_epochs=${EPOCH} # 2>&1 | tee verl_demo.log
    done
done

# python3 -m verl.trainer.main_ppo \
#                 data.train_files=/n/holylabs/LABS/kdbrantley_lab/Lab/zhaolin/reasoning/data/gsm8k/train.parquet \
#                 data.val_files=/n/holylabs/LABS/kdbrantley_lab/Lab/zhaolin/reasoning/data/gsm8k/test.parquet \
#                 data.max_prompt_length=256 \
#                 data.max_response_length=1024 \
#                 data.train_batch_size=256 \
#                 data.val_batch_size=500 \
#                 data.filter_time=4 \
#                 actor_rollout_ref.model.path=Qwen/Qwen2.5-1.5B \
#                 actor_rollout_ref.actor.optim.lr=1e-6 \
#                 actor_rollout_ref.actor.ppo_mini_batch_size=128 \
#                 actor_rollout_ref.actor.ppo_micro_batch_size=8 \
#                 actor_rollout_ref.rollout.log_prob_micro_batch_size=8 \
#                 actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
#                 actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
#                 actor_rollout_ref.ref.log_prob_micro_batch_size=8 \
#                 critic.optim.lr=1e-5 \
#                 critic.model.path=Qwen/Qwen2.5-1.5B \
#                 critic.ppo_micro_batch_size=8 \
#                 algorithm.kl_ctrl.kl_coef=3e-4 \
#                 trainer.logger=['wandb'] \
#                 +trainer.val_before_train=False \
#                 trainer.default_hdfs_dir=null \
#                 trainer.n_gpus_per_node=4 \
#                 trainer.nnodes=1 \
#                 trainer.save_freq=-1 \
#                 trainer.test_freq=50 \
#                 trainer.project_name=gsm8k \
#                 trainer.experiment_name=test \
#                 trainer.total_epochs=25 2>&1 | tee verl_demo.log