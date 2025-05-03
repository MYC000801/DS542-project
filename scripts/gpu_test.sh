set -x

export VLLM_ATTENTION_BACKEND=XFORMERS


GPUS_PER_NODE=4
MINI_BATCH_SIZE=256 # number of prompts for each update; each batch we do BATCH_SIZE / MINI_BATCH_SIZE updates
mbsd=(2)
DATASET="math"
batch_sizes=1024

model_sizes=("1.5")
kl_coef=(0.001)
biases=(0.2)


for MICRO_BATCH_SIZEPER_DEVICE in "${mbsd[@]}"; do
    for bias in "${biases[@]}"; do
        for kl in "${kl_coef[@]}"; do
            for model_size in "${model_sizes[@]}"; do
                python3 -m verl.trainer.main_ppo \
                    algorithm.adv_estimator=grpo \
                    data.train_files=/projectnb/rlhf/mingyuc/verl/data/${DATASET}/train.parquet \
                    data.val_files=/projectnb/rlhf/mingyuc/verl/data/${DATASET}/test.parquet \
                    data.train_batch_size=$batch_sizes \
                    data.val_batch_size=1312 \
                    data.max_prompt_length=256 \
                    data.max_response_length=1024 \
                    actor_rollout_ref.model.path=Qwen/Qwen2.5-${model_size}B \
                    actor_rollout_ref.actor.optim.lr=1e-6 \
                    actor_rollout_ref.model.use_remove_padding=True \
                    actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE  \
                    actor_rollout_ref.actor.ppo_micro_batch_size=$(expr $GPUS_PER_NODE \* $MICRO_BATCH_SIZEPER_DEVICE) \
                    actor_rollout_ref.actor.use_kl_loss=True \
                    actor_rollout_ref.actor.kl_loss_coef=${kl} \
                    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
                    actor_rollout_ref.actor.bias=${bias} \
                    actor_rollout_ref.model.enable_gradient_checkpointing=True \
                    actor_rollout_ref.actor.fsdp_config.param_offload=False \
                    actor_rollout_ref.actor.fsdp_config.grad_offload=False \
                    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
                    actor_rollout_ref.rollout.log_prob_micro_batch_size=$(expr $GPUS_PER_NODE \* $MICRO_BATCH_SIZEPER_DEVICE) \
                    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
                    actor_rollout_ref.rollout.name=vllm \
                    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
                    actor_rollout_ref.rollout.n=4 \
                    actor_rollout_ref.ref.log_prob_micro_batch_size=$(expr $GPUS_PER_NODE \* $MICRO_BATCH_SIZEPER_DEVICE) \
                    actor_rollout_ref.ref.fsdp_config.param_offload=True \
                    algorithm.kl_ctrl.kl_coef=${kl} \
                    trainer.critic_warmup=0 \
                    trainer.logger=['wandb'] \
                    +trainer.val_before_train=False \
                    trainer.project_name="verl_grpo_${DATASET}_learning_dynamic" \
                    trainer.experiment_name="qwen2_${model_size}b_grpo_kl_${kl}_bs_${batch_sizes}_mbs_${MINI_BATCH_SIZE}_n_4_advantage_minus_${bias}_per_response_norm" \
                    trainer.n_gpus_per_node=$GPUS_PER_NODE \
                    trainer.nnodes=1 \
                    trainer.save_freq=-1 \
                    trainer.test_freq=5 \
                    trainer.total_epochs=25  # 2>&1 | tee verl_demo.log
            done
        done
    done
done