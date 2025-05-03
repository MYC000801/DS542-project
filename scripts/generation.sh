DATASET="gsm8k"

python3 -m verl.trainer.main_generation \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=2 \
    data.path=/projectnb/rlhf/mingyuc/TinyZero_old/data/${DATASET}/train.parquet \
    data.prompt_key=prompt \
    data.n_samples=1 \
    data.output_path=/projectnb/rlhf/mingyuc/TinyZero_old/data/${DATASET}/deepseek_gen_train.parquet \
    model.path=Qwen/Qwen2.5-1.5B \
    +model.trust_remote_code=True \
    rollout.temperature=1.0 \
    rollout.top_k=50 \
    rollout.top_p=0.7 \
    rollout.prompt_length=256 \
    rollout.response_length=1024 \
    rollout.tensor_model_parallel_size=1 \
    rollout.gpu_memory_utilization=0.8
