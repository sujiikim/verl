#!/bin/bash
# Split-placement GRPO training for EvoCUA-8B
#
# GPU layout (Exclusive_Process mode safe):
#   GPU 0,1  →  actor_rollout_pool  (actor + vLLM rollout, gen_tp=2)
#   GPU 2,3  →  ref_pool            (ref policy, separate OS process group)
#
# This avoids CUDA Exclusive_Process conflicts that occur when actor and ref
# workers from different Ray process groups compete for the same GPUs.
#
# Key differences from run_evocua-8b.sh:
#   - Uses verl.trainer.main_ppo_split instead of verl.trainer.main_ppo
#   - gen_tp=2  (vLLM tensor-parallel size matches actor pool = 2 GPUs)
#   - trainer.n_gpus_per_node=4  (total GPUs; split logic divides by 2)

export CUDA_VISIBLE_DEVICES=0,1,2,3

set -x

project_name='GRPO-EvoCUA'
exp_name='GRPO-EvoCUA-8B-H100-split'
gen_tp=2          # vLLM TP = actor pool size (4 total // 2 = 2)
total_gpus=4      # total GPUs across all pools
sp_size=1
ENGINE=${1:-vllm}
RAY_DATA_HOME=${RAY_DATA_HOME:-"/c2/kangsan/verl"}
MODEL_PATH='meituan/EvoCUA-8B-20260105'
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"}
TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/dataset/agentnet_vscode_train_hist2.parquet"}
TEST_FILE=${TEST_FILE:-"${RAY_DATA_HOME}/dataset/agentnet_vscode_test_hist2.parquet"}

python3 -m verl.trainer.main_ppo_split \
    algorithm.adv_estimator=grpo \
    trainer.val_before_train=False \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    +data.num_workers=0 \
    data.train_batch_size=16 \
    data.max_prompt_length=20000 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.image_key=images \
    data.shuffle=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=40000 \
    actor_rollout_ref.rollout.max_model_len=22048 \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.optim.lr=3e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.lora_alpha=64 \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.model.exclude_modules='.*visual.*' \
    +actor_rollout_ref.model.lora.merge=False \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.name=$ENGINE \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=5000 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.max_num_seqs=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=${total_gpus} \
    trainer.nnodes=1 \
    +trainer.ray_init.num_cpus=32 \
    trainer.default_local_dir=${CKPTS_DIR} \
    trainer.save_freq=50 \
    trainer.test_freq=-1 \
    +trainer.log_freq=1 \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.max_critic_ckpt_to_keep=1 \
    trainer.total_training_steps=1000
