#!/bin/bash
# VLA-TTRL训练脚本 - 测试版本
# 配置: 2个prompts × 3条投票轨迹 = 6条轨迹候选 (极小batch测试修复)

set -x

export MUJOCO_GL=osmesa
export NCCL_DEBUG=WARN 
export WANDB_API_KEY='e24cdb09c08933b35e62651afdeac3d18a99ff30'
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=true
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export ROBOT_PLATFORM=LIBERO

PROJECT_NAME='SimpleVLA-TTRL-Test'
EXPERIMENT_NAME='vla_ttrl_2prompts_3votes_test' 
SFT_MODEL_PATH="/inspire/hdd/project/realtimedecisionmaking/yishenghong-CZXS25230064/sychen/vla_models/openvla-oft-traj1-libero_10"
CKPT_PATH="/inspire/hdd/project/realtimedecisionmaking/yishenghong-CZXS25230064/sychen/SimpleVLA-TTRL/checkpoint"
DATASET_NAME="libero_10"
VLA_NAME="openvla-oft"
NUM_GPUS=8
NUM_NODES=1 
ALIGN_PATH="/inspire/hdd/project/realtimedecisionmaking/yishenghong-CZXS25230064/sychen/SimpleVLA-TTRL/align.json"

bash examples/overwrite_vla_ckpt_utils.sh $SFT_MODEL_PATH 

HYDRA_FULL_ERROR=1 python -u -m verl.trainer.main_ppo \
    --config-name=ppo_trainer_vla_ttrl \
    data.task_suite_name=$DATASET_NAME \
    data.num_trials_per_task=50 \
    data.n_samples=1 \
    data.filter_accuracy=True \
    data.accuracy_lower_bound=0.1 \
    data.accuracy_upper_bound=0.9 \
    data.oversample_factor=1 \
    data.train_batch_size=64 \
    data.val_batch_size=96 \
    data.max_prompt_length=256 \
    data.max_response_length=128 \
    \
    actor_rollout_ref.model.path=$SFT_MODEL_PATH \
    actor_rollout_ref.model.vla=$VLA_NAME \
    actor_rollout_ref.model.action_token_len=7 \
    actor_rollout_ref.model.action_chunks_len=8 \
    actor_rollout_ref.actor.optim.lr=3e-6 \
    actor_rollout_ref.actor.optim.warmup_style=constant \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size=4 \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.grad_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.grad_clip=1 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.num_images_in_input=1 \
    actor_rollout_ref.actor.traj_mini_batch_size=16 \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.entropy_coeff=0. \
    \
    actor_rollout_ref.rollout.num_images_in_input=1 \
    actor_rollout_ref.rollout.use_proprio=False \
    actor_rollout_ref.rollout.val_micro_batch_size=8 \
    actor_rollout_ref.rollout.temperature=1.6 \
    actor_rollout_ref.rollout.experiment_name=$EXPERIMENT_NAME \
    actor_rollout_ref.rollout.micro_batch_size=1 \
    actor_rollout_ref.rollout.unnorm_key=$DATASET_NAME \
    actor_rollout_ref.rollout.model_family=openvla \
    actor_rollout_ref.rollout.task_suite_name=$DATASET_NAME \
    actor_rollout_ref.rollout.num_steps_wait=10 \
    actor_rollout_ref.rollout.pretrained_checkpoint=$SFT_MODEL_PATH \
    actor_rollout_ref.rollout.center_crop=True \
    actor_rollout_ref.rollout.max_prompt_length=512 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=hf \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    actor_rollout_ref.rollout.n=1 \
    \
    actor_rollout_ref.ref.log_prob_micro_batch_size=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    \
    critic.model.path=$SFT_MODEL_PATH \
    critic.optim.lr=1e-5 \
    \
    algorithm.kl_ctrl.kl_coef=0.00 \
    algorithm.adv_estimator=grpo \
    algorithm.adv_params.verifier_gamma=1.0 \
    algorithm.adv_params.reward_model_gamma=1.0 \
    \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.default_local_dir=$CKPT_PATH/$PROJECT_NAME/$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=$NUM_GPUS \
    trainer.nnodes=$NUM_NODES \
    trainer.save_freq=10 \
    trainer.test_freq=3 \
    trainer.total_epochs=50 \
    trainer.val_only=False \
    trainer.runtime_env=$ALIGN_PATH \
    trainer.wandb_mode=offline \
    trainer.val_before_train=False \
    \
    +vla_ttrl.enable=True \
    +vla_ttrl.n_votes_per_prompt=3 \
    +vla_ttrl.n_samples_per_prompt=1 \
    +vla_ttrl.rollout_batch_size=2 \
    +vla_ttrl.min_confidence_ratio=0.6 \
    +vla_ttrl.use_env_fallback=True \
    +vla_ttrl.log_detailed_metrics=True