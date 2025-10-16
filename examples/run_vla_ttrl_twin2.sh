#!/bin/bash
# VLA-TTRL Training Script for SimpleVLA-RL on RoboTwin 2.0
# This script demonstrates how to enable VLA-TTRL for RoboTwin 2.0 tasks

set -x

export NCCL_DEBUG=WARN 
export WANDB_API_KEY='YOUR WANDB KEY'
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=true
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export ROBOT_PLATFORM=ALOHA

PROJECT_NAME='SimpleVLA-RL-TTRL'
EXPERIMENT_NAME='vla_ttrl_twin2_votes5_samples3_test'
SFT_MODEL_PATH="YOUR SFT_MODEL_PATH"
CKPT_PATH="THE PATH YOU WANT TO SAVE YOUR CKPT"

# Currently Supported DATASET_NAME tasks for robotwin2.0 can be found at examples/robotwin2_tasks_info.txt
DATASET_NAME=click_bell  # Replace with your chosen task
TRAJ_MINI_BATCH_SIZE=8   # Check examples/robotwin2_tasks_info.txt for specific values
VLA_NAME="openvla-oft"
NUM_GPUS=8
NUM_NODES=1 
ALIGN_PATH="YOUR PATH TO SimpleVLA-RL/align.json"

# Overwrite VLA checkpoint utilities  
bash examples/overwrite_vla_ckpt_utils.sh $SFT_MODEL_PATH 

# Run VLA-TTRL training on RoboTwin 2.0
HYDRA_FULL_ERROR=1 python -u -m verl.trainer.main_ppo \
    --config-name=ppo_trainer_vla_ttrl \
    data.task_suite_name=robotwin2_$DATASET_NAME \
    data.num_trials_per_task=1000 \
    data.n_samples=3 \
    data.filter_accuracy=True \
    data.accuracy_lower_bound=0.1 \
    data.accuracy_upper_bound=0.9 \
    data.oversample_factor=1 \
    data.train_batch_size=32 \
    data.val_batch_size=256 \
    data.max_prompt_length=256 \
    data.max_response_length=128 \
    \
    actor_rollout_ref.model.path=$SFT_MODEL_PATH \
    actor_rollout_ref.model.vla=$VLA_NAME \
    actor_rollout_ref.model.action_token_len=14 \
    actor_rollout_ref.model.action_chunks_len=25 \
    actor_rollout_ref.model.resume=False \
    actor_rollout_ref.actor.optim.lr=5e-6 \
    actor_rollout_ref.actor.optim.warmup_style=constant \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size=$NUM_GPUS \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.grad_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.grad_clip=1 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.num_images_in_input=1 \
    actor_rollout_ref.actor.traj_mini_batch_size=$TRAJ_MINI_BATCH_SIZE \
    \
    actor_rollout_ref.rollout.vla=$VLA_NAME \
    actor_rollout_ref.rollout.action_chunks_len=25 \
    actor_rollout_ref.rollout.task_suite_name=robotwin2_$DATASET_NAME \
    actor_rollout_ref.rollout.experiment_name=$EXPERIMENT_NAME \
    actor_rollout_ref.rollout.unnorm_key=robotwin2 \
    actor_rollout_ref.rollout.twin2_task_config=demo_randomized \
    actor_rollout_ref.rollout.twin2_ckpt_setting=demo_randomized \
    actor_rollout_ref.rollout.twin2_instruction_type=unseen \
    actor_rollout_ref.rollout.micro_batch_size=1 \
    actor_rollout_ref.rollout.val_micro_batch_size=8 \
    actor_rollout_ref.rollout.num_images_in_input=1 \
    actor_rollout_ref.rollout.n=5 \
    \
    critic.model.action_token_len=14 \
    critic.model.action_chunks_len=25 \
    critic.model.path=$SFT_MODEL_PATH \
    critic.optim.lr=1e-5 \
    \
    trainer.default_hdfs_dir=null \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.save_freq=5 \
    trainer.logging_freq=1 \
    trainer.eval_freq=5 \
    trainer.test_freq=50 \
    trainer.max_epochs=200 \
    trainer.save_dir=$CKPT_PATH \
    trainer.nnodes=$NUM_NODES \
    trainer.nproc_per_node=$NUM_GPUS \
    trainer.save_tokenizer=True \
    trainer.load_checkpoint=False \
    trainer.rollout_data_dir=/tmp \
    \
    +vla_ttrl.enable=True \
    +vla_ttrl.n_votes_per_prompt=5 \
    +vla_ttrl.n_samples_per_prompt=3 \
    +vla_ttrl.min_confidence_ratio=0.5 \
    +vla_ttrl.use_env_fallback=True \
    +vla_ttrl.log_detailed_metrics=True \
    \
    ++align_path=$ALIGN_PATH