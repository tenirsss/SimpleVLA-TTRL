# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
VLA-TTRL Utils: Adapting TTRL methodology for Vision-Language-Action models.
This module implements majority voting for VLA task completion outcomes.
"""

import numpy as np
from collections import Counter
from typing import List, Dict, Any, Tuple


def _vla_majority_vote(task_outcomes: List[bool]) -> Tuple[bool, float]:
    """
    Perform majority voting on VLA task completion outcomes.
    
    Args:
        task_outcomes: List of boolean outcomes (True=success, False=failure)
    
    Returns:
        Tuple of (majority_outcome, confidence_ratio)
    """
    assert len(task_outcomes) > 0, "Need at least one outcome for voting"
    
    # Count successes and failures
    success_count = sum(task_outcomes)
    failure_count = len(task_outcomes) - success_count
    
    # Determine majority outcome
    majority_outcome = success_count > failure_count
    majority_count = max(success_count, failure_count)
    confidence_ratio = majority_count / len(task_outcomes)
    
    return majority_outcome, confidence_ratio


def _batch_vla_majority_vote(task_outcomes: List[bool], n: int) -> Tuple[List[bool], List[float]]:
    """
    Perform batch majority voting for VLA tasks.
    
    Args:
        task_outcomes: List of task completion outcomes for all prompts
        n: Number of rollouts per prompt
    
    Returns:
        Tuple of (majority_outcomes_list, confidence_ratios_list)
    """
    majority_outcomes_list = []
    confidence_ratios_list = []
    
    assert len(task_outcomes) % n == 0, f"Outcomes length {len(task_outcomes)} not divisible by {n}"
    n_prompts = len(task_outcomes) // n
    
    for i in range(n_prompts):
        prompt_outcomes = task_outcomes[i * n:(i + 1) * n]
        majority_outcome, confidence_ratio = _vla_majority_vote(prompt_outcomes)
        majority_outcomes_list.append(majority_outcome)
        confidence_ratios_list.append(confidence_ratio)
    
    return majority_outcomes_list, confidence_ratios_list


def apply_vla_ttrl_gt(batch, task_outcomes: List[bool], n: int):
    """
    Apply TTRL methodology to VLA tasks by replacing ground truth with majority voting results.
    
    Args:
        batch: The training batch
        task_outcomes: List of task completion outcomes from rollouts
        n: Number of rollouts per prompt
    
    Returns:
        Modified batch with majority voting ground truth
    """
    assert len(task_outcomes) % n == 0, "Task outcomes length must be divisible by n"
    num_prompts = len(task_outcomes) // n
    assert len(batch) == num_prompts, "Batch length must equal number of prompts"
    
    majority_outcomes, confidence_ratios = _batch_vla_majority_vote(task_outcomes, n)
    
    # Store original ground truth and apply majority voting results
    for i in range(num_prompts):
        data_item = batch[i]
        
        # Store original environment reward (0/1)
        if "reward_model" not in data_item.non_tensor_batch:
            data_item.non_tensor_batch["reward_model"] = {}
        
        original_gt = data_item.non_tensor_batch.get("task_success", majority_outcomes[i])
        data_item.non_tensor_batch["reward_model"]["original_gt"] = original_gt
        
        # Apply majority voting result as new ground truth
        majority_gt = majority_outcomes[i]
        data_item.non_tensor_batch["reward_model"]["ground_truth"] = majority_gt
        data_item.non_tensor_batch["reward_model"]["majority_gt"] = majority_gt
        data_item.non_tensor_batch["task_success"] = majority_gt
    
    # Store confidence ratios for metrics
    batch.non_tensor_batch["majority_confidence_ratios"] = np.array(confidence_ratios, dtype=float)
    
    return batch


def apply_vla_original_gt(batch):
    """
    Restore original ground truth for VLA tasks.
    
    Args:
        batch: The training batch to restore
    
    Returns:
        Batch with original ground truth restored
    """
    for data_item in batch:
        if "reward_model" in data_item.non_tensor_batch and "original_gt" in data_item.non_tensor_batch["reward_model"]:
            original_gt = data_item.non_tensor_batch["reward_model"]["original_gt"]
            data_item.non_tensor_batch["reward_model"]["ground_truth"] = original_gt
            data_item.non_tensor_batch["task_success"] = original_gt
    
    return batch


def _compute_vla_ttrl_metrics_per_prompt(
    majority_rewards: List[float],
    gt_rewards: List[float], 
    majority_outcome: bool,
    gt_outcome: bool
) -> Dict[str, float]:
    """
    Compute TTRL metrics for a single prompt in VLA setting.
    
    Args:
        majority_rewards: Rewards based on majority voting
        gt_rewards: Original environment rewards
        majority_outcome: Majority voting outcome (True/False)
        gt_outcome: Original environment outcome (True/False)
    
    Returns:
        Dictionary of computed metrics
    """
    # Outcome accuracy: whether majority vote matches environment truth
    outcome_accuracy = 1.0 if majority_outcome == gt_outcome else 0.0
    
    # Reward accuracy: how many individual rewards match
    reward_matches = sum(1 for maj_r, gt_r in zip(majority_rewards, gt_rewards) if maj_r == gt_r)
    reward_accuracy = reward_matches / len(majority_rewards)
    
    # Average rewards
    avg_majority_reward = sum(majority_rewards) / len(majority_rewards)
    avg_gt_reward = sum(gt_rewards) / len(gt_rewards)
    
    # Success rate metrics
    success_rate_majority = 1.0 if majority_outcome else 0.0
    success_rate_gt = 1.0 if gt_outcome else 0.0
    
    return {
        "outcome_accuracy": outcome_accuracy,
        "reward_accuracy": reward_accuracy,
        "avg_majority_reward": avg_majority_reward,
        "avg_gt_reward": avg_gt_reward,
        "success_rate_majority": success_rate_majority,
        "success_rate_gt": success_rate_gt,
        f"pass@{len(majority_rewards)}": success_rate_gt,
    }


def _batch_compute_vla_ttrl_metrics(
    majority_rewards: List[float],
    gt_rewards: List[float],
    majority_outcomes: List[bool],
    gt_outcomes: List[bool],
    n: int
) -> Dict[str, float]:
    """
    Compute VLA-TTRL metrics for a batch of prompts.
    
    Args:
        majority_rewards: All majority-based rewards
        gt_rewards: All ground truth rewards  
        majority_outcomes: All majority outcomes
        gt_outcomes: All ground truth outcomes
        n: Number of rollouts per prompt
        
    Returns:
        Averaged metrics across all prompts
    """
    assert len(majority_rewards) == len(gt_rewards) == len(majority_outcomes) == len(gt_outcomes)
    assert len(majority_rewards) % n == 0
    
    n_prompts = len(majority_rewards) // n
    all_metrics = []
    
    for i in range(n_prompts):
        start_idx = i * n
        end_idx = (i + 1) * n
        
        prompt_majority_rewards = majority_rewards[start_idx:end_idx]
        prompt_gt_rewards = gt_rewards[start_idx:end_idx]
        
        # For VLA, outcomes should be consistent within each prompt's rollouts
        # We take the first occurrence as the outcome for that prompt
        prompt_majority_outcome = majority_outcomes[start_idx]
        prompt_gt_outcome = gt_outcomes[start_idx]
        
        prompt_metrics = _compute_vla_ttrl_metrics_per_prompt(
            prompt_majority_rewards, prompt_gt_rewards,
            prompt_majority_outcome, prompt_gt_outcome
        )
        all_metrics.append(prompt_metrics)
    
    # Average metrics across all prompts
    if not all_metrics:
        return {}
    
    averaged_metrics = {}
    for key in all_metrics[0].keys():
        averaged_metrics[key] = sum(metric[key] for metric in all_metrics) / len(all_metrics)
    
    return averaged_metrics


def compute_vla_ttrl_metrics(batch, n: int) -> Dict[str, float]:
    """
    Compute VLA-TTRL metrics from a training batch.
    
    Args:
        batch: Training batch with both majority and original outcomes
        n: Number of rollouts per prompt
    
    Returns:
        Dictionary of computed metrics
    """
    assert len(batch) % n == 0, "Batch length must be divisible by n"
    
    # Sort batch by index to ensure correct ordering
    try:
        idx = sorted(range(len(batch)), key=lambda x: batch[x].non_tensor_batch["extra_info"]["index"])
    except KeyError:
        # If no index available, assume batch is already properly ordered
        idx = list(range(len(batch)))
    
    majority_rewards = []
    gt_rewards = []  
    majority_outcomes = []
    gt_outcomes = []
    
    for i in range(len(batch)):
        data_item = batch[idx[i]]
        
        # Extract rewards (token-level scores summed)
        majority_reward = data_item.batch["token_level_scores"].sum().item()
        majority_rewards.append(majority_reward)
        
        if "token_level_scores_original" in data_item.batch:
            gt_reward = data_item.batch["token_level_scores_original"].sum().item() 
        else:
            gt_reward = majority_reward  # Fallback if no original scores
        gt_rewards.append(gt_reward)
        
        # Extract outcomes
        majority_outcome = bool(data_item.non_tensor_batch["reward_model"]["majority_gt"])
        majority_outcomes.append(majority_outcome)
        
        gt_outcome = bool(data_item.non_tensor_batch["reward_model"]["original_gt"])
        gt_outcomes.append(gt_outcome)
    
    # Compute metrics
    vla_ttrl_metrics = _batch_compute_vla_ttrl_metrics(
        majority_rewards, gt_rewards, majority_outcomes, gt_outcomes, n
    )
    
    # Add confidence ratio if available
    if "majority_confidence_ratios" in batch.non_tensor_batch:
        confidence_ratios = batch.non_tensor_batch["majority_confidence_ratios"]
        vla_ttrl_metrics["avg_confidence_ratio"] = float(np.mean(confidence_ratios))
    
    return vla_ttrl_metrics


def process_vla_rollout_batch(batch, rollout_outputs, n_votes_per_prompt: int, n_samples_per_prompt: int):
    """
    Process VLA rollout batch with proper batch size handling.
    
    This handles the case where:
    - Input: BATCH_SIZE prompts (e.g., 192)  
    - Generation: Each prompt → n_votes_per_prompt rollouts (e.g., 3)
    - Total: BATCH_SIZE × n_votes_per_prompt rollouts (e.g., 192 × 3 = 576)
    - Selection: Keep n_samples_per_prompt per prompt for training
    
    Args:
        batch: Input batch with BATCH_SIZE prompts
        rollout_outputs: List of rollout outputs (BATCH_SIZE × n_votes_per_prompt)
        n_votes_per_prompt: Number of rollouts per prompt for voting
        n_samples_per_prompt: Number of rollouts per prompt for training
        
    Returns:
        Tuple of (processed_batch, selected_rollouts, vla_ttrl_info)
    """
    batch_size = len(batch)
    total_rollouts = len(rollout_outputs)
    expected_rollouts = batch_size * n_votes_per_prompt
    
    assert total_rollouts == expected_rollouts, (
        f"Rollout count mismatch: expected {expected_rollouts} "
        f"({batch_size} prompts × {n_votes_per_prompt} votes), got {total_rollouts}"
    )
    
    print(f"VLA-TTRL Batch Processing:")
    print(f"  Input prompts: {batch_size}")
    print(f"  Votes per prompt: {n_votes_per_prompt}")
    print(f"  Total rollouts: {total_rollouts}")
    print(f"  Samples per prompt for training: {n_samples_per_prompt}")
    
    # Extract task outcomes from all rollouts
    task_outcomes = []
    for output_item in rollout_outputs:
        # Extract task success - handle different possible formats
        task_success = False
        if hasattr(output_item, 'batch'):
            acc_field = output_item.batch.get('acc', None)
            if acc_field is not None:
                if hasattr(acc_field, '__getitem__') and len(acc_field) > 0:
                    task_success = acc_field[0]
                    if hasattr(task_success, 'item'):
                        task_success = task_success.item()
        
        task_outcomes.append(bool(task_success))
    
    # Apply majority voting to determine pseudo ground truth
    processed_batch = apply_vla_ttrl_gt(batch, task_outcomes, n_votes_per_prompt)
    
    # Select rollouts for training
    selected_rollouts = select_top_k_per_prompt_vla(rollout_outputs, n_votes_per_prompt, n_samples_per_prompt)
    
    # Compute voting statistics
    majority_outcomes, confidence_ratios = _batch_vla_majority_vote(task_outcomes, n_votes_per_prompt)
    
    vla_ttrl_info = {
        'batch_size': batch_size,
        'n_votes_per_prompt': n_votes_per_prompt,
        'n_samples_per_prompt': n_samples_per_prompt,
        'total_rollouts': total_rollouts,
        'selected_rollouts': len(selected_rollouts),
        'avg_confidence': np.mean(confidence_ratios),
        'majority_success_rate': np.mean(majority_outcomes),
        'original_success_rate': np.mean(task_outcomes),
    }
    
    return processed_batch, selected_rollouts, vla_ttrl_info


def select_top_k_per_prompt_vla(rollout_outputs, n_votes: int, n_samples: int):
    """
    Select top-k rollouts per prompt for VLA tasks.
    This is adapted from the original TTRL implementation for VLA use.
    
    Args:
        rollout_outputs: List of rollout outputs
        n_votes: Number of votes per prompt 
        n_samples: Number of samples to keep per prompt
        
    Returns:
        Filtered rollout outputs
    """
    if n_votes <= n_samples:
        return rollout_outputs
    
    assert len(rollout_outputs) % n_votes == 0, f"Rollout count {len(rollout_outputs)} not divisible by {n_votes}"
    n_prompts = len(rollout_outputs) // n_votes
    
    selected_outputs = []
    for i in range(n_prompts):
        start_idx = i * n_votes
        end_idx = (i + 1) * n_votes
        prompt_outputs = rollout_outputs[start_idx:end_idx]
        
        # For VLA, we can select based on task completion confidence
        # For now, just take the first n_samples (can be improved with better selection criteria)
        selected_outputs.extend(prompt_outputs[:n_samples])
    
    return selected_outputs