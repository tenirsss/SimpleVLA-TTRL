#!/usr/bin/env python3
"""
VLA-TTRL演示脚本

这个脚本演示了VLA-TTRL的核心功能，包括：
1. 多轨迹生成和多数投票
2. 伪ground truth生成
3. 指标计算

运行方式:
python demo_vla_ttrl.py
"""

import numpy as np
import torch
from typing import List, Dict, Any
from collections import Counter

# 模拟导入VLA-TTRL工具
# 在实际使用中，这将是: from verl.trainer.ppo.vla_ttrl_utils import *

def simulate_vla_rollouts(n_prompts: int, n_rollouts_per_prompt: int, 
                         success_rate: float = 0.6) -> List[bool]:
    """
    模拟VLA rollouts的任务完成结果
    
    Args:
        n_prompts: 提示数量
        n_rollouts_per_prompt: 每个提示的rollout数量
        success_rate: 基础成功率
    
    Returns:
        任务完成状态列表 (True=成功, False=失败)
    """
    results = []
    np.random.seed(42)  # 为了可重现的结果
    
    for prompt_id in range(n_prompts):
        # 为每个prompt生成多个rollout结果
        # 添加一些随机性来模拟真实情况
        prompt_success_rate = success_rate + np.random.normal(0, 0.1)
        prompt_success_rate = np.clip(prompt_success_rate, 0.1, 0.9)
        
        for rollout_id in range(n_rollouts_per_prompt):
            is_success = np.random.random() < prompt_success_rate
            results.append(is_success)
    
    return results

def majority_vote_demo(task_outcomes: List[bool], n: int) -> Dict[str, Any]:
    """
    演示多数投票功能
    """
    print(f"\n=== 多数投票演示 ===")
    print(f"总rollouts数量: {len(task_outcomes)}")
    print(f"每个prompt的rollout数量: {n}")
    
    assert len(task_outcomes) % n == 0
    n_prompts = len(task_outcomes) // n
    
    majority_results = []
    confidence_scores = []
    
    for i in range(n_prompts):
        start_idx = i * n
        end_idx = (i + 1) * n
        prompt_outcomes = task_outcomes[start_idx:end_idx]
        
        # 计算多数投票结果
        success_count = sum(prompt_outcomes)
        majority_outcome = success_count > (n / 2)
        confidence = max(success_count, n - success_count) / n
        
        majority_results.append(majority_outcome)
        confidence_scores.append(confidence)
        
        print(f"Prompt {i+1}: {prompt_outcomes} -> 多数投票: {majority_outcome} (置信度: {confidence:.2f})")
    
    avg_confidence = np.mean(confidence_scores)
    success_rate_original = np.mean(task_outcomes)
    success_rate_voted = np.mean(majority_results)
    
    return {
        'majority_results': majority_results,
        'confidence_scores': confidence_scores,
        'avg_confidence': avg_confidence,
        'success_rate_original': success_rate_original,
        'success_rate_voted': success_rate_voted,
    }

def compute_metrics_demo(original_outcomes: List[bool], voted_outcomes: List[bool], 
                        confidence_scores: List[float]) -> Dict[str, float]:
    """
    演示指标计算
    """
    print(f"\n=== 指标计算演示 ===")
    
    # 计算准确性指标
    outcome_matches = [orig == voted for orig, voted in zip(original_outcomes, voted_outcomes)]
    outcome_accuracy = np.mean(outcome_matches)
    
    # 计算成功率指标
    original_success_rate = np.mean(original_outcomes)
    voted_success_rate = np.mean(voted_outcomes)
    
    # 计算置信度相关指标
    avg_confidence = np.mean(confidence_scores)
    high_confidence_ratio = np.mean([c >= 0.6 for c in confidence_scores])
    
    metrics = {
        'outcome_accuracy': outcome_accuracy,
        'original_success_rate': original_success_rate,
        'voted_success_rate': voted_success_rate,
        'avg_confidence': avg_confidence,
        'high_confidence_ratio': high_confidence_ratio,
        'vote_improvement': voted_success_rate - original_success_rate,
    }
    
    print("计算得到的指标:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    return metrics

def demonstrate_vla_ttrl():
    """
    完整的VLA-TTRL演示
    """
    print("🚀 VLA-TTRL 演示开始")
    print("=" * 50)
    
    # 设置参数
    n_prompts = 10          # 任务提示数量
    n_votes = 5             # 每个提示的投票rollout数量
    n_samples = 3           # 每个提示用于训练的rollout数量
    base_success_rate = 0.6 # 基础成功率
    
    print(f"配置参数:")
    print(f"  任务数量: {n_prompts}")
    print(f"  每任务投票rollouts: {n_votes}")
    print(f"  每任务训练rollouts: {n_samples}")
    print(f"  基础成功率: {base_success_rate}")
    
    # 步骤1: 模拟rollout生成
    print(f"\n🎯 步骤1: 模拟VLA rollout生成")
    task_outcomes = simulate_vla_rollouts(n_prompts, n_votes, base_success_rate)
    
    # 步骤2: 多数投票
    print(f"\n🗳️ 步骤2: 执行多数投票")
    vote_results = majority_vote_demo(task_outcomes, n_votes)
    
    # 步骤3: 模拟原始环境奖励(用于对比)
    print(f"\n🏆 步骤3: 生成原始环境奖励(对比基准)")
    # 假设原始环境有一定噪声
    original_outcomes = simulate_vla_rollouts(n_prompts, 1, base_success_rate * 0.9)
    
    # 步骤4: 计算指标
    print(f"\n📊 步骤4: 计算VLA-TTRL指标")
    metrics = compute_metrics_demo(
        original_outcomes,
        vote_results['majority_results'],
        vote_results['confidence_scores']
    )
    
    # 步骤5: 展示轨迹选择过程
    print(f"\n✂️ 步骤5: 轨迹选择演示")
    print(f"从{n_votes}个投票rollouts中选择{n_samples}个用于训练")
    
    selected_count = 0
    for i in range(n_prompts):
        start_idx = i * n_votes
        end_idx = (i + 1) * n_votes
        prompt_outcomes = task_outcomes[start_idx:end_idx]
        
        # 简单选择策略：取前n_samples个（实际中可以有更复杂的选择策略）
        selected = prompt_outcomes[:n_samples]
        selected_count += len(selected)
        
        if i < 3:  # 只显示前3个作为示例
            print(f"  Prompt {i+1}: 全部{prompt_outcomes} -> 选择{selected}")
    
    print(f"总计选择了 {selected_count} 个rollouts用于训练")
    
    # 步骤6: 总结和建议
    print(f"\n📈 步骤6: VLA-TTRL效果分析")
    improvement = metrics['vote_improvement']
    confidence = metrics['avg_confidence']
    accuracy = metrics['outcome_accuracy']
    
    print(f"投票改进效果: {improvement:+.4f} ({improvement/base_success_rate*100:+.1f}%)")
    print(f"平均投票置信度: {confidence:.4f}")
    print(f"投票准确性: {accuracy:.4f}")
    
    if improvement > 0.05:
        print("✅ VLA-TTRL显示出明显的性能提升！")
    elif improvement > 0:
        print("🔶 VLA-TTRL显示出轻微的性能提升")
    else:
        print("⚠️ VLA-TTRL在当前设置下未显示改进，可能需要调整参数")
    
    if confidence < 0.6:
        print("💡 建议: 增加投票rollout数量以提高置信度")
    
    if accuracy < 0.7:
        print("💡 建议: 当前投票准确性较低，可能需要改进投票策略")
    
    print(f"\n🎉 演示完成！")
    print("=" * 50)
    
    return {
        'config': {
            'n_prompts': n_prompts,
            'n_votes': n_votes,
            'n_samples': n_samples,
            'base_success_rate': base_success_rate
        },
        'results': metrics,
        'vote_details': vote_results
    }

def show_integration_guide():
    """
    显示集成指南
    """
    print(f"\n📚 VLA-TTRL集成指南")
    print("=" * 30)
    
    integration_steps = [
        "1. 将vla_ttrl_utils.py添加到verl/trainer/ppo/目录",
        "2. 将ppo_trainer_vla_ttrl.yaml添加到verl/trainer/config/目录", 
        "3. 修改ray_trainer.py集成VLA-TTRL功能",
        "4. 使用提供的训练脚本启动VLA-TTRL训练",
        "5. 监控vla_ttrl/*指标以评估效果"
    ]
    
    for step in integration_steps:
        print(f"  {step}")
    
    print(f"\n使用示例:")
    print(f"bash examples/run_vla_ttrl_libero.sh")
    print(f"# 或")
    print(f"bash examples/run_vla_ttrl_twin2.sh")

if __name__ == "__main__":
    # 运行完整演示
    demo_results = demonstrate_vla_ttrl()
    
    # 显示集成指南
    show_integration_guide()
    
    print(f"\n🔍 完整结果摘要:")
    print(f"原始成功率: {demo_results['results']['original_success_rate']:.3f}")
    print(f"投票后成功率: {demo_results['results']['voted_success_rate']:.3f}")
    print(f"性能提升: {demo_results['results']['vote_improvement']:+.3f}")
    print(f"平均置信度: {demo_results['results']['avg_confidence']:.3f}")
    print(f"投票准确性: {demo_results['results']['outcome_accuracy']:.3f}")