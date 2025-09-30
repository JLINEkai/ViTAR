# verl/utils/reward_score/acc_reward.py

import os
import re


def compute_score(predict_str: str, ground_truth: str, extra_info=None) -> float:
    """计算准确率的奖励函数，支持符号验证和字符串匹配
    
    Args:
        predict_str: 模型预测的答案
        ground_truth: 标准答案
        extra_info: 额外信息（可选）
        
    Returns:
        float: 1.0 表示正确，0.0 表示错误
    """
    reward = 0.0
    
    try:
        # 1. 从答案中提取 <answer> 标签内容
        # match = re.search(r'<answer>\s*(.*?)\s*</answer>', predict_str)
        # student_answer = predict_match.group(1).strip() if predict_match else predict_str.strip()
        
        match = re.search(r'\{\s*"name"\s*:\s*"Terminate"\s*,\s*"arguments"\s*:\s*\{[^}]*?"answer"\s*:\s*"([^"]+)"', predict_str)
        if match:
            student_answer =  match.group(1)
        else:
            student_answer = predict_str.strip()
        ground_truth = ground_truth.strip()
        print(f"predict_answer: {student_answer}")
        print(f"ground_truth: {ground_truth}")

        student_answer = student_answer.replace(' ', '').replace('_', '').lower()
        ground_truth = ground_truth.replace(' ', '').replace('_', '').lower()
        

        # if ground_truth in student_answer or student_answer in ground_truth:
        if ground_truth == student_answer:
            reward = 1.0
            
    except Exception as e:
        print(f"Error in compute_score: {e}")

    
    return reward