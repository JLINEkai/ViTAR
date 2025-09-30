import json
import re
from collections import defaultdict


def extract_answer_from_response(response):
    # if not response or response.startswith("Error:"):
    #     return None
    # answer = re.sub(r'[^A-Za-z]', '', response).upper()
    # if len(answer) == 0:
    #     return None

    # return answer[0]
    
    try:
        answer = response['second_response']
        # print(answer)
        match = re.search(r'\{\s*"name"\s*:\s*"Terminate"\s*,\s*"arguments"\s*:\s*\{[^}]*?"answer"\s*:\s*"([^"]+)"', answer)
        # print(match)
        student_answer = match.group(1).upper()[0]
        return student_answer
    except:
        return None
    # 尝试提取<answer>标签中的内容
    # try:
    #     # answer = response['second_response']
    #     answer = response
    #     answer_match = re.search(r'<answer>(.*?)</answer>', answer, re.DOTALL | re.IGNORECASE)
    #     if answer_match:
    #         answer = answer_match.group(1).strip()
    #         # 清理答案，只保留字母
    #         answer = re.sub(r'[^A-Za-z]', '', answer).upper()
    #         return answer[0]
    # except:    
    
    #     return None

def calculate_accuracy(results):
    """
    计算准确率，支持dataset和subset分组
    """
    total_correct = 0
    total_questions = 0
    dataset_stats = defaultdict(lambda: {"correct": 0, "total": 0, "errors": 0, "subsets": defaultdict(lambda: {"correct": 0, "total": 0, "errors": 0})})
    
    for item in results:
        dataset = item.get("dataset", "unknown")
        subset = item.get("subset", "")
        # subset = item.get("question_type", "")
        ground_truth = item.get("ground_truth", "")
        answer = item.get("answer", "")
        
        # 提取模型答案
        predicted_answer = extract_answer_from_response(answer)
        
        dataset_stats[dataset]["total"] += 1
        dataset_stats[dataset]["subsets"][subset]["total"] += 1
        
        if predicted_answer is None:
            dataset_stats[dataset]["errors"] += 1
            dataset_stats[dataset]["subsets"][subset]["errors"] += 1
            continue
        
        # 比较答案
        # print("predicted_answer: ", predicted_answer, "ground_truth: ", ground_truth)
        if predicted_answer == ground_truth:
            total_correct += 1
            dataset_stats[dataset]["correct"] += 1
            dataset_stats[dataset]["subsets"][subset]["correct"] += 1
        
        total_questions += 1
    
    return total_correct, total_questions, dataset_stats

def analyze_results(file_path):
    """
    分析结果文件
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
    except FileNotFoundError:
        print(f"错误: 找不到文件 {file_path}")
        return
    except json.JSONDecodeError:
        print(f"错误: 文件 {file_path} 不是有效的JSON格式")
        return
    
    print(f"加载了 {len(results)} 个结果")
    
    # 计算准确率
    total_correct, total_questions, dataset_stats = calculate_accuracy(results)
    
    # 打印总体结果
    print("\n" + "="*60)
    print("总体结果")
    print("="*60)
    print(f"总问题数: {total_questions}")
    print(f"正确答案数: {total_correct}")
    print(f"总体准确率: {total_correct/total_questions*100:.2f}%" if total_questions > 0 else "总体准确率: 0.00%")
    
    # 打印各数据集结果
    print("\n" + "="*60)
    print("各数据集结果")
    print("="*60)
    
    # 保存详细统计结果
    output_stats = {
        "overall": {
            "total_questions": total_questions,
            "total_correct": total_correct,
            "accuracy": total_correct/total_questions*100 if total_questions > 0 else 0
        },
        "by_dataset": {},
        "by_subset": {}
    }
    
    for dataset, stats in sorted(dataset_stats.items()):
        if stats["total"] > 0:
            accuracy = stats["correct"] / stats["total"] * 100
            error_rate = stats["errors"] / stats["total"] * 100
            print(f"\n{dataset}:")
            print(f"  总问题数: {stats['total']}")
            print(f"  正确答案数: {stats['correct']}")
            print(f"  错误响应数: {stats['errors']}")
            print(f"  准确率: {accuracy:.2f}%")
            print(f"  错误率: {error_rate:.2f}%")
            
            # 保存数据集统计
            output_stats["by_dataset"][dataset] = {
                "total": stats["total"],
                "correct": stats["correct"],
                "errors": stats["errors"],
                "accuracy": accuracy,
                "error_rate": error_rate,
                "subsets": {}
            }
            
            # 打印subset结果
            if stats["subsets"]:
                print(f"  Subsets:")
                for subset, subset_stats in sorted(stats["subsets"].items()):
                    if subset_stats["total"] > 0:
                        subset_accuracy = subset_stats["correct"] / subset_stats["total"] * 100
                        subset_error_rate = subset_stats["errors"] / subset_stats["total"] * 100
                        subset_name = subset if subset else "default"
                        print(f"    {subset_name}:")
                        print(f"      问题数: {subset_stats['total']}")
                        print(f"      正确数: {subset_stats['correct']}")
                        print(f"      错误数: {subset_stats['errors']}")
                        print(f"      准确率: {subset_accuracy:.2f}%")
                        print(f"      错误率: {subset_error_rate:.2f}%")
                        
                        # 保存subset统计
                        output_stats["by_dataset"][dataset]["subsets"][subset_name] = {
                            "total": subset_stats["total"],
                            "correct": subset_stats["correct"],
                            "errors": subset_stats["errors"],
                            "accuracy": subset_accuracy,
                            "error_rate": subset_error_rate
                        }
                        
                        # 同时保存到by_subset中
                        subset_key = f"{dataset}_{subset_name}" if subset else dataset
                        output_stats["by_subset"][subset_key] = {
                            "dataset": dataset,
                            "subset": subset_name,
                            "total": subset_stats["total"],
                            "correct": subset_stats["correct"],
                            "errors": subset_stats["errors"],
                            "accuracy": subset_accuracy,
                            "error_rate": subset_error_rate
                        }
    
    # 保存详细统计结果
    with open('accuracy_analysis.json', 'w', encoding='utf-8') as f:
        json.dump(output_stats, f, ensure_ascii=False, indent=2)
    
    print(f"\n详细统计结果已保存到 accuracy_analysis.json")
    
    # 分析错误案例
    print("\n" + "="*60)
    print("错误案例分析")
    print("="*60)
    
    error_cases = []
    for item in results:
        ground_truth = item.get("ground_truth", "")
        answer = item.get("answer", "")
        # print("AAAA: ", answer)
        predicted_answer = extract_answer_from_response(answer)
        
        if predicted_answer and predicted_answer != ground_truth:
            error_cases.append({
                "dataset": item.get("dataset", "unknown"),
                "subset": item.get("subset", ""),
                "question": item.get("question", ""),
                "ground_truth": ground_truth,
                "predicted": predicted_answer,
                "options": item.get("options", [])
            })
    
    print(f"找到 {len(error_cases)} 个错误案例")
    
    # 按数据集统计错误
    error_by_dataset = defaultdict(int)
    error_by_subset = defaultdict(int)
    for case in error_cases:
        dataset = case["dataset"]
        subset = case["subset"] if case["subset"] else "default"
        error_by_dataset[dataset] += 1
        error_by_subset[f"{dataset}_{subset}"] += 1
    
    print("\n各数据集错误数量:")
    for dataset, count in sorted(error_by_dataset.items()):
        print(f"  {dataset}: {count}")
    
    print("\n各子集错误数量:")
    for subset_key, count in sorted(error_by_subset.items()):
        print(f"  {subset_key}: {count}")
    
    # 生成汇总表格
    print("\n" + "="*60)
    print("汇总表格")
    print("="*60)
    print(f"{'Dataset':<20} {'Subset':<15} {'Total':<8} {'Correct':<8} {'Accuracy':<10} {'Errors':<8}")
    print("-" * 80)
    
    for dataset, stats in sorted(dataset_stats.items()):
        if stats["total"] > 0:
            accuracy = stats["correct"] / stats["total"] * 100
            print(f"{dataset:<20} {'ALL':<15} {stats['total']:<8} {stats['correct']:<8} {accuracy:<10.2f} {stats['errors']:<8}")
            
            for subset, subset_stats in sorted(stats["subsets"].items()):
                if subset_stats["total"] > 0:
                    subset_accuracy = subset_stats["correct"] / subset_stats["total"] * 100
                    subset_name = subset if subset else "default"
                    print(f"{'':<20} {subset_name:<15} {subset_stats['total']:<8} {subset_stats['correct']:<8} {subset_accuracy:<10.2f} {subset_stats['errors']:<8}")

if __name__ == "__main__":
    # 分析output.json文件
    analyze_results("output_tool_v910_n16.json") 
