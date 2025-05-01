#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
问题三：PM2.5减排策略优化 - 主程序
整合模型加载和减排策略优化的主流程
"""

import os
import argparse
from model_loading import load_best_model, load_processed_data, predict_pm25_for_target_date, ROOT_DIR
from emission_reduction_optimizer import (
    build_emission_reduction_model,
    solve_emission_reduction_model,
    visualize_solution,
    generate_report
)

def create_output_dirs():
    """
    创建输出目录
    """
    dirs = [
        os.path.join(ROOT_DIR, "processed3"),
        os.path.join(ROOT_DIR, "processed3", "figures")
    ]
    
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"创建目录: {d}")

def run_pm25_prediction():
    """
    运行PM2.5预测步骤
    
    Returns:
        daily_avg_pm25: 目标日期的PM2.5日均浓度预测值
        actual_date: 实际使用的日期
        data_source: 数据来源
    """
    print("\n" + "="*50)
    print("步骤1: PM2.5预测")
    print("="*50)
    
    print("加载模型...")
    model, feature_columns = load_best_model()
    
    print("\n预测2016年3月1日PM2.5日均浓度...")
    daily_avg_pm25, actual_date, data_source = predict_pm25_for_target_date(model, None, feature_columns)
    
    # 保存结果
    output_dir = os.path.join(ROOT_DIR, "processed3")
    with open(os.path.join(output_dir, "predicted_pm25.txt"), "w", encoding="utf-8") as f:
        if actual_date != "2016-03-01":
            f.write(f"注意：数据中没有2016年3月1日的记录，使用了替代日期 {actual_date} (数据来源: {data_source})\n\n")
        else:
            f.write(f"数据来源: {data_source}\n\n")
        f.write(f"{actual_date} PM2.5日均浓度预测值：{daily_avg_pm25:.2f} μg/m³\n")
        f.write(f"减排目标值（80%）：{daily_avg_pm25 * 0.8:.2f} μg/m³\n")
        f.write(f"需要减少的浓度：{daily_avg_pm25 * 0.2:.2f} μg/m³\n")
    
    print(f"\nPM2.5预测结果已保存到 {os.path.join(output_dir, 'predicted_pm25.txt')}")
    
    return daily_avg_pm25, actual_date, data_source

def run_emission_reduction_optimization(daily_avg_pm25, actual_date="2016-03-01", data_source="未知"):
    """
    运行减排策略优化步骤
    
    Args:
        daily_avg_pm25: 目标日期的PM2.5日均浓度预测值
        actual_date: 实际使用的日期
        data_source: 数据来源
    """
    print("\n" + "="*50)
    print("步骤2: 减排策略优化")
    print("="*50)
    
    # 计算减排目标
    target_percent = 0.2  # 将PM2.5降至预测值的80%，即减少20%
    target_reduction = daily_avg_pm25 * target_percent
    print(f"\n减排目标：将PM2.5浓度从{daily_avg_pm25:.2f}μg/m³降至{daily_avg_pm25 * (1-target_percent):.2f}μg/m³")
    print(f"需要减少：{target_reduction:.2f}μg/m³")
    
    # 构建和求解减排优化模型
    print("\n构建减排优化模型...")
    problem, variables = build_emission_reduction_model(target_reduction)
    
    print("求解减排优化模型...")
    results = solve_emission_reduction_model(problem, variables)
    
    # 输出优化结果
    print("\n优化结果：")
    total_hours = sum(hours for hours in results['solution'].values())
    if total_hours > 0:
        for measure, hours in results['solution'].items():
            if hours > 0:
                cost = results['total_cost'] * hours / total_hours
                reduction = results['total_reduction'] * hours / total_hours
                print(f"- {measure}: 实施{hours}小时，成本{cost:,.2f}元，减排{reduction:.2f}μg/m³")
    else:
        print("- 所有措施实施时长为0，无需实施减排措施")
    
    print(f"\n总成本：{results['total_cost']:,}元，总减排效果：{results['total_reduction']:.2f}μg/m³")
    
    # 可视化结果
    print("\n可视化结果...")
    visualize_solution(results, target_reduction)
    
    # 添加实际日期和数据来源到results字典
    results['actual_date'] = actual_date
    results['data_source'] = data_source
    
    # 生成报告
    print("\n生成报告...")
    generate_report(results, daily_avg_pm25, target_reduction)
    
    print(f"\n减排策略优化结果已保存到 {os.path.join(ROOT_DIR, 'processed3', 'emission_reduction_report.md')}")

def main():
    """
    主函数
    """
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="PM2.5减排策略优化主程序")
    parser.add_argument('--skip-prediction', action='store_true', help='跳过PM2.5预测步骤')
    
    args = parser.parse_args()
    
    # 创建输出目录
    create_output_dirs()
    
    # PM2.5预测步骤
    if not args.skip_prediction:
        daily_avg_pm25, actual_date, data_source = run_pm25_prediction()
    else:
        print("跳过PM2.5预测步骤，从保存的结果文件加载...")
        try:
            predicted_file = os.path.join(ROOT_DIR, "processed3", "predicted_pm25.txt")
            with open(predicted_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
                
                if "注意：数据中没有2016年3月1日的记录" in lines[0]:
                    # 提取数据来源
                    data_source = lines[0].split("(数据来源: ")[1].split(")")[0]
                    # 跳过提示行和空行
                    pm25_line = lines[2]
                    actual_date = pm25_line.split()[0]
                else:
                    # 提取数据来源
                    data_source = lines[0].split("数据来源: ")[1].strip()
                    pm25_line = lines[2]
                    actual_date = pm25_line.split()[0]
                
                daily_avg_pm25 = float(pm25_line.split("：")[1].split(" ")[0])
            print(f"加载的PM2.5预测值：{daily_avg_pm25:.2f} μg/m³ (日期: {actual_date}, 数据来源: {data_source})")
        except Exception as e:
            print(f"加载PM2.5预测值失败: {e}")
            print("运行PM2.5预测步骤...")
            daily_avg_pm25, actual_date, data_source = run_pm25_prediction()
    
    # 减排策略优化步骤
    run_emission_reduction_optimization(daily_avg_pm25, actual_date, data_source)
    
    print("\n" + "="*50)
    print("处理完成")
    print("="*50)

if __name__ == "__main__":
    main() 