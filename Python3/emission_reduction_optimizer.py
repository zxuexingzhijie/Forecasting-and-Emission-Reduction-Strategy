#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
问题三：PM2.5减排策略优化 - 减排策略优化模块
构建线性规划模型，求解最优减排策略
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pulp import LpMinimize, LpProblem, LpVariable, LpInteger, LpStatus, value
from model_loading import load_best_model, load_processed_data, predict_pm25_for_target_date, ROOT_DIR

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 定义减排措施参数
MEASURES = {
    '限行': {'cost': 5000, 'efficiency': 3, 'color': 'skyblue'},
    '工厂限产': {'cost': 8000, 'efficiency': 5, 'color': 'lightgreen'},
    '洒水抑尘': {'cost': 3000, 'efficiency': 2, 'color': 'lightsalmon'}
}

def build_emission_reduction_model(target_reduction, total_budget=100000, max_hours=16):
    """
    构建减排优化模型
    
    Args:
        target_reduction: 目标减排量(μg/m³)
        total_budget: 总预算(元)，默认10万元
        max_hours: 最大实施时长(小时)，默认16小时(6:00-22:00)
        
    Returns:
        problem: 构建的线性规划问题
        variables: 决策变量字典
    """
    # 创建问题实例
    problem = LpProblem("PM25_Emission_Reduction", LpMinimize)
    
    # 创建决策变量：每种措施实施的小时数
    variables = {
        measure: LpVariable(f"hours_{measure}", 0, max_hours, LpInteger)
        for measure in MEASURES
    }
    
    # 目标函数：最小化总成本
    problem += sum(MEASURES[measure]['cost'] * variables[measure] for measure in MEASURES), "总成本"
    
    # 约束条件1：减排效果达到目标
    problem += sum(MEASURES[measure]['efficiency'] * variables[measure] for measure in MEASURES) >= target_reduction, "减排效果约束"
    
    # 约束条件2：总成本不超过预算
    problem += sum(MEASURES[measure]['cost'] * variables[measure] for measure in MEASURES) <= total_budget, "预算约束"
    
    # 约束条件3：每种措施实施时间不超过最大时长
    for measure in MEASURES:
        problem += variables[measure] <= max_hours, f"{measure}时长约束"
    
    return problem, variables

def solve_emission_reduction_model(problem, variables):
    """
    求解减排优化模型
    
    Args:
        problem: 构建的线性规划问题
        variables: 决策变量字典
        
    Returns:
        results: 求解结果字典
    """
    # 求解问题
    problem.solve()
    
    # 检查求解状态
    print(f"优化模型求解状态: {LpStatus[problem.status]}")
    if LpStatus[problem.status] != "Optimal":
        raise Exception(f"模型求解未找到最优解，状态：{LpStatus[problem.status]}")
    
    # 整理结果
    solution = {measure: int(value(variables[measure])) for measure in variables}
    total_cost = sum(MEASURES[measure]['cost'] * solution[measure] for measure in solution)
    total_reduction = sum(MEASURES[measure]['efficiency'] * solution[measure] for measure in solution)
    
    # 计算实施时段（假设从6:00开始）
    implementation_periods = {}
    for measure in solution:
        if solution[measure] > 0:
            start_hour = 6
            end_hour = start_hour + solution[measure]
            implementation_periods[measure] = f"{start_hour:02d}:00-{end_hour:02d}:00"
    
    results = {
        'solution': solution,
        'total_cost': total_cost,
        'total_reduction': total_reduction,
        'implementation_periods': implementation_periods
    }
    
    return results

def visualize_solution(results, target_reduction):
    """
    可视化减排策略优化结果
    
    Args:
        results: 优化结果字典
        target_reduction: 目标减排量
    """
    solution = results['solution']
    
    # 创建输出目录
    figures_dir = os.path.join(ROOT_DIR, "processed3", "figures")
    os.makedirs(figures_dir, exist_ok=True)
    
    # 1. 实施时长条形图
    plt.figure(figsize=(10, 6))
    measures = list(solution.keys())
    hours = [solution[measure] for measure in measures]
    colors = [MEASURES[measure]['color'] for measure in measures]
    
    plt.bar(measures, hours, color=colors)
    plt.axhline(y=16, color='red', linestyle='--', alpha=0.7, label='最大可实施时长(16小时)')
    
    plt.title('各减排措施实施时长')
    plt.xlabel('减排措施')
    plt.ylabel('实施时长(小时)')
    plt.ylim(0, max(16, max(hours) * 1.1) if hours else 16)
    plt.grid(axis='y', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "measures_implementation_hours.png"), dpi=300)
    plt.close()
    
    # 2. 成本分布饼图
    cost_data = [MEASURES[measure]['cost'] * solution[measure] for measure in measures]
    # 只有在有非零成本时才创建饼图
    if sum(cost_data) > 0:
        plt.figure(figsize=(10, 6))
        plt.pie(cost_data, labels=measures, autopct='%1.1f%%', colors=colors,
                wedgeprops={'edgecolor': 'white', 'linewidth': 1})
        plt.title(f'总成本分布 (总计: {results["total_cost"]:,} 元)')
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, "cost_distribution_pie.png"), dpi=300)
        plt.close()
    
    # 3. 减排效果分布饼图
    reduction_data = [MEASURES[measure]['efficiency'] * solution[measure] for measure in measures]
    # 只有在有非零减排效果时才创建饼图
    if sum(reduction_data) > 0:
        plt.figure(figsize=(10, 6))
        plt.pie(reduction_data, labels=measures, autopct='%1.1f%%', colors=colors,
                wedgeprops={'edgecolor': 'white', 'linewidth': 1})
        plt.title(f'减排效果分布 (总计: {results["total_reduction"]:.2f} μg/m³)')
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, "reduction_distribution_pie.png"), dpi=300)
        plt.close()
    
    # 4. 时间段实施计划甘特图
    # 只在有措施实施时创建甘特图
    if any(solution.values()):
        plt.figure(figsize=(12, 6))
        
        # 假设从6:00开始实施
        active_measures = [(i, measure) for i, measure in enumerate(measures) if solution[measure] > 0]
        
        for i, measure in active_measures:
            plt.barh(i, solution[measure], left=6, color=MEASURES[measure]['color'])
            # 添加标签
            plt.text(6 + solution[measure]/2, i, f"{solution[measure]}小时", 
                    ha='center', va='center', color='black', fontweight='bold')
        
        plt.yticks(range(len(active_measures)), [m[1] for m in active_measures])
        plt.xlim(6, 22)
        plt.xticks(range(6, 23), [f"{h}:00" for h in range(6, 23)])
        plt.grid(axis='x', alpha=0.3)
        plt.title('减排措施实施时间计划')
        plt.xlabel('时间')
        plt.ylabel('减排措施')
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, "implementation_schedule.png"), dpi=300)
        plt.close()
    
    # 5. 成本效益分析
    # 计算成本效益比
    efficiency_data = []
    active_measures = []
    for measure in measures:
        if solution[measure] > 0 and MEASURES[measure]['cost'] * solution[measure] > 0:
            efficiency = MEASURES[measure]['efficiency'] / MEASURES[measure]['cost']
            efficiency_data.append(efficiency)
            active_measures.append(measure)
    
    # 只在有活跃措施时创建成本效益图
    if efficiency_data:
        plt.figure(figsize=(10, 6))
        colors = [MEASURES[measure]['color'] for measure in active_measures]
        plt.bar(active_measures, efficiency_data, color=colors)
        plt.title('各减排措施成本效益比较 (μg/m³ 每万元)')
        plt.xlabel('减排措施')
        plt.ylabel('减排效率 (μg/m³ / 万元)')
        plt.grid(axis='y', alpha=0.3)
        
        # 添加数值标签
        for i, v in enumerate(efficiency_data):
            plt.text(i, v * 1.02, f"{v*10000:.2f}", ha='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, "cost_efficiency_comparison.png"), dpi=300)
        plt.close()

def generate_report(results, daily_avg_pm25, target_reduction):
    """
    生成减排策略优化报告
    
    Args:
        results: 优化结果字典
        daily_avg_pm25: 预测的日均PM2.5浓度
        target_reduction: 目标减排量
    """
    solution = results['solution']
    implementation_periods = results['implementation_periods']
    actual_date = results.get('actual_date', '2016-03-01')  # 获取实际使用的日期
    data_source = results.get('data_source', '未知')  # 获取数据来源
    
    report = "# PM2.5减排策略优化报告\n\n"
    
    # 基本信息
    report += "## 1. 基本信息\n\n"
    
    # 如果实际日期与目标日期不同，添加说明
    if actual_date != '2016-03-01':
        report += f"**注意：** 由于数据中未找到2016年3月1日的记录，本报告使用了数据集中的替代日期({actual_date})进行分析。\n"
        report += f"**数据来源：** {data_source}\n\n"
    else:
        report += f"**数据来源：** {data_source}\n\n"
    
    report += f"- 预测日期：{actual_date}\n"
    report += f"- 预测PM2.5日均浓度：{daily_avg_pm25:.2f} μg/m³\n"
    report += f"- 减排目标值（80%）：{daily_avg_pm25 * 0.8:.2f} μg/m³\n"
    report += f"- 需要减少的浓度：{target_reduction:.2f} μg/m³\n"
    report += f"- 总预算：10万元/日\n"
    report += f"- 可实施时段：6:00-22:00（16小时）\n\n"
    
    # 数学模型
    report += "## 2. 数学模型\n\n"
    report += "### 2.1 决策变量\n\n"
    report += "- $x_1$：限行措施实施小时数\n"
    report += "- $x_2$：工厂限产措施实施小时数\n"
    report += "- $x_3$：洒水抑尘措施实施小时数\n\n"
    
    report += "### 2.2 目标函数\n\n"
    report += "最小化总成本：\n\n"
    report += "$$min Z = 5000x_1 + 8000x_2 + 3000x_3$$\n\n"
    
    report += "### 2.3 约束条件\n\n"
    report += "1. 减排效果约束：\n\n"
    report += f"   $$3x_1 + 5x_2 + 2x_3 \\geq {target_reduction:.2f}$$\n\n"
    
    report += "2. 预算约束：\n\n"
    report += "   $$5000x_1 + 8000x_2 + 3000x_3 \\leq 100000$$\n\n"
    
    report += "3. 时间约束：\n\n"
    report += "   $$0 \\leq x_1, x_2, x_3 \\leq 16$$\n\n"
    
    report += "4. 整数约束：\n\n"
    report += "   $$x_1, x_2, x_3 \\in \\mathbb{Z}^+$$\n\n"
    
    # 优化结果
    report += "## 3. 优化结果\n\n"
    report += "### 3.1 最优减排策略\n\n"
    report += "| 减排措施 | 实施时长(小时) | 实施时段 | 减排效果(μg/m³) | 成本(元) |\n"
    report += "|---------|--------------|---------|--------------|--------|\n"
    
    for measure in solution:
        hours = solution[measure]
        period = implementation_periods.get(measure, "无需实施")
        reduction = MEASURES[measure]['efficiency'] * hours
        cost = MEASURES[measure]['cost'] * hours
        report += f"| {measure} | {hours} | {period} | {reduction:.2f} | {cost:,} |\n"
    
    report += "\n### 3.2 总体效果\n\n"
    report += f"- 总减排效果：{results['total_reduction']:.2f} μg/m³\n"
    report += f"- 总成本：{results['total_cost']:,} 元\n"
    report += f"- 预计实施后PM2.5浓度：{daily_avg_pm25 - results['total_reduction']:.2f} μg/m³\n"
    report += f"- 减排目标达成率：{(results['total_reduction'] / target_reduction * 100):.2f}%\n\n"
    
    # 敏感性分析
    report += "## 4. 敏感性分析和策略建议\n\n"
    
    # 成本效益分析
    report += "### 4.1 成本效益分析\n\n"
    report += "| 减排措施 | 单位成本(元/小时) | 单位效率(μg/m³·h) | 成本效益比(μg/m³/万元) |\n"
    report += "|---------|----------------|-----------------|---------------------|\n"
    
    for measure in MEASURES:
        cost = MEASURES[measure]['cost']
        efficiency = MEASURES[measure]['efficiency']
        cost_effectiveness = efficiency / cost * 10000  # 转换为每万元
        report += f"| {measure} | {cost:,} | {efficiency:.2f} | {cost_effectiveness:.2f} |\n"
    
    report += "\n### 4.2 策略建议\n\n"
    
    # 根据结果生成策略建议
    if solution['洒水抑尘'] > 0:
        report += "- **优先使用洒水抑尘**：洒水抑尘措施具有最高的成本效益比，应优先实施并尽可能延长实施时间。\n"
    
    if solution['限行'] > 0:
        report += "- **适当实施限行**：限行措施的成本效益比居中，可在交通高峰期实施以最大化效果。\n"
    
    if solution['工厂限产'] > 0:
        report += "- **谨慎使用工厂限产**：工厂限产成本高但减排效果显著，建议在污染峰值时段有针对性地实施。\n"
    
    total_hours = sum(solution.values())
    if total_hours < 16 * 3:  # 如果总实施时间未达到理论最大值
        report += "- **有序安排实施时段**：各措施总实施时长未达到理论最大值，说明当前预算下已获得经济有效的减排方案。\n"
    
    if results['total_cost'] < 100000:
        saved_budget = 100000 - results['total_cost']
        report += f"- **预算节约**：最优方案节约预算{saved_budget:,}元，可考虑用于其他环保工作或储备应对更严重污染天气。\n"
    
    # 保存报告
    output_dir = os.path.join(ROOT_DIR, "processed3")
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "emission_reduction_report.md"), "w", encoding="utf-8") as f:
        f.write(report)
    
    print(f"减排策略优化报告已保存到 {os.path.join(output_dir, 'emission_reduction_report.md')}")

def main():
    """
    主函数
    """
    # 第一步：加载模型，预测目标日期的PM2.5浓度
    print("加载模型...")
    model, feature_columns = load_best_model()
    
    print("\n预测2016年3月1日PM2.5日均浓度...")
    daily_avg_pm25, actual_date, data_source = predict_pm25_for_target_date(model, None, feature_columns)
    
    # 第二步：计算减排目标
    target_percent = 0.2  # 将PM2.5降至预测值的80%，即减少20%
    target_reduction = daily_avg_pm25 * target_percent
    print(f"\n减排目标：将PM2.5浓度从{daily_avg_pm25:.2f}μg/m³降至{daily_avg_pm25 * (1-target_percent):.2f}μg/m³")
    print(f"需要减少：{target_reduction:.2f}μg/m³")
    
    # 第三步：构建和求解减排优化模型
    print("\n构建减排优化模型...")
    problem, variables = build_emission_reduction_model(target_reduction)
    
    print("求解减排优化模型...")
    results = solve_emission_reduction_model(problem, variables)
    
    # 添加日期和数据来源信息
    results['actual_date'] = actual_date
    results['data_source'] = data_source
    
    print("\n优化结果：")
    for measure, hours in results['solution'].items():
        if hours > 0:
            cost = MEASURES[measure]['cost'] * hours
            reduction = MEASURES[measure]['efficiency'] * hours
            print(f"- {measure}: 实施{hours}小时，成本{cost:,}元，减排{reduction:.2f}μg/m³")
    
    print(f"\n总成本：{results['total_cost']:,}元，总减排效果：{results['total_reduction']:.2f}μg/m³")
    
    # 第四步：可视化结果并生成报告
    print("\n可视化结果...")
    visualize_solution(results, target_reduction)
    
    print("\n生成报告...")
    generate_report(results, daily_avg_pm25, target_reduction)
    
    print("\n处理完成")

if __name__ == "__main__":
    main() 