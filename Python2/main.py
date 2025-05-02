#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PM2.5预测模型主程序
整合数据加载、特征工程、模型训练和评估的主流程
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# 导入自定义模块
from data_loader import load_processed_data, train_test_split_by_time, analyze_static_weather
from data_loader import visualize_static_weather, compute_statistics, save_data_info
from feature_engineering import prepare_model_data, visualize_feature_importance, save_features_info
import model_training as mt

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def create_output_dirs():
    """
    创建输出目录
    """
    dirs = [
        "./processed2",
        "./processed2/figures",
        "./processed2/models"
    ]
    
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"创建目录: {d}")

def run_data_loading():
    """
    运行数据加载和预处理步骤
    
    Returns:
        train_df, test_df: 训练集和测试集
    """
    print("\n" + "="*50)
    print("步骤1: 数据加载和初步分析")
    print("="*50)
    
    # 加载数据
    print("加载数据...")
    df = load_processed_data()
    
    if df is None:
        print("数据加载失败，退出程序")
        return None, None
    
    # 进行探索性分析
    print("进行探索性分析...")
    
    # 分析静稳天气
    print("分析静稳天气...")
    stagnant_results = analyze_static_weather(df)
    
    # 可视化静稳天气
    print("可视化静稳天气分析结果...")
    visualize_static_weather(stagnant_results)
    
    # 计算统计信息
    print("计算统计信息...")
    stats = compute_statistics(stagnant_results['df_with_stagnant'])
    
    # 保存数据信息
    print("保存数据信息...")
    save_data_info(stagnant_results['df_with_stagnant'])
    
    # 进行训练集和测试集划分
    print("划分训练集和测试集...")
    train_df, test_df = train_test_split_by_time(stagnant_results['df_with_stagnant'], 
                                               test_size=0.2, 
                                               date_column='datetime',
                                               gap_days=10)  # 添加10天间隔防止信息泄露
    
    # 保存训练集和测试集
    train_df.to_csv("./processed2/train_data.csv", index=False)
    test_df.to_csv("./processed2/test_data.csv", index=False)
    print("训练集和测试集已保存")
    
    return train_df, test_df

def run_feature_engineering(train_df, test_df):
    """
    运行特征工程步骤
    
    Args:
        train_df: 训练集
        test_df: 测试集
        
    Returns:
        X_train, y_train, X_test, y_test: 特征和标签
    """
    print("\n" + "="*50)
    print("步骤2: 特征工程")
    print("="*50)
    
    # 确保datetime列是日期时间类型
    for df in [train_df, test_df]:
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
    
    # 准备训练数据
    print("准备训练数据...")
    processed_train, scaler, importance_df = prepare_model_data(
        train_df, 
        target_col='PM2.5',
        include_lagged=True,
        include_rolling=True,
        include_weather=True,
        include_pollutants=True,
        add_combinations=True,
        scale=True
    )
    
    # 准备测试数据
    print("准备测试数据...")
    processed_test, _, _ = prepare_model_data(
        test_df, 
        target_col='PM2.5',
        include_lagged=True,
        include_rolling=True,
        include_weather=True,
        include_pollutants=True,
        add_combinations=True,
        scale=True
    )
    
    # 可视化特征重要性
    print("可视化特征重要性...")
    # 确保只包含数值型数据
    numeric_importance_df = importance_df[importance_df['importance'].apply(lambda x: isinstance(x, (int, float)))]
    visualize_feature_importance(numeric_importance_df.sort_values('importance', ascending=False))
    
    # 保存特征信息
    print("保存特征信息...")
    save_features_info(processed_train, importance_df)
    
    # 保存处理后的数据
    processed_train.to_csv("./processed2/processed_train.csv", index=False)
    processed_test.to_csv("./processed2/processed_test.csv", index=False)
    print("处理后的训练和测试数据已保存")
    
    # 准备特征和标签
    print("准备模型训练的特征和标签...")
    
    # 排除不用于训练的列
    exclude_cols = ['datetime']
    target_col = 'PM2.5'
    
    # 为训练准备数据
    X_train = processed_train.drop(exclude_cols + [target_col], axis=1) if 'datetime' in processed_train.columns else processed_train.drop([target_col], axis=1)
    y_train = processed_train[target_col]
    
    X_test = processed_test.drop(exclude_cols + [target_col], axis=1) if 'datetime' in processed_test.columns else processed_test.drop([target_col], axis=1)
    y_test = processed_test[target_col]
    
    print(f"训练集特征形状: {X_train.shape}")
    print(f"测试集特征形状: {X_test.shape}")
    
    return X_train, y_train, X_test, y_test, processed_train, processed_test

def run_model_training(X_train, y_train, X_test, y_test, processed_train, processed_test):
    """
    运行模型训练和评估步骤
    
    Args:
        X_train, y_train: 训练集特征和标签
        X_test, y_test: 测试集特征和标签
        processed_train, processed_test: 处理后的完整训练和测试数据
    """
    print("\n" + "="*50)
    print("步骤3: 模型训练和评估")
    print("="*50)
    
    # 训练基础模型
    print("训练基础模型...")
    model_results, trained_models = mt.train_models(X_train, y_train, X_test, y_test)
    
    # 为最佳模型调优超参数
    best_model_name = model_results[model_results['model'].str.contains('test')]['RMSE'].idxmin()
    best_model_name = model_results.loc[best_model_name, 'model'].split('_')[0]
    print(f"\n最佳基础模型: {best_model_name}")
    
    if best_model_name in ['RF', 'XGB', 'LGB']:
        print(f"\n为{best_model_name}模型调优超参数...")
        tuned_model, train_cols = mt.tune_hyperparameters(X_train, y_train, model_type=best_model_name)
        
        # 确保测试集使用与训练时相同的特征列
        print(f"训练时使用了{len(train_cols)}个特征: {train_cols}")
        print(f"原始测试集有{X_test.shape[1]}个特征")
        
        # 处理测试集特征不匹配的问题
        missing_cols = set(train_cols) - set(X_test.columns)
        extra_cols = set(X_test.columns) - set(train_cols)
        
        if missing_cols:
            print(f"测试集缺少以下特征: {missing_cols}")
            # 为测试集添加缺失的特征并填充0
            for col in missing_cols:
                X_test[col] = 0
                
        if extra_cols:
            print(f"测试集包含多余特征: {extra_cols}")
            # 删除多余特征
            X_test = X_test.drop(columns=extra_cols)
            
        # 确保特征顺序一致
        X_test = X_test[train_cols]
        print(f"处理后测试集有{X_test.shape[1]}个特征")
        
        # 评估调优后的模型
        test_pred = tuned_model.predict(X_test)
        tuned_metrics = mt.evaluate_model(y_test, test_pred, f"{best_model_name}_tuned_test")
        print(f"\n调优后的{best_model_name}模型性能:")
        print(f"RMSE: {tuned_metrics['RMSE']:.2f}")
        print(f"MAE: {tuned_metrics['MAE']:.2f}")
        print(f"R2: {tuned_metrics['R2']:.2f}")
        
        # 分析静稳天气下的性能
        print("\n分析静稳天气下的模型性能...")
        performance = mt.analyze_static_weather_performance(X_test, y_test, tuned_model, train_cols=train_cols)
        
        # 可视化预测结果
        print("\n可视化预测结果...")
        # 添加datetime列用于可视化
        X_test_with_dt = X_test.copy()
        if 'datetime' in processed_test.columns:
            X_test_with_dt['datetime'] = processed_test['datetime']
        mt.visualize_predictions(
            test_data=X_test_with_dt,
            y_test=y_test,
            model=tuned_model,
            train_cols=train_cols,
            output_dir="./processed2/figures"
        )
        
        # 使用的最终模型
        final_model = tuned_model
    else:
        # 使用未调优的最佳模型
        best_model = trained_models[best_model_name]
        
        # 获取训练时使用的特征列
        train_cols = X_train.columns.tolist()
        
        # 分析静稳天气下的性能
        print("\n分析静稳天气下的模型性能...")
        performance = mt.analyze_static_weather_performance(X_test, y_test, best_model, train_cols=train_cols)
        
        # 可视化预测结果
        print("\n可视化预测结果...")
        # 添加datetime列用于可视化
        X_test_with_dt = X_test.copy()
        if 'datetime' in processed_test.columns:
            X_test_with_dt['datetime'] = processed_test['datetime']
        mt.visualize_predictions(
            test_data=X_test_with_dt,
            y_test=y_test,
            model=best_model,
            train_cols=train_cols,
            output_dir="./processed2/figures"
        )
        
        # 使用的最终模型
        final_model = best_model
    
    # 生成结论报告
    print("\n生成结论报告...")
    generate_conclusion_report(model_results, performance, best_model_name)
    
    print("\n模型训练和评估完成!")

def generate_conclusion_report(model_results, stagnant_performance, best_model_name):
    """
    生成结论报告
    
    Args:
        model_results: 模型评估结果
        stagnant_performance: 静稳天气性能分析结果
        best_model_name: 最佳模型名称
    """
    # 如果stagnant_performance为None，提前返回
    if stagnant_performance is None:
        print("无法生成静稳天气性能报告：未找到静稳天气标记")
        return
    
    with open("./processed2/conclusion_report.txt", "w", encoding="utf-8") as f:
        f.write("PM2.5预测模型结论报告\n")
        f.write("=" * 50 + "\n\n")
        
        # 1. 模型性能总结
        f.write("1. 模型性能总结\n")
        f.write("-" * 50 + "\n\n")
        
        # 获取测试集上的性能结果
        test_results = model_results[model_results['model'].str.contains('test')].copy()
        test_results['model'] = test_results['model'].str.replace('_test', '')
        test_results = test_results.sort_values('RMSE')
        
        f.write("各模型在测试集上的性能（按RMSE排序）：\n")
        for idx, row in test_results.iterrows():
            f.write(f"{row['model']:<15} RMSE: {row['RMSE']:.2f}  MAE: {row['MAE']:.2f}  MAPE: {row['MAPE']:.2f}%  R2: {row['R2']:.2f}\n")
        
        f.write(f"\n最佳模型: {best_model_name}\n\n")
        
        # 2. 静稳天气影响分析
        f.write("2. 静稳天气影响分析\n")
        f.write("-" * 50 + "\n\n")
        
        # 静稳天气分析
        non_stagnant = stagnant_performance['stagnant_0']
        stagnant = stagnant_performance['stagnant_1']
        
        f.write("非静稳天气和静稳天气下模型性能对比：\n\n")
        
        f.write(f"非静稳天气 (样本数: {non_stagnant['count']})\n")
        f.write(f"  RMSE: {non_stagnant['RMSE']:.2f}\n")
        f.write(f"  MAE: {non_stagnant['MAE']:.2f}\n")
        f.write(f"  MAPE: {non_stagnant['MAPE']:.2f}%\n")
        f.write(f"  R2: {non_stagnant['R2']:.2f}\n\n")
        
        f.write(f"静稳天气 (样本数: {stagnant['count']})\n")
        f.write(f"  RMSE: {stagnant['RMSE']:.2f}\n")
        f.write(f"  MAE: {stagnant['MAE']:.2f}\n")
        f.write(f"  MAPE: {stagnant['MAPE']:.2f}%\n")
        f.write(f"  R2: {stagnant['R2']:.2f}\n\n")
        
        # 计算性能差异
        rmse_diff = stagnant['RMSE'] / non_stagnant['RMSE'] - 1
        mae_diff = stagnant['MAE'] / non_stagnant['MAE'] - 1
        mape_diff = stagnant['MAPE'] / non_stagnant['MAPE'] - 1
        r2_diff = non_stagnant['R2'] - stagnant['R2']
        
        f.write(f"静稳天气相对于非静稳天气的性能变化:\n")
        f.write(f"  RMSE增加: {rmse_diff:.2%}\n")
        f.write(f"  MAE增加: {mae_diff:.2%}\n")
        f.write(f"  MAPE增加: {mape_diff:.2%}\n")
        f.write(f"  R2减少: {r2_diff:.2f}\n\n")
        
        # 结论
        f.write("3. 结论与建议\n")
        f.write("-" * 50 + "\n\n")
        
        f.write("主要发现：\n")
        if rmse_diff > 0.1:  # 如果RMSE增加超过10%
            f.write("1. 静稳天气条件下，模型预测精度显著下降，预测误差明显增大。\n")
            f.write(f"2. 静稳天气下RMSE比非静稳天气增加了{rmse_diff:.2%}，表明模型在静稳条件下预测能力受限。\n")
            f.write("3. 这可能是由于静稳天气下，污染物扩散受限，积累效应更为复杂，常规特征难以充分捕捉。\n\n")
            
            f.write("改进建议：\n")
            f.write("1. 为静稳天气构建专门的预测子模型，以提高在特殊气象条件下的预测精度。\n")
            f.write("2. 增加静稳天气持续时间特征，以捕捉污染物在静稳条件下的积累效应。\n")
            f.write("3. 考虑引入更多大气稳定度和边界层高度等相关特征，以更好地表征静稳条件下的扩散环境。\n")
            f.write("4. 使用集成学习方法，结合针对静稳和非静稳天气的多个子模型，以提高整体预测性能。\n")
        else:
            f.write("1. 静稳天气条件下，模型预测精度略有下降，但差异不十分显著。\n")
            f.write(f"2. 静稳天气下RMSE比非静稳天气增加了{rmse_diff:.2%}，表明现有模型对静稳条件有一定的适应能力。\n")
            f.write("3. 当前特征工程和模型选择已经能较好地处理不同气象条件下的预测任务。\n\n")
            
            f.write("改进建议：\n")
            f.write("1. 进一步优化特征工程，特别是针对静稳天气的特征提取，以减小预测误差。\n")
            f.write("2. 考虑使用集成模型或深度学习方法，以提高模型的泛化能力。\n")
            f.write("3. 探索引入更多外部数据源，如卫星遥感数据、交通流量数据等，以增强模型对复杂环境的理解。\n")
    
    print(f"结论报告已保存至 ./processed2/conclusion_report.txt")

def main():
    """
    主函数
    """
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="PM2.5预测模型主程序")
    parser.add_argument('--skip-data-loading', action='store_true', help='跳过数据加载步骤')
    parser.add_argument('--skip-feature-engineering', action='store_true', help='跳过特征工程步骤')
    parser.add_argument('--skip-model-training', action='store_true', help='跳过模型训练步骤')
    
    args = parser.parse_args()
    
    # 创建输出目录
    create_output_dirs()
    
    # 数据加载步骤
    if not args.skip_data_loading:
        train_df, test_df = run_data_loading()
        if train_df is None or test_df is None:
            return
    else:
        print("跳过数据加载步骤，从保存的文件加载数据...")
        train_df = pd.read_csv("./processed2/train_data.csv")
        test_df = pd.read_csv("./processed2/test_data.csv")
    
    # 特征工程步骤
    if not args.skip_feature_engineering:
        X_train, y_train, X_test, y_test, processed_train, processed_test = run_feature_engineering(train_df, test_df)
    else:
        print("跳过特征工程步骤，从保存的文件加载特征...")
        processed_train = pd.read_csv("./processed2/processed_train.csv")
        processed_test = pd.read_csv("./processed2/processed_test.csv")
        
        # 排除不用于训练的列
        exclude_cols = ['datetime']
        target_col = 'PM2.5'
        
        # 为训练准备数据
        X_train = processed_train.drop(exclude_cols + [target_col], axis=1) if 'datetime' in processed_train.columns else processed_train.drop([target_col], axis=1)
        y_train = processed_train[target_col]
        
        X_test = processed_test.drop(exclude_cols + [target_col], axis=1) if 'datetime' in processed_test.columns else processed_test.drop([target_col], axis=1)
        y_test = processed_test[target_col]
    
    # 模型训练步骤
    if not args.skip_model_training:
        run_model_training(X_train, y_train, X_test, y_test, processed_train, processed_test)
    else:
        print("跳过模型训练步骤")
    
    print("\n程序执行完成！")

if __name__ == "__main__":
    main() 