#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
模型改进模块
针对静稳天气预测误差较大的问题，提出改进策略
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import xgboost as xgb
import lightgbm as lgb
from tqdm import tqdm

# 从model_training导入评估函数
from model_training import evaluate_model, visualize_predictions, analyze_static_weather_performance

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def build_specialized_models(X_train, y_train, is_stagnant_col='is_stagnant'):
    """
    针对静稳天气和非静稳天气分别构建专门的模型
    
    Args:
        X_train: 训练特征
        y_train: 训练标签
        is_stagnant_col: 静稳天气标记列
        
    Returns:
        specialized_models: 专门的模型字典
    """
    print("构建针对静稳天气和非静稳天气的专门模型...")
    
    if is_stagnant_col not in X_train.columns:
        raise ValueError(f"在特征中未找到静稳天气列 '{is_stagnant_col}'")
    
    # 划分静稳天气和非静稳天气样本
    stagnant_mask = X_train[is_stagnant_col] == 1
    
    X_train_stagnant = X_train[stagnant_mask].copy()
    y_train_stagnant = y_train[stagnant_mask].copy()
    
    X_train_non_stagnant = X_train[~stagnant_mask].copy()
    y_train_non_stagnant = y_train[~stagnant_mask].copy()
    
    print(f"静稳天气样本数: {len(X_train_stagnant)}")
    print(f"非静稳天气样本数: {len(X_train_non_stagnant)}")
    
    # 为静稳天气训练模型
    print("训练静稳天气专用模型...")
    stagnant_model = lgb.LGBMRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=7,
        num_leaves=50,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    stagnant_model.fit(
        X_train_stagnant.drop(is_stagnant_col, axis=1), 
        y_train_stagnant,
        verbose=False
    )
    
    # 为非静稳天气训练模型
    print("训练非静稳天气专用模型...")
    non_stagnant_model = xgb.XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    non_stagnant_model.fit(
        X_train_non_stagnant.drop(is_stagnant_col, axis=1), 
        y_train_non_stagnant,
        verbose=False
    )
    
    # 整合为专门的模型字典
    specialized_models = {
        'stagnant_model': stagnant_model,
        'non_stagnant_model': non_stagnant_model
    }
    
    # 保存模型
    os.makedirs("./processed2/models", exist_ok=True)
    joblib.dump(stagnant_model, "./processed2/models/stagnant_specialized_model.pkl")
    joblib.dump(non_stagnant_model, "./processed2/models/non_stagnant_specialized_model.pkl")
    
    return specialized_models

def predict_with_specialized_models(X_test, specialized_models, is_stagnant_col='is_stagnant'):
    """
    使用专门的模型进行预测
    
    Args:
        X_test: 测试特征
        specialized_models: 专门的模型字典
        is_stagnant_col: 静稳天气标记列
        
    Returns:
        y_pred: 预测值
    """
    if is_stagnant_col not in X_test.columns:
        raise ValueError(f"在特征中未找到静稳天气列 '{is_stagnant_col}'")
    
    # 划分静稳天气和非静稳天气样本
    stagnant_mask = X_test[is_stagnant_col] == 1
    
    # 预测结果初始化
    y_pred = np.zeros(len(X_test))
    
    # 对静稳天气样本进行预测
    if np.any(stagnant_mask):
        X_test_stagnant = X_test[stagnant_mask].drop(is_stagnant_col, axis=1)
        y_pred_stagnant = specialized_models['stagnant_model'].predict(X_test_stagnant)
        y_pred[stagnant_mask] = y_pred_stagnant
    
    # 对非静稳天气样本进行预测
    if np.any(~stagnant_mask):
        X_test_non_stagnant = X_test[~stagnant_mask].drop(is_stagnant_col, axis=1)
        y_pred_non_stagnant = specialized_models['non_stagnant_model'].predict(X_test_non_stagnant)
        y_pred[~stagnant_mask] = y_pred_non_stagnant
    
    return y_pred

def build_ensemble_model(X_train, y_train, is_stagnant_col='is_stagnant'):
    """
    构建集成模型
    
    Args:
        X_train: 训练特征
        y_train: 训练标签
        is_stagnant_col: 静稳天气标记列
        
    Returns:
        ensemble_model: 训练好的集成模型
    """
    print("构建集成模型...")
    
    # 创建基础模型
    models = [
        ('lgb', lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, random_state=42)),
        ('xgb', xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, random_state=42)),
        ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
        ('gb', GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, random_state=42))
    ]
    
    # 为静稳天气和非静稳天气分别构建专门的模型
    specialized_models = build_specialized_models(X_train, y_train, is_stagnant_col)
    
    # 创建Stacking集成模型
    ensemble_model = StackingRegressor(
        estimators=models,
        final_estimator=Ridge(),
        cv=5,
        n_jobs=-1
    )
    
    # 训练集成模型（排除静稳天气列）
    if is_stagnant_col in X_train.columns:
        X_train_ens = X_train.drop(is_stagnant_col, axis=1)
    else:
        X_train_ens = X_train
    
    print("训练集成模型...")
    ensemble_model.fit(X_train_ens, y_train)
    
    # 保存模型
    os.makedirs("./processed2/models", exist_ok=True)
    joblib.dump(ensemble_model, "./processed2/models/ensemble_model.pkl")
    
    # 创建混合预测器，结合集成模型和专门模型
    class HybridPredictor:
        def __init__(self, ensemble_model, specialized_models, is_stagnant_col):
            self.ensemble_model = ensemble_model
            self.specialized_models = specialized_models
            self.is_stagnant_col = is_stagnant_col
        
        def predict(self, X):
            # 集成模型预测
            if self.is_stagnant_col in X.columns:
                X_ens = X.drop(self.is_stagnant_col, axis=1)
            else:
                X_ens = X
            ensemble_pred = self.ensemble_model.predict(X_ens)
            
            # 专门模型预测
            if self.is_stagnant_col in X.columns:
                specialized_pred = predict_with_specialized_models(X, self.specialized_models, self.is_stagnant_col)
                
                # 根据静稳天气标记决定使用哪个预测结果
                stagnant_mask = X[self.is_stagnant_col] == 1
                
                # 初始化混合预测结果
                hybrid_pred = ensemble_pred.copy()
                
                # 静稳天气样本使用专门模型和集成模型的加权平均
                if np.any(stagnant_mask):
                    hybrid_pred[stagnant_mask] = 0.7 * specialized_pred[stagnant_mask] + 0.3 * ensemble_pred[stagnant_mask]
                
                # 非静稳天气样本使用集成模型和专门模型的加权平均
                if np.any(~stagnant_mask):
                    hybrid_pred[~stagnant_mask] = 0.3 * specialized_pred[~stagnant_mask] + 0.7 * ensemble_pred[~stagnant_mask]
                
                return hybrid_pred
            else:
                return ensemble_pred
    
    # 创建混合预测器
    hybrid_model = HybridPredictor(ensemble_model, specialized_models, is_stagnant_col)
    
    # 保存混合模型
    joblib.dump(hybrid_model, "./processed2/models/hybrid_model.pkl")
    
    return hybrid_model

def evaluate_improved_models(X_test, y_test, orig_model, hybrid_model, output_dir="./processed2"):
    """
    评估改进模型的性能
    
    Args:
        X_test: 测试特征
        y_test: 测试标签
        orig_model: 原始模型
        hybrid_model: 混合模型
        output_dir: 输出目录
    """
    print("评估改进模型的性能...")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/figures", exist_ok=True)
    
    # 使用原始模型预测
    if 'is_stagnant' in X_test.columns:
        X_test_orig = X_test.drop('is_stagnant', axis=1)
    else:
        X_test_orig = X_test
    y_pred_orig = orig_model.predict(X_test_orig)
    
    # 使用混合模型预测
    y_pred_hybrid = hybrid_model.predict(X_test)
    
    # 评估原始模型性能
    orig_metrics = evaluate_model(y_test, y_pred_orig, "original_model")
    
    # 评估混合模型性能
    hybrid_metrics = evaluate_model(y_test, y_pred_hybrid, "hybrid_model")
    
    # 输出性能对比
    print("\n原始模型与改进模型性能对比:")
    print(f"原始模型 - RMSE: {orig_metrics['RMSE']:.2f}, MAE: {orig_metrics['MAE']:.2f}, MAPE: {orig_metrics['MAPE']:.2f}%, R2: {orig_metrics['R2']:.2f}")
    print(f"混合模型 - RMSE: {hybrid_metrics['RMSE']:.2f}, MAE: {hybrid_metrics['MAE']:.2f}, MAPE: {hybrid_metrics['MAPE']:.2f}%, R2: {hybrid_metrics['R2']:.2f}")
    
    # 计算改进百分比
    rmse_improvement = (orig_metrics['RMSE'] - hybrid_metrics['RMSE']) / orig_metrics['RMSE'] * 100
    mae_improvement = (orig_metrics['MAE'] - hybrid_metrics['MAE']) / orig_metrics['MAE'] * 100
    r2_improvement = (hybrid_metrics['R2'] - orig_metrics['R2'])
    
    print(f"\n性能改进:")
    print(f"RMSE减少: {rmse_improvement:.2f}%")
    print(f"MAE减少: {mae_improvement:.2f}%")
    print(f"R2增加: {r2_improvement:.4f}")
    
    # 分析静稳天气下的性能
    if 'is_stagnant' in X_test.columns:
        print("\n分析静稳天气下的模型性能...")
        
        # 原始模型
        X_test_with_stagnant = X_test.copy()
        analyze_static_weather_performance(X_test_with_stagnant, y_test, orig_model, 
                                         output_dir=f"{output_dir}/figures/original")
        
        # 混合模型
        analyze_static_weather_performance(X_test, y_test, hybrid_model, 
                                         output_dir=f"{output_dir}/figures/hybrid")
    
    # 可视化预测结果对比
    visualize_prediction_comparison(X_test, y_test, y_pred_orig, y_pred_hybrid, output_dir=f"{output_dir}/figures")
    
    # 保存评估结果
    results_df = pd.DataFrame([orig_metrics, hybrid_metrics])
    results_df.to_csv(f"{output_dir}/model_improvement_evaluation.csv", index=False)
    
    # 生成改进报告
    generate_improvement_report(orig_metrics, hybrid_metrics, rmse_improvement, mae_improvement, r2_improvement, output_dir)

def visualize_prediction_comparison(X_test, y_test, y_pred_orig, y_pred_hybrid, output_dir="./processed2/figures"):
    """
    可视化预测结果对比
    
    Args:
        X_test: 测试特征
        y_test: 测试标签
        y_pred_orig: 原始模型预测值
        y_pred_hybrid: 混合模型预测值
        output_dir: 输出目录
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 构建可视化数据框
    comparison_df = pd.DataFrame({
        'true': y_test,
        'original': y_pred_orig,
        'hybrid': y_pred_hybrid,
        'orig_error': np.abs(y_pred_orig - y_test),
        'hybrid_error': np.abs(y_pred_hybrid - y_test),
        'improvement': np.abs(y_pred_orig - y_test) - np.abs(y_pred_hybrid - y_test)
    })
    
    # 添加静稳天气标记
    if 'is_stagnant' in X_test.columns:
        comparison_df['is_stagnant'] = X_test['is_stagnant'].values
    
    # 添加日期时间(如果有)
    if 'datetime' in X_test.columns:
        comparison_df['datetime'] = X_test['datetime'].values
        comparison_df = comparison_df.sort_values('datetime')
    
    # 1. 原始模型vs混合模型预测散点图
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 2, 1)
    plt.scatter(comparison_df['true'], comparison_df['original'], alpha=0.5, label='原始模型')
    plt.scatter(comparison_df['true'], comparison_df['hybrid'], alpha=0.5, label='混合模型')
    
    # 添加对角线
    max_val = max(comparison_df['true'].max(), comparison_df['original'].max(), comparison_df['hybrid'].max())
    min_val = min(comparison_df['true'].min(), comparison_df['original'].min(), comparison_df['hybrid'].min())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.xlabel('真实值 (μg/m³)')
    plt.ylabel('预测值 (μg/m³)')
    plt.title('原始模型vs混合模型预测值对比')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # 2. 误差箱线图对比
    plt.subplot(2, 2, 2)
    error_compare = pd.DataFrame({
        '原始模型': comparison_df['orig_error'],
        '混合模型': comparison_df['hybrid_error']
    })
    error_compare.boxplot()
    plt.ylabel('绝对误差 (μg/m³)')
    plt.title('预测误差对比')
    plt.grid(alpha=0.3)
    
    # 3. 如果有静稳天气标记，分组对比误差
    if 'is_stagnant' in comparison_df.columns:
        plt.subplot(2, 2, 3)
        
        # 重组数据用于seaborn
        melt_df = pd.melt(comparison_df[['is_stagnant', 'orig_error', 'hybrid_error']], 
                         id_vars=['is_stagnant'], 
                         value_vars=['orig_error', 'hybrid_error'],
                         var_name='model', value_name='error')
        melt_df['model'] = melt_df['model'].map({'orig_error': '原始模型', 'hybrid_error': '混合模型'})
        
        sns.boxplot(x='is_stagnant', y='error', hue='model', data=melt_df)
        plt.xlabel('是否静稳天气 (1=是, 0=否)')
        plt.ylabel('绝对误差 (μg/m³)')
        plt.title('静稳天气条件下的预测误差对比')
        plt.legend(title='模型')
        plt.grid(alpha=0.3)
    
    # 4. 如果有日期时间，绘制时间序列误差对比
    if 'datetime' in comparison_df.columns:
        plt.subplot(2, 2, 4)
        
        # 使用部分数据点以避免图太密集
        if len(comparison_df) > 500:
            sample_size = 500
            sampled_df = comparison_df.sample(sample_size)
            sampled_df = sampled_df.sort_values('datetime')
        else:
            sampled_df = comparison_df
        
        plt.plot(sampled_df['datetime'], sampled_df['orig_error'], 'b-', alpha=0.6, label='原始模型误差')
        plt.plot(sampled_df['datetime'], sampled_df['hybrid_error'], 'r-', alpha=0.6, label='混合模型误差')
        plt.xlabel('日期')
        plt.ylabel('绝对误差 (μg/m³)')
        plt.title('预测误差随时间变化对比')
        plt.legend()
        plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/model_comparison.png", dpi=300)
    plt.close()
    
    # 5. 改进效果分布
    plt.figure(figsize=(10, 6))
    sns.histplot(comparison_df['improvement'], kde=True, bins=50)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.xlabel('误差改进 (μg/m³)')
    plt.ylabel('频次')
    plt.title('混合模型相对于原始模型的误差改进分布')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/error_improvement_distribution.png", dpi=300)
    plt.close()
    
    # 6. 如果有静稳天气标记，分析改进效果与静稳天气的关系
    if 'is_stagnant' in comparison_df.columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='is_stagnant', y='improvement', data=comparison_df)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('是否静稳天气 (1=是, 0=否)')
        plt.ylabel('误差改进 (μg/m³)')
        plt.title('静稳天气条件下的误差改进对比')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/error_improvement_by_stagnant.png", dpi=300)
        plt.close()

def generate_improvement_report(orig_metrics, hybrid_metrics, rmse_improvement, mae_improvement, r2_improvement, output_dir="./processed2"):
    """
    生成改进报告
    
    Args:
        orig_metrics: 原始模型评估指标
        hybrid_metrics: 混合模型评估指标
        rmse_improvement: RMSE改进百分比
        mae_improvement: MAE改进百分比
        r2_improvement: R2改进值
        output_dir: 输出目录
    """
    with open(f"{output_dir}/model_improvement_report.txt", "w", encoding="utf-8") as f:
        f.write("PM2.5预测模型改进报告\n")
        f.write("=" * 50 + "\n\n")
        
        # 1. 改进模型概述
        f.write("1. 改进模型概述\n")
        f.write("-" * 50 + "\n\n")
        
        f.write("针对静稳天气条件下PM2.5预测误差较大的问题，我们采用了以下改进策略：\n")
        f.write("1. 为静稳天气和非静稳天气分别构建专门的预测子模型\n")
        f.write("2. 将多种基础模型（LightGBM、XGBoost、RandomForest、GradientBoosting）通过Stacking方法集成\n")
        f.write("3. 设计混合预测器，根据天气条件动态结合专门模型和集成模型的预测结果\n")
        f.write("4. 对静稳天气样本，赋予专门模型更高的权重；对非静稳天气样本，赋予集成模型更高的权重\n\n")
        
        # 2. 性能对比
        f.write("2. 性能对比\n")
        f.write("-" * 50 + "\n\n")
        
        f.write("原始模型性能：\n")
        f.write(f"RMSE: {orig_metrics['RMSE']:.2f}\n")
        f.write(f"MAE: {orig_metrics['MAE']:.2f}\n")
        f.write(f"MAPE: {orig_metrics['MAPE']:.2f}%\n")
        f.write(f"R2: {orig_metrics['R2']:.4f}\n\n")
        
        f.write("改进模型性能：\n")
        f.write(f"RMSE: {hybrid_metrics['RMSE']:.2f}\n")
        f.write(f"MAE: {hybrid_metrics['MAE']:.2f}\n")
        f.write(f"MAPE: {hybrid_metrics['MAPE']:.2f}%\n")
        f.write(f"R2: {hybrid_metrics['R2']:.4f}\n\n")
        
        f.write("性能改进：\n")
        f.write(f"RMSE减少: {rmse_improvement:.2f}%\n")
        f.write(f"MAE减少: {mae_improvement:.2f}%\n")
        f.write(f"R2增加: {r2_improvement:.4f}\n\n")
        
        # 3. 结论与建议
        f.write("3. 结论与建议\n")
        f.write("-" * 50 + "\n\n")
        
        # 根据改进效果给出不同的结论
        if rmse_improvement > 10:  # 如果RMSE改进超过10%
            f.write("主要结论：\n")
            f.write("1. 混合模型策略显著提高了PM2.5预测精度，尤其在静稳天气条件下。\n")
            f.write(f"2. 整体而言，混合模型使RMSE降低了{rmse_improvement:.2f}%，表明预测误差显著减小。\n")
            f.write("3. 为不同气象条件构建专门模型的策略证明是有效的，能够更好地适应复杂多变的污染扩散条件。\n\n")
            
            f.write("进一步改进建议：\n")
            f.write("1. 考虑引入更多描述静稳天气特征的变量，如边界层高度、大气稳定度指数等。\n")
            f.write("2. 优化静稳天气与非静稳天气模型的集成权重，可以探索自适应加权方案。\n")
            f.write("3. 进一步细分静稳天气类型（如持续时间、程度等），为更具体的气象条件构建专门模型。\n")
            f.write("4. 探索深度学习方法，尤其是能够捕捉时序特征的LSTM、GRU等模型，可能在处理复杂的时间序列预测任务时有更好的表现。\n")
        elif rmse_improvement > 5:  # 如果RMSE改进在5%到10%之间
            f.write("主要结论：\n")
            f.write("1. 混合模型策略有效提高了PM2.5预测精度，特别是对静稳天气条件的适应性有所提升。\n")
            f.write(f"2. 整体而言，混合模型使RMSE降低了{rmse_improvement:.2f}%，表明预测效果有明显改善。\n")
            f.write("3. 针对性建模策略证明是有价值的，为复杂气象条件下的污染预测提供了新思路。\n\n")
            
            f.write("进一步改进建议：\n")
            f.write("1. 进一步优化静稳天气专门模型的特征选择和参数调优，提高其预测精度。\n")
            f.write("2. 探索更复杂的集成方法，如动态加权或基于局部模型性能的自适应选择。\n")
            f.write("3. 引入更多与污染物扩散相关的特征，提高模型对复杂扩散环境的理解能力。\n")
        else:  # 如果RMSE改进不到5%
            f.write("主要结论：\n")
            f.write("1. 混合模型策略对PM2.5预测有一定改进，但效果不够显著。\n")
            f.write(f"2. 整体而言，混合模型使RMSE降低了{rmse_improvement:.2f}%，改进空间仍然较大。\n")
            f.write("3. 仅通过模型集成和分类预测可能难以充分解决静稳天气下的预测难题，需要更深入的改进。\n\n")
            
            f.write("进一步改进建议：\n")
            f.write("1. 重新审视特征工程，引入更多关键要素，特别是能够表征静稳天气污染物累积效应的特征。\n")
            f.write("2. 考虑使用深度学习方法，尤其是时序模型（LSTM、Transformer等），以更好地捕捉时间相关性。\n")
            f.write("3. 探索引入外部数据源，如卫星遥感数据、交通流量数据等，提供更全面的污染物扩散环境信息。\n")
            f.write("4. 考虑多尺度建模方法，分别建模短期波动和长期趋势，再进行集成。\n")
    
    print(f"改进报告已保存至 {output_dir}/model_improvement_report.txt")

def main():
    """
    主函数
    """
    # 加载处理好的训练和测试数据
    print("加载处理好的特征数据...")
    try:
        processed_train = pd.read_csv("./processed2/processed_train.csv")
        processed_test = pd.read_csv("./processed2/processed_test.csv")
    except FileNotFoundError:
        print("错误：找不到处理好的特征数据，请先运行特征工程步骤")
        return
    
    # 确保datetime列是日期时间类型
    for df in [processed_train, processed_test]:
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
    
    # 准备特征和标签
    print("准备特征和标签...")
    
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
    
    # 加载原始模型
    print("加载原始最佳模型...")
    try:
        orig_model = joblib.load("./processed2/models/XGB_tuned_model.pkl")
    except FileNotFoundError:
        print("警告：找不到调优后的XGB模型，尝试加载基础XGB模型...")
        try:
            orig_model = joblib.load("./processed2/models/XGB_model.pkl")
        except FileNotFoundError:
            print("错误：找不到XGB模型，请先运行模型训练步骤")
            return
    
    # 构建混合模型
    print("构建混合模型...")
    hybrid_model = build_ensemble_model(X_train, y_train)
    
    # 评估模型性能
    print("评估改进后的模型性能...")
    evaluate_improved_models(X_test, y_test, orig_model, hybrid_model)
    
    print("\n模型改进完成!")

if __name__ == "__main__":
    main() 