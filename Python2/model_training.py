#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
模型训练模块
训练多种模型预测PM2.5，并评估各模型性能
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import joblib
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, train_test_split
import xgboost as xgb
import lightgbm as lgb
from tqdm import tqdm

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 定义要训练的模型
def get_base_models():
    """
    获取基础模型列表
    
    Returns:
        models: 模型字典
    """
    models = {
        'Linear': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.1),
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5),
        'RF': RandomForestRegressor(n_estimators=100, random_state=42),
        'GB': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'XGB': xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
        'LGB': lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
        'ExtraTrees': ExtraTreesRegressor(n_estimators=100, random_state=42),
        'KNN': KNeighborsRegressor(n_neighbors=5)
    }
    return models

def evaluate_model(y_true, y_pred, model_name):
    """
    评估模型性能
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        model_name: 模型名称
        
    Returns:
        metrics: 评估指标字典
    """
    # 计算评估指标
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # 计算MAPE(平均绝对百分比误差)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
    
    metrics = {
        'model': model_name,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R2': r2
    }
    
    return metrics

def train_models(X_train, y_train, X_test, y_test, models=None, output_dir="./processed2/models"):
    """
    训练多个模型并评估性能
    
    Args:
        X_train: 训练特征
        y_train: 训练标签
        X_test: 测试特征
        y_test: 测试标签
        models: 要训练的模型字典，如果为None则使用默认模型
        output_dir: 模型保存目录
        
    Returns:
        results_df: 模型评估结果数据框
        trained_models: 训练好的模型字典
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 如果没有指定模型，使用默认模型
    if models is None:
        models = get_base_models()
    
    # 确保训练集和测试集有相同的特征
    missing_cols = set(X_train.columns) - set(X_test.columns)
    if missing_cols:
        print(f"警告：测试集缺少以下特征列：{missing_cols}")
        # 为测试集添加缺失的列，并填充0
        for col in missing_cols:
            X_test[col] = 0
        print("已为测试集添加缺失的特征列并填充0")
    
    # 确保列的顺序一致
    X_test = X_test[X_train.columns]
    print(f"训练集特征数：{X_train.shape[1]}，测试集特征数：{X_test.shape[1]}")
    
    results = []
    trained_models = {}
    
    # 遍历所有模型进行训练和评估
    for name, model in tqdm(models.items(), desc="训练模型"):
        print(f"\n训练模型: {name}")
        
        # 训练模型
        model.fit(X_train, y_train)
        
        # 预测训练集和测试集
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        # 评估训练集性能
        train_metrics = evaluate_model(y_train, train_pred, f"{name}_train")
        
        # 评估测试集性能
        test_metrics = evaluate_model(y_test, test_pred, f"{name}_test")
        
        # 保存结果
        results.append(train_metrics)
        results.append(test_metrics)
        
        # 保存模型
        trained_models[name] = model
        joblib.dump(model, f"{output_dir}/{name}_model.pkl")
        
        print(f"模型 {name} 训练完成")
        print(f"训练集 RMSE: {train_metrics['RMSE']:.2f}, MAE: {train_metrics['MAE']:.2f}, R2: {train_metrics['R2']:.2f}")
        print(f"测试集 RMSE: {test_metrics['RMSE']:.2f}, MAE: {test_metrics['MAE']:.2f}, R2: {test_metrics['R2']:.2f}")
    
    # 将结果转换为数据框
    results_df = pd.DataFrame(results)
    
    # 保存评估结果
    results_df.to_csv(f"{output_dir}/model_evaluation.csv", index=False)
    
    return results_df, trained_models

def tune_hyperparameters(X_train, y_train, model_type='RF', n_splits=5, output_dir="./processed2/models"):
    """
    使用时间序列交叉验证调优超参数
    
    Args:
        X_train: 训练特征
        y_train: 训练标签
        model_type: 模型类型
        n_splits: 交叉验证折数
        output_dir: 输出目录
        
    Returns:
        best_model: 调优后的最佳模型
        train_cols: 训练时使用的特征列名
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存训练列名，用于后续预测
    train_cols = X_train.columns.tolist()
    
    # 时间序列交叉验证
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    print(f"开始为{model_type}模型调优超参数...")
    
    # 根据模型类型设置参数网格
    if model_type == 'RF':
        model = RandomForestRegressor(random_state=42)
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['auto', 'sqrt'],  # 限制每个决策树使用的特征数量
            'bootstrap': [True]  # 使用bootstrap样本
        }
    elif model_type == 'XGB':
        model = xgb.XGBRegressor(random_state=42)
        param_grid = {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'subsample': [0.7, 0.8],
            'colsample_bytree': [0.7, 0.8],  # 控制每棵树的特征采样
            'reg_alpha': [0.01, 0.1, 1],     # L1正则化
            'reg_lambda': [0.01, 0.1, 1]     # L2正则化
        }
    elif model_type == 'LGB':
        model = lgb.LGBMRegressor(random_state=42)
        param_grid = {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'num_leaves': [20, 31],
            'subsample': [0.7, 0.8],
            'reg_alpha': [0.01, 0.1],     # L1正则化
            'reg_lambda': [0.01, 0.1]     # L2正则化
        }
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    # 对所有模型统一使用网格搜索
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=tscv,
        scoring='neg_root_mean_squared_error',
        n_jobs=1,  # 设置为1以禁用并行处理，避免中文路径编码问题
        verbose=1
    )
    
    # 执行网格搜索
    grid_search.fit(X_train, y_train)
    
    # 获取最佳参数和模型
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_
    
    print(f"最佳参数: {best_params}")
    print(f"最佳交叉验证分数: {-grid_search.best_score_:.3f} RMSE")
    
    # 保存调优结果
    cv_results = pd.DataFrame(grid_search.cv_results_)
    cv_results.to_csv(f"{output_dir}/{model_type}_cv_results.csv", index=False)
    
    # 保存最佳模型
    joblib.dump(best_model, f"{output_dir}/{model_type}_tuned_model.pkl")
    
    # 保存训练时使用的特征列
    with open(f"{output_dir}/{model_type}_feature_columns.txt", "w") as f:
        f.write("\n".join(train_cols))
    
    return best_model, train_cols

def analyze_static_weather_performance(X_test, y_test, model, 
                                       is_stagnant_col='is_stagnant',
                                       output_dir="./processed2/figures",
                                       train_cols=None):
    """
    分析模型在静稳天气下的预测性能
    
    Args:
        X_test: 测试特征
        y_test: 测试标签
        model: 训练好的模型
        is_stagnant_col: 静稳天气标记列
        output_dir: 输出目录
        train_cols: 训练时使用的特征列名
        
    Returns:
        performance: 静稳天气和非静稳天气下的性能指标
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存is_stagnant列
    is_stagnant_values = None
    if is_stagnant_col in X_test.columns:
        is_stagnant_values = X_test[is_stagnant_col].copy()
    
    # 准备用于预测的特征集
    X_test_pred = X_test.copy()
    
    # 确保测试集特征与训练时一致
    if train_cols is not None:
        # 只保留训练时使用的特征列
        keep_cols = [col for col in X_test_pred.columns if col in train_cols or col == is_stagnant_col]
        X_test_pred = X_test_pred[keep_cols]
        
        # 添加缺失的列
        missing_cols = set(train_cols) - set(X_test_pred.columns)
        for col in missing_cols:
            X_test_pred[col] = 0
        
        # 按训练时的列顺序排列，但仅选择模型需要的特征列进行预测
        predict_cols = [col for col in train_cols if col in X_test_pred.columns]
        X_predict = X_test_pred[predict_cols]
    else:
        # 如果没有提供train_cols，尝试移除非数值列
        numeric_cols = X_test_pred.select_dtypes(include=['int64', 'float64']).columns
        X_predict = X_test_pred[numeric_cols]
    
    # 预测
    y_pred = model.predict(X_predict)
    
    # 构建分析用数据框
    analysis_df = pd.DataFrame({
        'true': y_test,
        'pred': y_pred,
        'error': np.abs(y_pred - y_test),
        'error_pct': np.abs((y_pred - y_test) / (y_test + 1e-10)) * 100
    })
    
    # 如果静稳天气列在X_test中
    if is_stagnant_values is not None:
        analysis_df['is_stagnant'] = is_stagnant_values
        
        # 按静稳天气分组计算性能指标
        performance = {}
        for stagnant in [0, 1]:
            mask = analysis_df['is_stagnant'] == stagnant
            group_df = analysis_df[mask]
            
            if len(group_df) > 0:
                rmse = np.sqrt(mean_squared_error(group_df['true'], group_df['pred']))
                mae = mean_absolute_error(group_df['true'], group_df['pred'])
                mape = group_df['error_pct'].mean()
                r2 = r2_score(group_df['true'], group_df['pred'])
                
                performance[f'stagnant_{stagnant}'] = {
                    'count': len(group_df),
                    'RMSE': rmse,
                    'MAE': mae,
                    'MAPE': mape,
                    'R2': r2
                }
        
        # 输出性能比较
        print("\n静稳天气与非静稳天气下的预测性能对比:")
        print(f"非静稳天气 (样本数: {performance['stagnant_0']['count']})")
        print(f"  RMSE: {performance['stagnant_0']['RMSE']:.2f}")
        print(f"  MAE: {performance['stagnant_0']['MAE']:.2f}")
        print(f"  MAPE: {performance['stagnant_0']['MAPE']:.2f}%")
        print(f"  R2: {performance['stagnant_0']['R2']:.2f}")
        
        print(f"静稳天气 (样本数: {performance['stagnant_1']['count']})")
        print(f"  RMSE: {performance['stagnant_1']['RMSE']:.2f}")
        print(f"  MAE: {performance['stagnant_1']['MAE']:.2f}")
        print(f"  MAPE: {performance['stagnant_1']['MAPE']:.2f}%")
        print(f"  R2: {performance['stagnant_1']['R2']:.2f}")
        
        # 计算性能差异
        rmse_diff = performance['stagnant_1']['RMSE'] / performance['stagnant_0']['RMSE'] - 1
        mae_diff = performance['stagnant_1']['MAE'] / performance['stagnant_0']['MAE'] - 1
        mape_diff = performance['stagnant_1']['MAPE'] / performance['stagnant_0']['MAPE'] - 1
        r2_diff = performance['stagnant_0']['R2'] - performance['stagnant_1']['R2']
        
        print(f"\n静稳天气相对于非静稳天气的性能变化:")
        print(f"  RMSE增加: {rmse_diff:.2%}")
        print(f"  MAE增加: {mae_diff:.2%}")
        print(f"  MAPE增加: {mape_diff:.2%}")
        print(f"  R2减少: {r2_diff:.2f}")
        
        # 可视化静稳天气和非静稳天气下的预测误差分布
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='is_stagnant', y='error', data=analysis_df)
        plt.xlabel('是否静稳天气 (1=是, 0=否)')
        plt.ylabel('绝对误差 (μg/m³)')
        plt.title('静稳天气与非静稳天气下的预测误差分布')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/prediction_error_by_stagnant.png", dpi=300)
        plt.close()
        
        # 可视化不同PM2.5浓度区间下的预测误差
        analysis_df['pm25_bin'] = pd.cut(analysis_df['true'], bins=[0, 35, 75, 115, 150, 250, 500], 
                                    labels=['优', '良', '轻度污染', '中度污染', '重度污染', '严重污染'])
        
        plt.figure(figsize=(14, 7))
        sns.boxplot(x='pm25_bin', y='error', hue='is_stagnant', data=analysis_df)
        plt.xlabel('PM2.5浓度等级')
        plt.ylabel('绝对误差 (μg/m³)')
        plt.title('不同PM2.5浓度等级下的预测误差对比')
        plt.legend(title='是否静稳天气', labels=['否', '是'])
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/prediction_error_by_pm25_level.png", dpi=300)
        plt.close()
        
        # 保存分析结果
        with open(f"./processed2/stagnant_weather_performance.txt", "w", encoding="utf-8") as f:
            f.write("静稳天气与非静稳天气下的预测性能对比\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"非静稳天气 (样本数: {performance['stagnant_0']['count']})\n")
            f.write(f"  RMSE: {performance['stagnant_0']['RMSE']:.2f}\n")
            f.write(f"  MAE: {performance['stagnant_0']['MAE']:.2f}\n")
            f.write(f"  MAPE: {performance['stagnant_0']['MAPE']:.2f}%\n")
            f.write(f"  R2: {performance['stagnant_0']['R2']:.2f}\n\n")
            
            f.write(f"静稳天气 (样本数: {performance['stagnant_1']['count']})\n")
            f.write(f"  RMSE: {performance['stagnant_1']['RMSE']:.2f}\n")
            f.write(f"  MAE: {performance['stagnant_1']['MAE']:.2f}\n")
            f.write(f"  MAPE: {performance['stagnant_1']['MAPE']:.2f}%\n")
            f.write(f"  R2: {performance['stagnant_1']['R2']:.2f}\n\n")
            
            f.write(f"静稳天气相对于非静稳天气的性能变化:\n")
            f.write(f"  RMSE增加: {rmse_diff:.2%}\n")
            f.write(f"  MAE增加: {mae_diff:.2%}\n")
            f.write(f"  MAPE增加: {mape_diff:.2%}\n")
            f.write(f"  R2减少: {r2_diff:.2f}\n")
            
        return performance
    else:
        print(f"警告: 在特征中未找到静稳天气列 '{is_stagnant_col}'，无法分析静稳天气性能")
        return None

def visualize_predictions(X_test, y_test, model, datetime_col='datetime', output_dir="./processed2/figures", train_cols=None):
    """
    可视化预测结果
    
    Args:
        X_test: 测试特征
        y_test: 测试标签
        model: 训练好的模型
        datetime_col: 日期时间列
        output_dir: 输出目录
        train_cols: 训练时使用的特征列名
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存datetime列，稍后用于可视化
    datetime_values = None
    if datetime_col in X_test.columns:
        datetime_values = X_test[datetime_col].copy()
        X_test_pred = X_test.drop(columns=[datetime_col]).copy()
    else:
        X_test_pred = X_test.copy()
    
    # 确保测试集特征与训练时一致
    if train_cols is not None:
        # 添加缺失的列
        missing_cols = set(train_cols) - set(X_test_pred.columns)
        for col in missing_cols:
            X_test_pred[col] = 0
        
        # 删除多余的列
        extra_cols = set(X_test_pred.columns) - set(train_cols)
        if extra_cols:
            X_test_pred = X_test_pred.drop(columns=extra_cols)
        
        # 按训练时的列顺序排列
        X_test_pred = X_test_pred[train_cols]
    
    # 预测
    y_pred = model.predict(X_test_pred)
    
    # 构建可视化用数据框
    vis_df = pd.DataFrame({
        'true': y_test,
        'pred': y_pred,
        'error': y_pred - y_test,
        'abs_error': np.abs(y_pred - y_test)
    })
    
    # 如果有日期时间值，添加到可视化数据框
    if datetime_values is not None:
        vis_df['datetime'] = datetime_values
        vis_df = vis_df.sort_values('datetime')
    
    # 1. 真实值vs预测值散点图
    plt.figure(figsize=(10, 8))
    plt.scatter(vis_df['true'], vis_df['pred'], alpha=0.5)
    
    # 添加对角线
    max_val = max(vis_df['true'].max(), vis_df['pred'].max())
    min_val = min(vis_df['true'].min(), vis_df['pred'].min())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.xlabel('真实值 (μg/m³)')
    plt.ylabel('预测值 (μg/m³)')
    plt.title('PM2.5预测值vs真实值')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/pred_vs_true_scatter.png", dpi=300)
    plt.close()
    
    # 2. 时间序列对比图
    if 'datetime' in vis_df.columns:
        # 使用一部分数据以避免图太密集
        if len(vis_df) > 1000:
            # 按月抽样
            vis_df['month'] = vis_df['datetime'].dt.to_period('M')
            sample_df = vis_df.groupby('month').apply(lambda x: x.sample(min(30, len(x)))).reset_index(drop=True)
            sample_df = sample_df.sort_values('datetime')
        else:
            sample_df = vis_df
        
        plt.figure(figsize=(15, 8))
        plt.plot(sample_df['datetime'], sample_df['true'], 'b-', label='真实值')
        plt.plot(sample_df['datetime'], sample_df['pred'], 'r-', label='预测值')
        plt.xlabel('日期')
        plt.ylabel('PM2.5浓度 (μg/m³)')
        plt.title('PM2.5浓度预测时间序列对比')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/pred_vs_true_timeseries.png", dpi=300)
        plt.close()
        
        # 3. 误差随时间变化图
        plt.figure(figsize=(15, 8))
        plt.scatter(sample_df['datetime'], sample_df['error'], alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='-')
        plt.xlabel('日期')
        plt.ylabel('预测误差 (μg/m³)')
        plt.title('PM2.5预测误差随时间变化')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/prediction_error_timeseries.png", dpi=300)
        plt.close()
        
        # 4. 按月份的误差箱线图
        vis_df['month'] = vis_df['datetime'].dt.month
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='month', y='abs_error', data=vis_df)
        plt.xlabel('月份')
        plt.ylabel('绝对误差 (μg/m³)')
        plt.title('各月份的预测误差分布')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/prediction_error_by_month.png", dpi=300)
        plt.close()
    
    # 5. 误差分布直方图
    plt.figure(figsize=(10, 6))
    sns.histplot(vis_df['error'], kde=True, bins=50)
    plt.xlabel('预测误差 (μg/m³)')
    plt.ylabel('频次')
    plt.title('PM2.5预测误差分布')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/prediction_error_histogram.png", dpi=300)
    plt.close()

def main():
    """
    主函数
    """
    # 加载处理好的特征数据
    print("加载处理好的特征数据...")
    train_data = pd.read_csv("./processed2/processed_train.csv")
    test_data = pd.read_csv("./processed2/processed_test.csv")
    
    # 确保datetime列是日期时间类型
    for df in [train_data, test_data]:
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
    
    # 准备特征和标签
    print("准备特征和标签...")
    
    # 排除不用于训练的列
    exclude_cols = ['datetime']
    target_col = 'PM2.5'
    
    # 为训练准备数据
    X_train = train_data.drop(exclude_cols + [target_col], axis=1)
    y_train = train_data[target_col]
    
    X_test = test_data.drop(exclude_cols + [target_col], axis=1)
    y_test = test_data[target_col]
    
    print(f"训练集特征形状: {X_train.shape}")
    print(f"测试集特征形状: {X_test.shape}")
    
    # 训练基础模型
    print("训练基础模型...")
    model_results, trained_models = train_models(X_train, y_train, X_test, y_test)
    
    # 为最佳模型调优超参数
    best_model_name = model_results[model_results['model'].str.contains('test')]['RMSE'].idxmin()
    best_model_name = model_results.loc[best_model_name, 'model'].split('_')[0]
    print(f"\n最佳基础模型: {best_model_name}")
    
    if best_model_name in ['RF', 'XGB', 'LGB']:
        print(f"\n为{best_model_name}模型调优超参数...")
        tuned_model, train_cols = tune_hyperparameters(X_train, y_train, model_type=best_model_name)
        
        # 评估调优后的模型
        test_pred = tuned_model.predict(X_test)
        tuned_metrics = evaluate_model(y_test, test_pred, f"{best_model_name}_tuned_test")
        print(f"\n调优后的{best_model_name}模型性能:")
        print(f"RMSE: {tuned_metrics['RMSE']:.2f}")
        print(f"MAE: {tuned_metrics['MAE']:.2f}")
        print(f"R2: {tuned_metrics['R2']:.2f}")
        
        # 分析静稳天气下的性能
        print("\n分析静稳天气下的模型性能...")
        analyze_static_weather_performance(X_test, y_test, tuned_model, train_cols=train_cols)
        
        # 可视化预测结果
        print("\n可视化预测结果...")
        # 添加datetime列用于可视化
        X_test_with_dt = X_test.copy()
        X_test_with_dt['datetime'] = test_data['datetime']
        visualize_predictions(X_test_with_dt, y_test, tuned_model)
    else:
        # 使用未调优的最佳模型
        best_model = trained_models[best_model_name]
        
        # 分析静稳天气下的性能
        print("\n分析静稳天气下的模型性能...")
        analyze_static_weather_performance(X_test, y_test, best_model)
        
        # 可视化预测结果
        print("\n可视化预测结果...")
        # 添加datetime列用于可视化
        X_test_with_dt = X_test.copy()
        X_test_with_dt['datetime'] = test_data['datetime']
        visualize_predictions(X_test_with_dt, y_test, best_model)
    
    print("\n模型训练和评估完成!")

if __name__ == "__main__":
    main() 