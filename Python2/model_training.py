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
        plt.ylabel('绝对误差 (ug/m3)')
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
        plt.ylabel('绝对误差 (ug/m3)')
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

def visualize_predictions(test_data, y_test, model, train_cols=None, output_dir='output'):
    """可视化预测结果"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取预测值
    if train_cols is None:
        X_test = test_data.copy()
    else:
        X_test = test_data[train_cols].copy()
    
    y_pred = model.predict(X_test)
    
    # 创建可视化数据框
    vis_df = pd.DataFrame({
        'datetime': test_data['datetime'],
        'true': y_test,
        'pred': y_pred,
        'error': y_pred - y_test,
        'abs_error': np.abs(y_pred - y_test),
        'is_stagnant': test_data['is_stagnant'] if 'is_stagnant' in test_data.columns else 0
    })
    
    # 定义月份颜色 - 提前定义，避免引用错误
    month_colors = plt.cm.viridis(np.linspace(0.1, 0.9, 12))
    
    # 1. 真实值vs预测值散点图
    plt.figure(figsize=(14, 12))
    
    # 散点图使用透明度和颜色渐变以展示密度
    density_cmap = plt.cm.viridis
    scatter = plt.scatter(vis_df['true'], vis_df['pred'], 
                         alpha=0.6, 
                         c=vis_df['abs_error'], 
                         cmap=density_cmap,
                         s=30,
                         edgecolor='white',
                         linewidth=0.5)
    
    # 添加线性回归拟合线
    z = np.polyfit(vis_df['true'], vis_df['pred'], 1)
    p = np.poly1d(z)
    x_range = np.linspace(vis_df['true'].min(), vis_df['true'].max(), 100)
    plt.plot(x_range, p(x_range), 'r--', linewidth=2, 
             label=f'拟合线: y={z[0]:.3f}x+{z[1]:.3f}')
    
    # 添加对角线
    max_val = max(vis_df['true'].max(), vis_df['pred'].max())
    min_val = min(vis_df['true'].min(), vis_df['pred'].min())
    plt.plot([min_val, max_val], [min_val, max_val], 'g-', 
             linewidth=2, label='理想预测线 (y=x)')
    
    # 添加颜色条
    cbar = plt.colorbar(scatter)
    cbar.set_label('预测误差 (ug/m3)', fontsize=12, labelpad=10)
    
    # 设置轴标签和标题
    plt.xlabel('真实值 (ug/m3)', fontsize=14, labelpad=10, color='#333333')
    plt.ylabel('预测值 (ug/m3)', fontsize=14, labelpad=10, color='#333333')
    plt.title('PM2.5预测值与真实值对比', fontsize=18, pad=20, fontweight='bold', color='#333333')
    
    # 添加图例和网格
    plt.legend(fontsize=12, loc='upper left')
    plt.grid(alpha=0.3, linestyle='--')
    plt.tick_params(axis='both', which='major', labelsize=12)
    
    # 添加统计信息
    r2 = r2_score(vis_df['true'], vis_df['pred'])
    rmse = np.sqrt(mean_squared_error(vis_df['true'], vis_df['pred']))
    mae = mean_absolute_error(vis_df['true'], vis_df['pred'])
    
    stats_text = (f"R2: {r2:.4f}\n"
                 f"RMSE: {rmse:.2f} ug/m3\n"
                 f"MAE: {mae:.2f} ug/m3\n"
                 f"样本数: {len(vis_df)}")
    
    plt.annotate(stats_text, xy=(0.02, 0.95), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8),
                fontsize=12, verticalalignment='top')
    
    plt.tight_layout(pad=2.0)
    plt.savefig(f"{output_dir}/pred_vs_true_scatter.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/pred_vs_true_scatter.pdf", format='pdf', bbox_inches='tight')
    plt.close()
    
    # 2. 时间序列对比图
    if 'datetime' in vis_df.columns:
        # 使用一部分数据以避免图太密集
        if len(vis_df) > 1000:
            # 按月抽样以保持时间分布
            vis_df['month'] = vis_df['datetime'].dt.to_period('M')
            sample_df = vis_df.groupby('month').apply(lambda x: x.sample(min(30, len(x)))).reset_index(drop=True)
            sample_df = sample_df.sort_values('datetime')
        else:
            sample_df = vis_df
        
        plt.figure(figsize=(16, 10))
        
        # 绘制真实值和预测值，使用更好的颜色和线宽
        plt.plot(sample_df['datetime'], sample_df['true'], 'o-', 
                color='#3498db', linewidth=2, markersize=6, alpha=0.7, label='真实值')
        plt.plot(sample_df['datetime'], sample_df['pred'], 'o-', 
                color='#e74c3c', linewidth=2, markersize=6, alpha=0.7, label='预测值')
        
        # 添加误差带
        plt.fill_between(sample_df['datetime'], 
                        sample_df['pred'] - sample_df['abs_error'], 
                        sample_df['pred'] + sample_df['abs_error'], 
                        color='#e74c3c', alpha=0.2)
        
        # 设置轴标签和标题
        plt.xlabel('日期', fontsize=14, labelpad=10, color='#333333')
        plt.ylabel('PM2.5浓度 (ug/m3)', fontsize=14, labelpad=10, color='#333333')
        plt.title('PM2.5浓度预测时间序列对比', fontsize=18, pad=20, fontweight='bold', color='#333333')
        
        # 添加图例和网格
        plt.legend(fontsize=14, loc='upper right')
        plt.grid(alpha=0.3, linestyle='--')
        plt.tick_params(axis='both', which='major', labelsize=12)
        
        # 自动格式化日期标签
        fig = plt.gcf()
        fig.autofmt_xdate()
        
        # 添加注释，说明只显示了部分数据点
        if len(vis_df) > len(sample_df):
            plt.figtext(0.5, 0.01, 
                      f"注: 为保持清晰度，图表只显示了{len(sample_df)}个采样点（总共{len(vis_df)}个数据点）", 
                      ha='center', fontsize=10, 
                      bbox=dict(facecolor='#eeeeee', alpha=0.5, boxstyle='round,pad=0.5'))
        
        plt.tight_layout(pad=2.0)
        plt.savefig(f"{output_dir}/pred_vs_true_timeseries.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{output_dir}/pred_vs_true_timeseries.pdf", format='pdf', bbox_inches='tight')
        plt.close()
        
        # 3. 误差随时间变化图
        plt.figure(figsize=(16, 8))
        
        # 创建渐变颜色映射
        cmap = plt.cm.coolwarm
        norm = plt.Normalize(vis_df['error'].min(), vis_df['error'].max())
        colors = [cmap(norm(err)) for err in sample_df['error']]
        
        # 绘制误差点
        plt.scatter(sample_df['datetime'], sample_df['error'], 
                   c=colors, s=50, alpha=0.7, edgecolor='white', linewidth=0.5)
        
        # 添加零线和移动平均线
        plt.axhline(y=0, color='#2c3e50', linestyle='-', linewidth=1.5)
        
        # 计算并绘制移动平均误差
        window = min(30, len(sample_df) // 5)
        sample_df['error_ma'] = sample_df['error'].rolling(window=window, center=True).mean()
        plt.plot(sample_df['datetime'], sample_df['error_ma'], 
               'r-', linewidth=2, label=f'{window}点移动平均')
        
        # 设置轴标签和标题
        plt.xlabel('日期', fontsize=14, labelpad=10, color='#333333')
        plt.ylabel('预测误差 (ug/m3)', fontsize=14, labelpad=10, color='#333333')
        plt.title('PM2.5预测误差随时间变化', fontsize=18, pad=20, fontweight='bold', color='#333333')
        
        # 添加网格和图例
        plt.grid(alpha=0.3, linestyle='--')
        plt.legend(fontsize=12)
        plt.tick_params(axis='both', which='major', labelsize=12)
        
        # 自动格式化日期标签
        fig = plt.gcf()
        fig.autofmt_xdate()
        
        plt.tight_layout(pad=2.0)
        plt.savefig(f"{output_dir}/prediction_error_timeseries.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{output_dir}/prediction_error_timeseries.pdf", format='pdf', bbox_inches='tight')
        plt.close()
        
        # 4. 各月份误差箱线图
        vis_df['month'] = vis_df['datetime'].dt.month
        
        plt.figure(figsize=(16, 8))
        
        # 修复palette警告，使用hue参数
        ax = sns.boxplot(x='month', y='abs_error', data=vis_df,
                       hue='month',  # 添加hue参数
                       palette=list(month_colors),  # 将颜色转换为列表
                       width=0.6, linewidth=1.2,
                       legend=False)  # 不显示图例
        
        # 添加散点显示实际分布
        sns.stripplot(x='month', y='abs_error', data=vis_df.sample(min(500, len(vis_df))), 
                     size=3, color='black', alpha=0.4, jitter=True)
        
        # 设置月份标签
        month_names = ['一月', '二月', '三月', '四月', '五月', '六月', 
                      '七月', '八月', '九月', '十月', '十一月', '十二月']
        plt.xticks(range(12), month_names, rotation=45, ha='right')
        
        # 设置轴标签和标题
        plt.xlabel('月份', fontsize=14, labelpad=10, color='#333333')
        plt.ylabel('绝对误差 (ug/m3)', fontsize=14, labelpad=10, color='#333333')
        plt.title('各月份的PM2.5预测误差分布', fontsize=18, pad=20, fontweight='bold', color='#333333')
        
        # 添加网格线
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tick_params(axis='both', which='major', labelsize=12)
        
        # 计算每月平均误差并添加标注
        monthly_means = vis_df.groupby('month')['abs_error'].mean()
        for i, mean_val in enumerate(monthly_means):
            plt.text(i, mean_val*1.05, f'{mean_val:.1f}', 
                    ha='center', fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))
        
        plt.tight_layout(pad=2.0)
        plt.savefig(f"{output_dir}/prediction_error_by_month.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{output_dir}/prediction_error_by_month.pdf", format='pdf', bbox_inches='tight')
        plt.close()
    
    # 5. 误差分布直方图
    plt.figure(figsize=(14, 8))
    
    # 使用seaborn的更美观的直方图 - 完全修复KDE兼容性问题
    ax = sns.histplot(vis_df['error'], kde=True, bins=50, 
                color='#3498db', 
                alpha=0.7, edgecolor='white', linewidth=0.5)
    
    # 手动设置KDE线的样式
    if len(ax.lines) > 0:  # 检查是否存在KDE线条
        ax.lines[0].set_color('#e74c3c')  # 设置KDE线的颜色
        ax.lines[0].set_linewidth(2.5)    # 设置KDE线的宽度
    
    # 添加均值和中位数线
    mean_error = vis_df['error'].mean()
    median_error = vis_df['error'].median()
    
    plt.axvline(x=mean_error, color='#27ae60', linestyle='--', linewidth=2, 
               label=f'均值: {mean_error:.2f}')
    plt.axvline(x=median_error, color='#8e44ad', linestyle='--', linewidth=2, 
               label=f'中位数: {median_error:.2f}')
    plt.axvline(x=0, color='#2c3e50', linestyle='-', linewidth=2,
               label='无误差')
    
    # 设置轴标签和标题
    plt.xlabel('预测误差 (ug/m3)', fontsize=14, labelpad=10, color='#333333')
    plt.ylabel('频次', fontsize=14, labelpad=10, color='#333333')
    plt.title('PM2.5预测误差分布', fontsize=18, pad=20, fontweight='bold', color='#333333')
    
    # 添加误差统计信息
    stats_text = (f"均值: {mean_error:.2f} ug/m3\n"
                 f"中位数: {median_error:.2f} ug/m3\n"
                 f"标准差: {vis_df['error'].std():.2f} ug/m3\n"
                 f"最小值: {vis_df['error'].min():.2f} ug/m3\n"
                 f"最大值: {vis_df['error'].max():.2f} ug/m3")
    
    plt.annotate(stats_text, xy=(0.02, 0.95), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8),
                fontsize=12, verticalalignment='top')
    
    # 添加图例
    plt.legend(fontsize=12)
    
    # 添加网格线
    plt.grid(alpha=0.3, linestyle='--')
    plt.tick_params(axis='both', which='major', labelsize=12)
    
    plt.tight_layout(pad=2.0)
    plt.savefig(f"{output_dir}/prediction_error_histogram.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/prediction_error_histogram.pdf", format='pdf', bbox_inches='tight')
    plt.close()
    
    print(f"预测结果可视化图表已保存至 {output_dir} 目录")

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