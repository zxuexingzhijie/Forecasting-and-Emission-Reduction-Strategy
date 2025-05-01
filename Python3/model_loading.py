#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
问题三：PM2.5减排策略优化 - 模型加载模块
用于加载问题二训练好的模型，并进行PM2.5预测
"""

import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
from datetime import datetime

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 获取当前脚本所在目录的父目录（项目根目录）
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def load_best_model(model_dir=None):
    """
    加载问题二中训练好的最佳模型
    
    Args:
        model_dir: 模型保存目录
        
    Returns:
        model: 加载的模型
        feature_columns: 模型使用的特征列名
    """
    # 如果没有指定model_dir，使用项目根目录下的processed2/models
    if model_dir is None:
        model_dir = os.path.join(ROOT_DIR, "processed2", "models")
    
    print(f"尝试从目录加载模型：{model_dir}")
    
    # 首先尝试加载调优后的LGB模型，这是问题二中表现最好的模型
    try:
        model_path = os.path.join(model_dir, "LGB_tuned_model.pkl")
        print(f"尝试加载模型：{model_path}")
        model = joblib.load(model_path)
        print("成功加载调优后的LGB模型")
    except FileNotFoundError:
        # 如果找不到调优后的模型，尝试加载基础LGB模型
        try:
            model_path = os.path.join(model_dir, "LGB_model.pkl")
            print(f"尝试加载模型：{model_path}")
            model = joblib.load(model_path)
            print("成功加载基础LGB模型")
        except FileNotFoundError:
            # 如果LGB模型都找不到，尝试加载XGBoost模型
            try:
                model_path = os.path.join(model_dir, "XGB_model.pkl")
                print(f"尝试加载模型：{model_path}")
                model = joblib.load(model_path)
                print("成功加载XGBoost模型")
            except FileNotFoundError:
                # 最后尝试列出目录中的所有文件
                try:
                    files = os.listdir(model_dir)
                    print(f"目录中的文件列表：{files}")
                    # 尝试加载任何可用的模型
                    for filename in files:
                        if filename.endswith('.pkl') and ('model' in filename):
                            model_path = os.path.join(model_dir, filename)
                            print(f"尝试加载模型：{model_path}")
                            model = joblib.load(model_path)
                            print(f"成功加载模型：{filename}")
                            break
                    else:
                        raise Exception("未找到任何可用的模型文件")
                except Exception as e:
                    print(f"列出目录内容时出错：{e}")
                    raise Exception(f"无法找到可用的预测模型，请确保问题二的模型训练正确完成。错误：{e}")
    
    # 加载特征列名
    try:
        feature_columns_path = os.path.join(model_dir, "LGB_feature_columns.txt")
        print(f"尝试加载特征列名：{feature_columns_path}")
        with open(feature_columns_path, "r") as f:
            feature_columns = [line.strip() for line in f.readlines()]
        print(f"成功加载特征列名，共{len(feature_columns)}个特征")
    except FileNotFoundError:
        feature_columns = None
        print("警告：无法加载特征列名文件，将使用所有输入特征")
    
    return model, feature_columns

def load_processed_data(data_path=None, data_dir=None):
    """
    加载已处理的数据
    
    Args:
        data_path: 数据文件的完整路径
        data_dir: 数据保存目录（如果未提供data_path则使用）
        
    Returns:
        processed_data: 处理过的数据
    """
    # 如果提供了完整路径，直接使用
    if data_path is not None:
        print(f"尝试加载数据：{data_path}")
        try:
            processed_data = pd.read_csv(data_path)
            # 确保datetime列是日期时间类型
            if 'datetime' in processed_data.columns:
                processed_data['datetime'] = pd.to_datetime(processed_data['datetime'])
            print(f"成功加载数据，共{len(processed_data)}条记录")
            return processed_data
        except FileNotFoundError:
            print(f"找不到数据文件：{data_path}")
            raise
    
    # 如果没有指定data_dir，使用项目根目录下的processed2
    if data_dir is None:
        data_dir = os.path.join(ROOT_DIR, "processed2")
    
    print(f"尝试从目录加载数据：{data_dir}")
    
    # 加载处理后的测试数据
    try:
        data_path = os.path.join(data_dir, "processed_test.csv")
        print(f"尝试加载数据：{data_path}")
        processed_data = pd.read_csv(data_path)
        # 确保datetime列是日期时间类型
        if 'datetime' in processed_data.columns:
            processed_data['datetime'] = pd.to_datetime(processed_data['datetime'])
        print(f"成功加载处理后的测试数据，共{len(processed_data)}条记录")
        return processed_data
    except FileNotFoundError:
        # 尝试列出目录中的所有文件
        try:
            files = os.listdir(data_dir)
            print(f"目录中的文件列表：{files}")
        except Exception as e:
            print(f"列出目录内容时出错：{e}")
        raise Exception(f"无法找到处理后的测试数据，请确保问题二的特征工程正确完成。尝试的路径：{data_path}")

def load_data_for_target_date(target_date="2016-03-01"):
    """
    尝试从测试集和训练集中加载指定日期的数据
    
    Args:
        target_date: 目标日期，格式为"YYYY-MM-DD"
        
    Returns:
        target_data: 目标日期的数据
        source_name: 数据来源描述（'测试集'或'训练集'）
    """
    # 转换目标日期为datetime对象
    target_date_dt = pd.to_datetime(target_date).date()
    
    # 首先尝试从测试集中加载数据
    try:
        test_data_path = os.path.join(ROOT_DIR, "processed2", "processed_test.csv")
        print(f"尝试从测试集加载{target_date}的数据...")
        test_data = load_processed_data(test_data_path)
        test_target_data = test_data[test_data['datetime'].dt.date == target_date_dt].copy()
        
        if len(test_target_data) > 0:
            print(f"在测试集中找到了{target_date}的数据，共{len(test_target_data)}条记录")
            return test_target_data, "测试集"
        else:
            print(f"测试集中没有{target_date}的数据")
    except Exception as e:
        print(f"加载测试集数据出错：{e}")
    
    # 如果测试集中没有找到数据，尝试从训练集中加载
    try:
        train_data_path = os.path.join(ROOT_DIR, "processed2", "processed_train.csv")
        print(f"尝试从训练集加载{target_date}的数据...")
        train_data = load_processed_data(train_data_path)
        train_target_data = train_data[train_data['datetime'].dt.date == target_date_dt].copy()
        
        if len(train_target_data) > 0:
            print(f"在训练集中找到了{target_date}的数据，共{len(train_target_data)}条记录")
            return train_target_data, "训练集"
        else:
            print(f"训练集中也没有{target_date}的数据")
    except Exception as e:
        print(f"加载训练集数据出错：{e}")
    
    # 如果两个数据集中都没有找到目标日期的数据，就使用最接近的数据
    print(f"在测试集和训练集中都没有找到{target_date}的数据，将尝试使用最接近的日期...")
    
    # 先尝试测试集
    try:
        test_data = load_processed_data(os.path.join(ROOT_DIR, "processed2", "processed_test.csv"))
        if len(test_data) > 0:
            # 找出最接近的日期
            test_data['date'] = test_data['datetime'].dt.date
            unique_dates = sorted(test_data['date'].unique())
            print(f"测试集中可用的日期范围：{min(unique_dates)} 至 {max(unique_dates)}")
            
            # 选择最接近的日期
            closest_date = min(unique_dates, key=lambda x: abs((pd.to_datetime(x) - pd.to_datetime(target_date_dt)).days))
            closest_data = test_data[test_data['date'] == closest_date].copy()
            
            print(f"在测试集中选择了最接近的日期：{closest_date}，共{len(closest_data)}条记录")
            return closest_data, f"测试集（最接近日期：{closest_date}）"
    except Exception as e:
        print(f"查找测试集中最接近日期的数据出错：{e}")
    
    # 最后尝试训练集
    try:
        train_data = load_processed_data(os.path.join(ROOT_DIR, "processed2", "processed_train.csv"))
        if len(train_data) > 0:
            # 找出最接近的日期
            train_data['date'] = train_data['datetime'].dt.date
            unique_dates = sorted(train_data['date'].unique())
            print(f"训练集中可用的日期范围：{min(unique_dates)} 至 {max(unique_dates)}")
            
            # 选择最接近的日期
            closest_date = min(unique_dates, key=lambda x: abs((pd.to_datetime(x) - pd.to_datetime(target_date_dt)).days))
            closest_data = train_data[train_data['date'] == closest_date].copy()
            
            print(f"在训练集中选择了最接近的日期：{closest_date}，共{len(closest_data)}条记录")
            return closest_data, f"训练集（最接近日期：{closest_date}）"
    except Exception as e:
        print(f"查找训练集中最接近日期的数据出错：{e}")
    
    # 如果所有尝试都失败，返回空数据和错误信息
    return pd.DataFrame(), "未找到数据"

def predict_pm25_for_target_date(model, processed_data=None, feature_columns=None, target_date="2016-03-01"):
    """
    预测指定日期的PM2.5日均浓度
    
    Args:
        model: 加载的预测模型
        processed_data: 处理后的数据，如果为None则会调用load_data_for_target_date加载
        feature_columns: 模型使用的特征列名
        target_date: 目标日期，格式为"YYYY-MM-DD"
        
    Returns:
        daily_avg_pm25: 目标日期的PM2.5日均浓度预测值
        actual_date: 实际使用的日期
        data_source: 数据来源描述
    """
    # 如果没有提供数据，则尝试从测试集或训练集加载指定日期的数据
    if processed_data is None:
        target_data, data_source = load_data_for_target_date(target_date)
        if len(target_data) == 0:
            raise Exception(f"无法找到{target_date}或接近日期的数据")
    else:
        # 转换目标日期为datetime对象
        target_date_dt = pd.to_datetime(target_date).date()
        
        # 筛选目标日期的数据
        target_data = processed_data[processed_data['datetime'].dt.date == target_date_dt].copy()
        
        if len(target_data) == 0:
            print(f"警告：在提供的数据中找不到目标日期{target_date}的记录")
            print("将使用数据中的第一天进行预测...")
            
            # 获取数据中的第一天日期
            first_date = processed_data['datetime'].dt.date.min()
            target_date_dt = first_date
            target_data = processed_data[processed_data['datetime'].dt.date == target_date_dt].copy()
            
            print(f"使用替代日期: {target_date_dt}")
            data_source = "提供的数据集（使用替代日期）"
        else:
            data_source = "提供的数据集"
    
    # 实际使用的日期字符串
    actual_date = target_data['datetime'].dt.date.iloc[0].strftime("%Y-%m-%d")
    
    # 准备特征
    if 'datetime' in target_data.columns:
        X = target_data.drop(['datetime', 'PM2.5'], axis=1)
    else:
        X = target_data.drop(['PM2.5'], axis=1)
    
    # 如果提供了特征列表，只使用这些特征
    if feature_columns is not None:
        # 处理特征列不匹配的问题
        missing_cols = set(feature_columns) - set(X.columns)
        if missing_cols:
            for col in missing_cols:
                X[col] = 0  # 对缺失的特征填充0
        
        # 只保留需要的特征，并按照模型训练时的顺序排列
        X = X[feature_columns]
    
    # 使用模型预测
    predictions = model.predict(X)
    
    # 计算日均浓度
    daily_avg_pm25 = np.mean(predictions)
    
    # 计算真实日均浓度以便比较
    true_daily_avg_pm25 = target_data['PM2.5'].mean()
    
    print(f"目标日期 {actual_date} 的PM2.5预测结果 (数据来源: {data_source}):")
    print(f"小时预测值均值: {daily_avg_pm25:.2f} μg/m³")
    print(f"实际日均值: {true_daily_avg_pm25:.2f} μg/m³")
    print(f"预测误差: {daily_avg_pm25 - true_daily_avg_pm25:.2f} μg/m³")
    
    # 创建小时预测与实际值的对比图
    plt.figure(figsize=(12, 6))
    plt.plot(target_data['datetime'], target_data['PM2.5'], 'b-', label='实际值')
    plt.plot(target_data['datetime'], predictions, 'r-', label='预测值')
    plt.axhline(y=daily_avg_pm25, color='r', linestyle='--', label=f'预测日均值 ({daily_avg_pm25:.2f})')
    plt.axhline(y=true_daily_avg_pm25, color='b', linestyle='--', label=f'实际日均值 ({true_daily_avg_pm25:.2f})')
    plt.title(f'PM2.5 小时浓度预测 (日期: {actual_date}, 数据来源: {data_source})')
    plt.xlabel('时间')
    plt.ylabel('PM2.5浓度 (μg/m³)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    # 确保输出目录存在
    figures_dir = os.path.join(ROOT_DIR, "processed3", "figures")
    os.makedirs(figures_dir, exist_ok=True)
    plt.savefig(os.path.join(figures_dir, "target_date_pm25_prediction.png"), dpi=300)
    plt.close()
    
    # 返回日均浓度预测值、实际使用的日期和数据来源
    return daily_avg_pm25, actual_date, data_source

def main():
    """
    主函数
    """
    print("加载模型和数据...")
    model, feature_columns = load_best_model()
    processed_data = load_processed_data()
    
    print("\n预测2016年3月1日PM2.5日均浓度...")
    daily_avg_pm25, actual_date, data_source = predict_pm25_for_target_date(model, processed_data, feature_columns)
    
    # 保存结果
    output_dir = os.path.join(ROOT_DIR, "processed3")
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "predicted_pm25.txt"), "w", encoding="utf-8") as f:
        if actual_date != "2016-03-01":
            f.write(f"注意：数据中没有2016年3月1日的记录，使用了替代日期{actual_date}\n\n")
        f.write(f"{actual_date} PM2.5日均浓度预测值：{daily_avg_pm25:.2f} μg/m³\n")
        f.write(f"减排目标值（80%）：{daily_avg_pm25 * 0.8:.2f} μg/m³\n")
        f.write(f"需要减少的浓度：{daily_avg_pm25 * 0.2:.2f} μg/m³\n")
    
    print(f"\n计算结果已保存到 {os.path.join(output_dir, 'predicted_pm25.txt')}")
    
    return daily_avg_pm25, actual_date, data_source

if __name__ == "__main__":
    main() 