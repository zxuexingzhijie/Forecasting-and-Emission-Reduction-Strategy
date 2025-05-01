#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据加载和预处理模块
用于PM2.5预测项目中的数据加载、预处理和基础分析
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def load_processed_data(file_path="./processed/processed_data.csv"):
    """
    加载处理过的数据
    
    Args:
        file_path: 数据文件路径
        
    Returns:
        df: 加载的数据框
    """
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误：文件 {file_path} 不存在")
        return None
        
    try:
        # 尝试使用utf-8编码读取
        df = pd.read_csv(file_path)
    except UnicodeDecodeError:
        # 如果失败，尝试使用gb18030编码
        df = pd.read_csv(file_path, encoding='gb18030')
    except Exception as e:
        print(f"加载数据时出错: {e}")
        return None
    
    # 确保datetime列是日期时间类型
    if 'datetime' in df.columns:
        if df['datetime'].dtype != 'datetime64[ns]':
            df['datetime'] = pd.to_datetime(df['datetime'])
    
    return df

def train_test_split_by_time(df, test_size=0.2, date_column='datetime', gap_days=7):
    """
    按时间顺序划分训练集和测试集，保持时间序列的完整性，并添加时间间隔以防止信息泄露
    
    Args:
        df: 输入数据框
        test_size: 测试集比例
        date_column: 日期时间列名
        gap_days: 训练集和测试集之间的间隔天数，用于防止信息泄露
        
    Returns:
        train_df: 训练集
        test_df: 测试集
    """
    # 确保日期列是日期时间类型
    if date_column in df.columns:
        df[date_column] = pd.to_datetime(df[date_column])
    else:
        raise ValueError(f"数据框中不存在指定的日期列: {date_column}")
    
    # 按日期排序
    df = df.sort_values(by=date_column)
    
    # 获取日期范围
    min_date = df[date_column].min()
    max_date = df[date_column].max()
    
    # 计算划分点
    date_range = (max_date - min_date).days
    split_days = int(date_range * (1 - test_size))
    
    # 划分时间点
    split_date = min_date + pd.Timedelta(days=split_days)
    
    # 添加间隔，防止信息泄露
    gap_end_date = split_date + pd.Timedelta(days=gap_days)
    
    print(f"数据时间范围: {min_date.date()} 至 {max_date.date()}")
    print(f"训练集截止日期: {split_date.date()}")
    print(f"为防止信息泄露，设置{gap_days}天的时间间隔")
    print(f"测试集起始日期: {gap_end_date.date()}")
    
    # 划分训练集和测试集
    train_df = df[df[date_column] <= split_date].copy()
    test_df = df[df[date_column] >= gap_end_date].copy()
    
    # 检查划分后的集合大小
    print(f"训练集大小: {len(train_df)} 行, 测试集大小: {len(test_df)} 行")
    print(f"丢弃的间隔期数据: {len(df) - len(train_df) - len(test_df)} 行")
    
    # 根据时间排序
    train_df = train_df.sort_values(by=date_column)
    test_df = test_df.sort_values(by=date_column)
    
    return train_df, test_df

def analyze_static_weather(df, wind_speed_col='WSPM', pm25_col='PM2.5', threshold=1.0):
    """
    分析静稳天气情况
    
    Args:
        df: 数据框
        wind_speed_col: 风速列名
        pm25_col: PM2.5列名
        threshold: 静稳天气风速阈值，默认1.0 m/s
        
    Returns:
        results: 分析结果字典
    """
    # 确保datetime列是日期时间类型
    if 'datetime' in df.columns and df['datetime'].dtype != 'datetime64[ns]':
        df['datetime'] = pd.to_datetime(df['datetime'])
    
    # 标记静稳天气
    df['is_stagnant'] = (df[wind_speed_col] < threshold).astype(int)
    
    # 计算静稳天气比例
    stagnant_ratio = df['is_stagnant'].mean()
    
    # 按静稳天气分组，计算PM2.5均值
    pm25_by_stagnant = df.groupby('is_stagnant')[pm25_col].mean()
    
    # 计算每年静稳天气比例
    if 'datetime' in df.columns:
        yearly_stagnant = df.groupby(df['datetime'].dt.year)['is_stagnant'].mean()
        
        # 计算每月静稳天气比例
        monthly_stagnant = df.groupby([df['datetime'].dt.year.rename('year'), 
                                      df['datetime'].dt.month.rename('month')])['is_stagnant'].mean()
        monthly_stagnant = monthly_stagnant.reset_index()
        monthly_stagnant.columns = ['year', 'month', 'stagnant_ratio']
        
        # 添加季节
        monthly_stagnant['season'] = ((monthly_stagnant['month'] % 12 + 3) // 3).astype(int)
        seasonal_stagnant = monthly_stagnant.groupby(['year', 'season'])['stagnant_ratio'].mean().reset_index()
    else:
        yearly_stagnant = None
        monthly_stagnant = None
        seasonal_stagnant = None
    
    # 整理结果
    results = {
        'stagnant_ratio': stagnant_ratio,
        'pm25_by_stagnant': pm25_by_stagnant,
        'yearly_stagnant': yearly_stagnant,
        'monthly_stagnant': monthly_stagnant,
        'seasonal_stagnant': seasonal_stagnant,
        'df_with_stagnant': df
    }
    
    return results

def visualize_static_weather(results, output_dir="../processed2/figures"):
    """
    可视化静稳天气分析结果
    
    Args:
        results: analyze_static_weather函数返回的结果
        output_dir: 输出目录
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 静稳天气与PM2.5浓度箱线图
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='is_stagnant', y='PM2.5', data=results['df_with_stagnant'])
    plt.xlabel('是否静稳天气 (1=是, 0=否)')
    plt.ylabel('PM2.5浓度 (μg/m³)')
    plt.title('静稳天气与非静稳天气下的PM2.5浓度分布')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/pm25_by_stagnant_boxplot.png", dpi=300)
    plt.close()
    
    # 每年静稳天气比例
    if results['yearly_stagnant'] is not None:
        plt.figure(figsize=(10, 6))
        results['yearly_stagnant'].plot(kind='bar', color='skyblue')
        plt.xlabel('年份')
        plt.ylabel('静稳天气比例')
        plt.title('各年静稳天气比例')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/yearly_stagnant_ratio.png", dpi=300)
        plt.close()
    
    # 季节性静稳天气比例
    if results['seasonal_stagnant'] is not None:
        plt.figure(figsize=(12, 6))
        seasonal_pivot = results['seasonal_stagnant'].pivot(index='year', columns='season', values='stagnant_ratio')
        seasonal_pivot.columns = ['冬季', '春季', '夏季', '秋季']
        seasonal_pivot.plot(kind='bar', figsize=(12, 6))
        plt.xlabel('年份')
        plt.ylabel('静稳天气比例')
        plt.title('各季节静稳天气比例')
        plt.grid(axis='y', alpha=0.3)
        plt.legend(title='季节')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/seasonal_stagnant_ratio.png", dpi=300)
        plt.close()
    
    # 静稳天气持续时间分析
    df = results['df_with_stagnant'].copy()
    
    # 确保datetime列是日期时间类型并排序
    if 'datetime' in df.columns:
        if df['datetime'].dtype != 'datetime64[ns]':
            df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime')
        
        # 计算静稳天气持续时间
        df['stagnant_change'] = df['is_stagnant'].diff().ne(0).cumsum()
        stagnant_periods = df[df['is_stagnant'] == 1].groupby('stagnant_change').size()
        
        # 绘制静稳天气持续时间直方图
        plt.figure(figsize=(12, 6))
        sns.histplot(stagnant_periods, bins=30, kde=True)
        plt.xlabel('持续时间 (小时)')
        plt.ylabel('频次')
        plt.title('静稳天气持续时间分布')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/stagnant_duration_histogram.png", dpi=300)
        plt.close()
        
        # 静稳天气与PM2.5浓度的时间序列图（采样部分数据以避免过于密集）
        sample_size = min(5000, len(df))
        df_sample = df.sample(sample_size) if len(df) > sample_size else df
        df_sample = df_sample.sort_values('datetime')
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
        
        # PM2.5浓度时间序列
        ax1.plot(df_sample['datetime'], df_sample['PM2.5'], 'b-', alpha=0.7)
        ax1.set_ylabel('PM2.5浓度 (μg/m³)')
        ax1.set_title('PM2.5浓度与静稳天气的时间序列')
        ax1.grid(alpha=0.3)
        
        # 静稳天气指示
        ax2.scatter(df_sample['datetime'], df_sample['is_stagnant'], c=df_sample['is_stagnant'], cmap='coolwarm', alpha=0.7)
        ax2.set_ylabel('静稳天气 (1=是, 0=否)')
        ax2.set_xlabel('日期')
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/pm25_stagnant_timeseries.png", dpi=300)
        plt.close()

def compute_statistics(df, pm25_col='PM2.5'):
    """
    计算基本统计信息
    
    Args:
        df: 数据框
        pm25_col: PM2.5列名
        
    Returns:
        stats: 统计信息字典
    """
    # 基本描述性统计
    basic_stats = df[pm25_col].describe()
    
    # 分位数
    quantiles = df[pm25_col].quantile([0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
    
    # 按静稳天气分组统计
    if 'is_stagnant' in df.columns:
        grouped_stats = df.groupby('is_stagnant')[pm25_col].describe()
    else:
        grouped_stats = None
    
    # 整理结果
    stats = {
        'basic_stats': basic_stats,
        'quantiles': quantiles,
        'grouped_stats': grouped_stats
    }
    
    return stats

def save_data_info(df, output_dir="../processed2"):
    """
    保存数据基本信息
    
    Args:
        df: 数据框
        output_dir: 输出目录
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成报告
    with open(f"{output_dir}/data_info.txt", "w", encoding="utf-8") as f:
        f.write("===== 数据集基本信息 =====\n\n")
        
        # 数据集基本信息
        f.write(f"数据形状: {df.shape}\n")
        
        # 时间范围
        if 'datetime' in df.columns:
            f.write(f"时间范围: {df['datetime'].min()} 到 {df['datetime'].max()}\n")
        
        # 列名及数据类型
        f.write("\n列名及数据类型:\n")
        for col, dtype in zip(df.columns, df.dtypes):
            f.write(f"{col}: {dtype}\n")
        
        # 缺失值
        f.write("\n缺失值统计:\n")
        missing = df.isnull().sum()
        missing_percent = missing / len(df) * 100
        for col, miss, perc in zip(missing.index, missing, missing_percent):
            f.write(f"{col}: {miss} ({perc:.2f}%)\n")
    
    # 保存描述性统计信息
    df.describe().to_csv(f"{output_dir}/descriptive_stats.csv")
    
    print(f"数据信息已保存至 {output_dir}/data_info.txt")
    print(f"描述性统计已保存至 {output_dir}/descriptive_stats.csv")

def main():
    """
    主函数
    """
    # 创建输出目录
    os.makedirs("../processed2", exist_ok=True)
    os.makedirs("../processed2/figures", exist_ok=True)
    
    # 加载数据
    print("加载数据...")
    df = load_processed_data()
    
    if df is None:
        print("数据加载失败，退出程序")
        return
    
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
    train_df, test_df = train_test_split_by_time(stagnant_results['df_with_stagnant'])
    
    # 保存训练集和测试集
    train_df.to_csv("../processed2/train_data.csv", index=False)
    test_df.to_csv("../processed2/test_data.csv", index=False)
    print("训练集和测试集已保存")
    
    return train_df, test_df

if __name__ == "__main__":
    main() 