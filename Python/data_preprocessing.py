#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据预处理脚本
处理附件一中的空气质量数据，包括缺失值填充、异常值检测、特征转换等
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
from datetime import datetime

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def load_data(file_path):
    """
    加载数据
    
    Args:
        file_path: 数据文件路径
        
    Returns:
        df: 加载的数据框
    """
    print(f"正在读取数据: {file_path}")
    # 检查文件是否存在
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件 {file_path} 不存在")
    
    # 尝试加载数据
    try:
        # 尝试自动检测编码
        df = pd.read_csv(file_path)
    except UnicodeDecodeError:
        # 如果默认编码失败，尝试其他常见编码
        for encoding in ['utf-8', 'gbk', 'gb2312', 'gb18030']:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                print(f"成功使用 {encoding} 编码读取文件")
                break
            except UnicodeDecodeError:
                continue
    
    print(f"数据加载完成，形状: {df.shape}")
    return df

def check_missing_values(df):
    """
    检查缺失值
    
    Args:
        df: 数据框
        
    Returns:
        missing_info: 缺失值信息
    """
    # 计算每列的缺失值数量和比例
    missing = df.isnull().sum()
    missing_percent = (missing / len(df)) * 100
    missing_info = pd.DataFrame({
        '缺失值数量': missing,
        '缺失值比例(%)': missing_percent
    })
    missing_info = missing_info[missing_info['缺失值数量'] > 0].sort_values('缺失值比例(%)', ascending=False)
    
    return missing_info

def fill_missing_values(df):
    """
    填充缺失值
    
    Args:
        df: 数据框
        
    Returns:
        df: 填充缺失值后的数据框
    """
    # 创建数据副本
    df_filled = df.copy()
    
    # 首先将时间列转换为datetime格式，方便后续处理
    # df_filled['datetime'] = pd.to_datetime(df_filled[['year', 'month', 'day', 'hour']].assign(
    #     hour=lambda x: x['hour'].astype(str) + ':00:00'
    # ).agg(' '.join, axis=1), format='%Y %m %d %H:%M:%S')
    
    # 修复：先将所有列转换为字符串，然后再连接
    time_cols = df_filled[['year', 'month', 'day', 'hour']].astype(str)
    df_filled['datetime'] = pd.to_datetime(
        time_cols['year'] + ' ' + time_cols['month'] + ' ' + 
        time_cols['day'] + ' ' + time_cols['hour'] + ':00:00',
        format='%Y %m %d %H:%M:%S'
    )
    
    # 使用时间索引
    df_filled.set_index('datetime', inplace=True)
    
    # 对于不同类型的变量，采用不同的填充策略
    # 1. 气象数据: 线性插值，但对于大块缺失使用同期历史数据
    weather_cols = ['TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']
    for col in weather_cols:
        # 首先尝试线性插值
        df_filled[col] = df_filled[col].interpolate(method='linear', limit=24)
        
        # 对于仍然存在的缺失值，使用同时段历史均值填充
        if df_filled[col].isnull().sum() > 0:
            # 创建时间特征用于匹配同期数据
            df_filled['month_day_hour'] = df_filled.index.strftime('%m-%d-%H')
            monthly_hourly_mean = df_filled.groupby('month_day_hour')[col].transform('mean')
            df_filled[col] = df_filled[col].fillna(monthly_hourly_mean)
            df_filled.drop('month_day_hour', axis=1, inplace=True)
            
            # 如果还有缺失，使用全局均值填充
            df_filled[col] = df_filled[col].fillna(df_filled[col].mean())
    
    # 2. 污染物数据: 使用相关性高的污染物线性回归预测
    pollution_cols = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
    
    # 计算污染物之间的相关性
    correlation = df_filled[pollution_cols].corr()
    
    for col in pollution_cols:
        if df_filled[col].isnull().sum() > 0:
            # 找出与当前污染物相关性最高的其他污染物
            most_correlated = correlation[col].drop(col).abs().idxmax()
            
            # 使用非缺失值数据训练简单线性回归模型
            mask = ~df_filled[col].isnull() & ~df_filled[most_correlated].isnull()
            if mask.sum() > 0:
                # 简单线性回归填充
                slope, intercept, _, _, _ = stats.linregress(
                    df_filled.loc[mask, most_correlated],
                    df_filled.loc[mask, col]
                )
                # 预测缺失值
                missing_mask = df_filled[col].isnull() & ~df_filled[most_correlated].isnull()
                df_filled.loc[missing_mask, col] = slope * df_filled.loc[missing_mask, most_correlated] + intercept
            
            # 对剩余缺失使用线性插值
            df_filled[col] = df_filled[col].interpolate(method='linear', limit=24)
            
            # 如果还有缺失，使用同时段历史数据填充
            if df_filled[col].isnull().sum() > 0:
                df_filled['month_day_hour'] = df_filled.index.strftime('%m-%d-%H')
                monthly_hourly_mean = df_filled.groupby('month_day_hour')[col].transform('mean')
                df_filled[col] = df_filled[col].fillna(monthly_hourly_mean)
                df_filled.drop('month_day_hour', axis=1, inplace=True)
                
                # 最后使用全局均值填充
                df_filled[col] = df_filled[col].fillna(df_filled[col].mean())
    
    # 3. 风向处理: 风向是循环性的，使用前后值的角度平均
    if 'wd' in df_filled.columns and df_filled['wd'].isnull().sum() > 0:
        # 将风向文本转换为角度值
        wind_dir_map = {
            'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5, 
            'E': 90, 'ESE': 112.5, 'SE': 135, 'SSE': 157.5,
            'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5,
            'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5
        }
        
        # 如果风向是文本格式，转换为角度
        if df_filled['wd'].dtype == 'object':
            df_filled['wd_angle'] = df_filled['wd'].map(wind_dir_map)
        else:
            df_filled['wd_angle'] = df_filled['wd']
        
        # 使用线性插值填充角度
        df_filled['wd_angle'] = df_filled['wd_angle'].interpolate(method='linear')
        
        # 将角度转回最接近的风向类别
        if df_filled['wd'].dtype == 'object':
            # 找到距离最近的风向类别
            def find_closest_direction(angle):
                # 标准化角度到0-360范围
                angle = angle % 360
                # 计算与每个风向的距离
                distances = {dir_name: min(abs(angle - dir_angle), 360 - abs(angle - dir_angle)) 
                            for dir_name, dir_angle in wind_dir_map.items()}
                # 返回距离最小的风向
                return min(distances.items(), key=lambda x: x[1])[0]
            
            # 填充缺失的风向类别
            wind_dir_na_mask = df_filled['wd'].isnull()
            df_filled.loc[wind_dir_na_mask, 'wd'] = df_filled.loc[wind_dir_na_mask, 'wd_angle'].apply(find_closest_direction)
            
            # 删除临时角度列
            df_filled.drop('wd_angle', axis=1, inplace=True)
        else:
            # 如果风向本身就是角度，直接使用插值结果
            df_filled['wd'] = df_filled['wd_angle']
            df_filled.drop('wd_angle', axis=1, inplace=True)
    
    # 重置索引，保留datetime作为列
    df_filled = df_filled.reset_index()
    
    return df_filled

def detect_outliers(df, columns=None, method='zscore', threshold=3):
    """
    检测异常值
    
    Args:
        df: 数据框
        columns: 需要检测异常值的列，默认为所有数值列
        method: 异常值检测方法，支持'zscore', 'iqr'
        threshold: 异常值阈值，对于zscore方法为标准差倍数，对于iqr方法为四分位距倍数
        
    Returns:
        outliers_info: 异常值信息
    """
    # 如果未指定列，使用所有数值列
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    outliers_count = {}
    outliers_percent = {}
    
    for col in columns:
        if method == 'zscore':
            # Z-score法检测异常值
            z_scores = np.abs(stats.zscore(df[col].dropna()))
            outliers = np.where(z_scores > threshold)[0]
            outliers_count[col] = len(outliers)
            outliers_percent[col] = (len(outliers) / len(df[col].dropna())) * 100
            
        elif method == 'iqr':
            # IQR法检测异常值
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index
            outliers_count[col] = len(outliers)
            outliers_percent[col] = (len(outliers) / len(df[col].dropna())) * 100
    
    outliers_info = pd.DataFrame({
        '异常值数量': outliers_count,
        '异常值比例(%)': outliers_percent
    })
    
    return outliers_info

def handle_outliers(df, columns=None, method='zscore', threshold=3, strategy='cap'):
    """
    处理异常值
    
    Args:
        df: 数据框
        columns: 需要处理异常值的列，默认为所有数值列
        method: 异常值检测方法，支持'zscore', 'iqr'
        threshold: 异常值阈值
        strategy: 处理策略，'cap'表示截断，'mean'表示均值替换，'median'表示中位数替换
        
    Returns:
        df_clean: 处理异常值后的数据框
    """
    # 创建数据副本
    df_clean = df.copy()
    
    # 如果未指定列，使用所有数值列
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        if method == 'zscore':
            # Z-score法检测异常值
            data = df_clean[col].dropna()
            z_scores = np.abs(stats.zscore(data))
            outlier_indices = data[z_scores > threshold].index
            
            if strategy == 'cap':
                # 截断异常值
                upper_bound = data.mean() + threshold * data.std()
                lower_bound = data.mean() - threshold * data.std()
                df_clean.loc[outlier_indices, col] = df_clean.loc[outlier_indices, col].clip(lower_bound, upper_bound)
            elif strategy == 'mean':
                # 均值替换
                df_clean.loc[outlier_indices, col] = data.mean()
            elif strategy == 'median':
                # 中位数替换
                df_clean.loc[outlier_indices, col] = data.median()
                
        elif method == 'iqr':
            # IQR法检测异常值
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            outlier_indices = df_clean[(df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)].index
            
            if strategy == 'cap':
                # 截断异常值
                df_clean.loc[outlier_indices, col] = df_clean.loc[outlier_indices, col].clip(lower_bound, upper_bound)
            elif strategy == 'mean':
                # 均值替换
                df_clean.loc[outlier_indices, col] = df_clean[col].mean()
            elif strategy == 'median':
                # 中位数替换
                df_clean.loc[outlier_indices, col] = df_clean[col].median()
    
    return df_clean

def feature_engineering(df):
    """
    特征工程，添加新特征
    
    Args:
        df: 数据框
        
    Returns:
        df_fe: 特征工程后的数据框
    """
    # 创建数据副本
    df_fe = df.copy()
    
    # 确保有datetime列
    if 'datetime' not in df_fe.columns and all(col in df_fe.columns for col in ['year', 'month', 'day', 'hour']):
        # df_fe['datetime'] = pd.to_datetime(df_fe[['year', 'month', 'day', 'hour']].assign(
        #     hour=lambda x: x['hour'].astype(str) + ':00:00'
        # ).agg(' '.join, axis=1), format='%Y %m %d %H:%M:%S')
        
        # 修复：先将所有列转换为字符串，然后再连接
        time_cols = df_fe[['year', 'month', 'day', 'hour']].astype(str)
        df_fe['datetime'] = pd.to_datetime(
            time_cols['year'] + ' ' + time_cols['month'] + ' ' + 
            time_cols['day'] + ' ' + time_cols['hour'] + ':00:00',
            format='%Y %m %d %H:%M:%S'
        )
    
    # 1. 提取时间特征
    df_fe['day_of_week'] = df_fe['datetime'].dt.dayofweek  # 星期几 (0-6)
    df_fe['is_weekend'] = df_fe['day_of_week'].isin([5, 6]).astype(int)  # 是否周末
    df_fe['season'] = (df_fe['month'] % 12 + 3) // 3  # 季节 (1-4)
    
    # 2. 添加供暖期标记 (北京市11月15日至次年3月15日为供暖期)
    df_fe['is_heating_period'] = ((df_fe['month'] >= 11) | (df_fe['month'] <= 3)).astype(int)
    # 细化供暖期定义：11月15日至3月15日
    mask_start = (df_fe['month'] == 11) & (df_fe['day'] < 15)
    mask_end = (df_fe['month'] == 3) & (df_fe['day'] > 15)
    df_fe.loc[mask_start | mask_end, 'is_heating_period'] = 0
    
    # 3. 时间周期性特征（使用正弦和余弦变换）
    # 小时的周期性 (0-23)
    df_fe['hour_sin'] = np.sin(2 * np.pi * df_fe['hour'] / 24)
    df_fe['hour_cos'] = np.cos(2 * np.pi * df_fe['hour'] / 24)
    
    # 月份的周期性 (1-12)
    df_fe['month_sin'] = np.sin(2 * np.pi * df_fe['month'] / 12)
    df_fe['month_cos'] = np.cos(2 * np.pi * df_fe['month'] / 12)
    
    # 星期的周期性 (0-6)
    df_fe['day_of_week_sin'] = np.sin(2 * np.pi * df_fe['day_of_week'] / 7)
    df_fe['day_of_week_cos'] = np.cos(2 * np.pi * df_fe['day_of_week'] / 7)
    
    # 4. 风向处理
    # 如果风向是文本格式，转换为角度
    if 'wd' in df_fe.columns and df_fe['wd'].dtype == 'object':
        wind_dir_map = {
            'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5, 
            'E': 90, 'ESE': 112.5, 'SE': 135, 'SSE': 157.5,
            'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5,
            'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5
        }
        df_fe['wd_angle'] = df_fe['wd'].map(wind_dir_map)
    elif 'wd' in df_fe.columns:
        df_fe['wd_angle'] = df_fe['wd']
    
    # 风向的周期性特征
    if 'wd_angle' in df_fe.columns:
        df_fe['wd_sin'] = np.sin(np.radians(df_fe['wd_angle']))
        df_fe['wd_cos'] = np.cos(np.radians(df_fe['wd_angle']))
    
    # 5. 添加静稳天气标记
    if 'WSPM' in df_fe.columns:
        df_fe['is_stagnant'] = (df_fe['WSPM'] < 1.0).astype(int)
        
        # 计算静稳天气持续时间（向前计算已持续的小时数）
        df_fe['stagnant_duration'] = 0
        stagnant_groups = (df_fe['is_stagnant'].diff() != 0).cumsum()
        for group in df_fe[df_fe['is_stagnant'] == 1].groupby(stagnant_groups).groups:
            indices = df_fe[df_fe['is_stagnant'] == 1].groupby(stagnant_groups).groups[group]
            for i, idx in enumerate(indices):
                df_fe.loc[idx, 'stagnant_duration'] = i + 1
    
    # 6. 污染物比值特征
    if all(col in df_fe.columns for col in ['PM2.5', 'PM10']):
        df_fe['PM_ratio'] = df_fe['PM2.5'] / df_fe['PM10']
        # 处理可能的除零情况
        df_fe['PM_ratio'] = df_fe['PM_ratio'].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # 7. 添加空气质量等级 (基于PM2.5)
    if 'PM2.5' in df_fe.columns:
        # 参考中国环保部标准
        bins = [0, 35, 75, 115, 150, 250, np.inf]
        labels = ['优', '良', '轻度污染', '中度污染', '重度污染', '严重污染']
        df_fe['air_quality_level'] = pd.cut(df_fe['PM2.5'], bins=bins, labels=labels)
    
    # 8. 极端气象条件标记
    if 'TEMP' in df_fe.columns:
        # 极端温度 (根据数据分布定义)
        temp_q1 = df_fe['TEMP'].quantile(0.25)
        temp_q3 = df_fe['TEMP'].quantile(0.75)
        df_fe['is_extreme_cold'] = (df_fe['TEMP'] < temp_q1).astype(int)
        df_fe['is_extreme_hot'] = (df_fe['TEMP'] > temp_q3).astype(int)
    
    if 'RAIN' in df_fe.columns:
        # 降雨标记
        df_fe['is_raining'] = (df_fe['RAIN'] > 0).astype(int)
    
    return df_fe

def save_processed_data(df, output_path):
    """
    保存处理后的数据
    
    Args:
        df: 数据框
        output_path: 输出路径
    """
    # 创建输出目录（如果不存在）
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 保存数据
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"数据已保存至: {output_path}")

def main():
    # 定义输入和输出路径
    input_file = "../附件一.csv"  # 根据实际情况修改路径
    output_file = "../processed/processed_data.csv"
    
    # 1. 加载数据
    try:
        df = load_data(input_file)
    except Exception as e:
        print(f"加载数据失败: {e}")
        return
    
    # 显示数据基本信息
    print("\n数据基本信息:")
    print(df.info())
    print("\n数据前5行:")
    print(df.head())
    
    # 2. 检查缺失值
    print("\n检查缺失值:")
    missing_info = check_missing_values(df)
    print(missing_info)
    
    # 3. 填充缺失值
    print("\n填充缺失值...")
    df_filled = fill_missing_values(df)
    
    # 验证填充结果
    print("\n填充后缺失值情况:")
    missing_after = check_missing_values(df_filled)
    print(missing_after)
    
    # 4. 检测异常值
    print("\n检测异常值:")
    outliers_info = detect_outliers(df_filled)
    print(outliers_info)
    
    # 5. 处理异常值
    print("\n处理异常值...")
    df_clean = handle_outliers(df_filled, strategy='cap')
    
    # 6. 特征工程
    print("\n进行特征工程...")
    df_processed = feature_engineering(df_clean)
    
    # 7. 保存处理后的数据
    save_processed_data(df_processed, output_file)
    
    print("\n数据预处理完成!")
    
    # 8. 数据可视化 (可选)
    # 这里可以添加一些简单的可视化，帮助理解数据

if __name__ == "__main__":
    main() 