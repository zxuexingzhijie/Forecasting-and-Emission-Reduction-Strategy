#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
特征工程模块
为PM2.5预测模型构建各类特征，包括时间特征、滞后特征、气象特征等
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def add_time_features(df):
    """
    添加时间特征
    
    Args:
        df: 包含datetime列的数据框
        
    Returns:
        df: 添加时间特征后的数据框
    """
    df_copy = df.copy()
    
    # 确保datetime列是日期时间类型
    if df_copy['datetime'].dtype != 'datetime64[ns]':
        df_copy['datetime'] = pd.to_datetime(df_copy['datetime'])
    
    # 基本时间特征
    df_copy['hour'] = df_copy['datetime'].dt.hour
    df_copy['day'] = df_copy['datetime'].dt.day
    df_copy['month'] = df_copy['datetime'].dt.month
    df_copy['year'] = df_copy['datetime'].dt.year
    df_copy['day_of_week'] = df_copy['datetime'].dt.dayofweek
    df_copy['is_weekend'] = df_copy['day_of_week'].isin([5, 6]).astype(int)
    
    # 季节特征
    df_copy['season'] = ((df_copy['month'] % 12 + 3) // 3).astype(int)
    
    # 供暖期特征
    heating_mask = ((df_copy['month'] >= 11) | (df_copy['month'] <= 3))
    # 细化供暖期定义：11月15日至3月15日
    early_nov_mask = (df_copy['month'] == 11) & (df_copy['day'] < 15)
    late_mar_mask = (df_copy['month'] == 3) & (df_copy['day'] > 15)
    df_copy['is_heating_period'] = (heating_mask & ~(early_nov_mask | late_mar_mask)).astype(int)
    
    # 周期性时间特征（正弦和余弦变换）
    df_copy['hour_sin'] = np.sin(2 * np.pi * df_copy['hour'] / 24)
    df_copy['hour_cos'] = np.cos(2 * np.pi * df_copy['hour'] / 24)
    
    df_copy['day_sin'] = np.sin(2 * np.pi * df_copy['day'] / 31)  # 31天周期
    df_copy['day_cos'] = np.cos(2 * np.pi * df_copy['day'] / 31)
    
    df_copy['month_sin'] = np.sin(2 * np.pi * df_copy['month'] / 12)
    df_copy['month_cos'] = np.cos(2 * np.pi * df_copy['month'] / 12)
    
    df_copy['day_of_week_sin'] = np.sin(2 * np.pi * df_copy['day_of_week'] / 7)
    df_copy['day_of_week_cos'] = np.cos(2 * np.pi * df_copy['day_of_week'] / 7)
    
    return df_copy

def add_lag_features(df, target_col='PM2.5', lag_hours=[1, 2, 3, 6, 12, 24, 48]):
    """
    添加滞后特征
    
    Args:
        df: 数据框，包含datetime列作为索引或正常列
        target_col: 目标列名
        lag_hours: 滞后小时列表
        
    Returns:
        df: 添加滞后特征后的数据框
    """
    df_copy = df.copy()
    
    # 确保datetime列是日期时间类型并设为索引（临时）
    if 'datetime' in df_copy.columns:
        if df_copy['datetime'].dtype != 'datetime64[ns]':
            df_copy['datetime'] = pd.to_datetime(df_copy['datetime'])
        df_copy = df_copy.set_index('datetime')
    
    # 创建滞后特征
    for lag in lag_hours:
        lag_col_name = f'{target_col}_lag_{lag}h'
        df_copy[lag_col_name] = df_copy[target_col].shift(lag)
    
    # 如果之前有datetime列，恢复索引
    if 'datetime' not in df_copy.columns:
        df_copy = df_copy.reset_index()
    
    return df_copy

def add_rolling_features(df, target_col='PM2.5', windows=[3, 6, 12, 24, 48]):
    """
    添加滚动平均特征
    
    Args:
        df: 数据框，包含datetime列作为索引或正常列
        target_col: 目标列名
        windows: 窗口大小列表（小时）
        
    Returns:
        df: 添加滚动平均特征后的数据框
    """
    df_copy = df.copy()
    
    # 确保datetime列是日期时间类型并设为索引（临时）
    datetime_was_index = False
    if 'datetime' in df_copy.columns:
        if df_copy['datetime'].dtype != 'datetime64[ns]':
            df_copy['datetime'] = pd.to_datetime(df_copy['datetime'])
        df_copy = df_copy.set_index('datetime')
    else:
        datetime_was_index = True
    
    # 创建滚动平均特征
    for window in windows:
        # 滚动平均
        mean_col_name = f'{target_col}_rolling_{window}h_mean'
        df_copy[mean_col_name] = df_copy[target_col].rolling(window=window, min_periods=1).mean()
        
        # 滚动标准差（反映波动性）
        std_col_name = f'{target_col}_rolling_{window}h_std'
        df_copy[std_col_name] = df_copy[target_col].rolling(window=window, min_periods=1).std()
        
        # 滚动最大值和最小值
        max_col_name = f'{target_col}_rolling_{window}h_max'
        df_copy[max_col_name] = df_copy[target_col].rolling(window=window, min_periods=1).max()
        
        min_col_name = f'{target_col}_rolling_{window}h_min'
        df_copy[min_col_name] = df_copy[target_col].rolling(window=window, min_periods=1).min()
    
    # 如果之前有datetime列，恢复索引
    if not datetime_was_index:
        df_copy = df_copy.reset_index()
    
    return df_copy

def add_diff_features(df, target_col='PM2.5', diff_orders=[1, 2]):
    """
    添加差分特征
    
    Args:
        df: 数据框
        target_col: 目标列名
        diff_orders: 差分阶数列表
        
    Returns:
        df: 添加差分特征后的数据框
    """
    df_copy = df.copy()
    
    # 确保datetime列是日期时间类型并设为索引（临时）
    datetime_was_index = False
    if 'datetime' in df_copy.columns:
        if df_copy['datetime'].dtype != 'datetime64[ns]':
            df_copy['datetime'] = pd.to_datetime(df_copy['datetime'])
        df_copy = df_copy.set_index('datetime')
    else:
        datetime_was_index = True
    
    # 创建差分特征
    for order in diff_orders:
        diff_col_name = f'{target_col}_diff_{order}'
        df_copy[diff_col_name] = df_copy[target_col].diff(order)
    
    # 计算变化率（百分比变化）
    df_copy[f'{target_col}_pct_change'] = df_copy[target_col].pct_change() * 100
    
    # 如果之前有datetime列，恢复索引
    if not datetime_was_index:
        df_copy = df_copy.reset_index()
    
    return df_copy

def add_weather_features(df):
    """
    添加气象特征
    
    Args:
        df: 数据框
        
    Returns:
        df: 添加气象特征后的数据框
    """
    df_copy = df.copy()
    
    # 风向处理
    if 'wd_angle' in df_copy.columns:
        # 添加风向的正弦和余弦分量
        df_copy['wd_sin'] = np.sin(np.radians(df_copy['wd_angle']))
        df_copy['wd_cos'] = np.cos(np.radians(df_copy['wd_angle']))
    
    # 静稳天气标记
    if 'WSPM' in df_copy.columns and 'is_stagnant' not in df_copy.columns:
        df_copy['is_stagnant'] = (df_copy['WSPM'] < 1.0).astype(int)
    
    # 计算静稳天气持续时间
    if 'is_stagnant' in df_copy.columns and 'stagnant_duration' not in df_copy.columns:
        df_copy['stagnant_duration'] = 0
        
        # 确保datetime列是日期时间类型并按时间排序
        if 'datetime' in df_copy.columns:
            if df_copy['datetime'].dtype != 'datetime64[ns]':
                df_copy['datetime'] = pd.to_datetime(df_copy['datetime'])
            df_copy = df_copy.sort_values('datetime')
        
        # 创建分组标识，当is_stagnant值变化时分组
        stagnant_groups = (df_copy['is_stagnant'].diff() != 0).cumsum()
        
        # 对每个静稳组计算持续时间
        for group in df_copy[df_copy['is_stagnant'] == 1].groupby(stagnant_groups).groups:
            indices = df_copy[df_copy['is_stagnant'] == 1].groupby(stagnant_groups).groups[group]
            for i, idx in enumerate(indices):
                df_copy.loc[idx, 'stagnant_duration'] = i + 1
    
    # 降水标记
    if 'RAIN' in df_copy.columns:
        df_copy['is_raining'] = (df_copy['RAIN'] > 0).astype(int)
    
    # 极端温度标记
    if 'TEMP' in df_copy.columns:
        temp_q1 = df_copy['TEMP'].quantile(0.25)
        temp_q3 = df_copy['TEMP'].quantile(0.75)
        df_copy['is_extreme_cold'] = (df_copy['TEMP'] < temp_q1).astype(int)
        df_copy['is_extreme_hot'] = (df_copy['TEMP'] > temp_q3).astype(int)
    
    # 气象因子交互特征
    if all(col in df_copy.columns for col in ['TEMP', 'PRES']):
        df_copy['temp_pres_interaction'] = df_copy['TEMP'] * df_copy['PRES'] / 1000  # 标准化以防止数值过大
    
    if all(col in df_copy.columns for col in ['TEMP', 'DEWP']):
        df_copy['temp_dewp_diff'] = df_copy['TEMP'] - df_copy['DEWP']  # 温度露点差，反映相对湿度
    
    if all(col in df_copy.columns for col in ['WSPM', 'TEMP']):
        # 风寒指数近似计算
        df_copy['wind_chill_index'] = 13.12 + 0.6215 * df_copy['TEMP'] - 11.37 * df_copy['WSPM']**0.16 + 0.3965 * df_copy['TEMP'] * df_copy['WSPM']**0.16
    
    return df_copy

def add_pollutant_features(df, pollutants=['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']):
    """
    添加污染物相关特征
    
    Args:
        df: 数据框
        pollutants: 污染物列表
        
    Returns:
        df: 添加污染物相关特征后的数据框
    """
    df_copy = df.copy()
    
    # 污染物比值特征
    if all(pol in df_copy.columns for pol in ['PM2.5', 'PM10']):
        df_copy['PM_ratio'] = df_copy['PM2.5'] / df_copy['PM10']
        # 处理除零和无穷大
        df_copy['PM_ratio'] = df_copy['PM_ratio'].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    if all(pol in df_copy.columns for pol in ['SO2', 'NO2']):
        df_copy['SO2_NO2_ratio'] = df_copy['SO2'] / df_copy['NO2']
        # 处理除零和无穷大
        df_copy['SO2_NO2_ratio'] = df_copy['SO2_NO2_ratio'].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # 污染物综合指数
    valid_pollutants = [p for p in pollutants if p in df_copy.columns]
    if len(valid_pollutants) > 0:
        # 计算污染物浓度Z-score的均值作为综合指数
        z_scores = {}
        for pol in valid_pollutants:
            z_scores[pol] = (df_copy[pol] - df_copy[pol].mean()) / df_copy[pol].std()
        
        df_copy['pollutant_index'] = sum(z_scores.values()) / len(z_scores)
    
    # AQI计算（简化版）
    if 'PM2.5' in df_copy.columns:
        # PM2.5的AQI计算（简化）
        def pm25_to_aqi(pm25):
            if pm25 <= 35:
                return pm25 * 50 / 35
            elif pm25 <= 75:
                return 50 + (pm25 - 35) * 50 / 40
            elif pm25 <= 115:
                return 100 + (pm25 - 75) * 50 / 40
            elif pm25 <= 150:
                return 150 + (pm25 - 115) * 50 / 35
            elif pm25 <= 250:
                return 200 + (pm25 - 150) * 100 / 100
            else:
                return 300 + (pm25 - 250) * 200 / 250
                
        df_copy['PM25_AQI'] = df_copy['PM2.5'].apply(pm25_to_aqi)
    
    return df_copy

def encode_categorical_features(df, cat_cols=None):
    """
    对分类变量进行编码
    
    Args:
        df: 数据框
        cat_cols: 分类变量列表，如果为None则自动检测
        
    Returns:
        df: 编码后的数据框
    """
    df_copy = df.copy()
    
    # 如果没有指定分类变量，自动检测
    if cat_cols is None:
        cat_cols = []
        for col in df_copy.columns:
            if df_copy[col].dtype == 'object' or df_copy[col].nunique() < 10:
                # 排除日期时间列
                if col != 'datetime' and not pd.api.types.is_datetime64_any_dtype(df_copy[col]):
                    cat_cols.append(col)
    
    # 对每个分类变量进行One-Hot编码
    for col in cat_cols:
        # 对于高基数分类变量（如站点），可以直接使用
        if col in df_copy.columns:
            dummies = pd.get_dummies(df_copy[col], prefix=col, drop_first=True)
            df_copy = pd.concat([df_copy, dummies], axis=1)
            df_copy.drop(col, axis=1, inplace=True)
    
    return df_copy

def scale_features(df, exclude_cols=None, scaler_type='standard'):
    """
    特征缩放
    
    Args:
        df: 数据框
        exclude_cols: 排除的列
        scaler_type: 缩放类型，'standard'或'minmax'
        
    Returns:
        df_scaled: 缩放后的数据框
        scaler: 缩放器对象，用于之后的转换
    """
    df_copy = df.copy()
    
    # 默认排除的列
    if exclude_cols is None:
        exclude_cols = ['datetime', 'year', 'month', 'day', 'hour', 
                        'day_of_week', 'is_weekend', 'is_stagnant', 
                        'is_raining', 'is_extreme_cold', 'is_extreme_hot',
                        'season', 'is_heating_period']
    
    # 识别需要缩放的列
    numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
    scale_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    # 选择缩放器
    if scaler_type == 'standard':
        scaler = StandardScaler()
    else:  # minmax
        scaler = MinMaxScaler()
    
    # 应用缩放
    if scale_cols:
        df_copy[scale_cols] = scaler.fit_transform(df_copy[scale_cols])
    
    return df_copy, scaler

def compute_feature_importance(df, target_col):
    """
    计算特征重要性
    
    Args:
        df: 数据框
        target_col: 目标列名
        
    Returns:
        importance_df: 特征重要性数据框
    """
    # 拷贝数据框，防止修改原始数据
    df_copy = df.copy()
    
    # 只保留目标列和数值型特征，移除datetime列
    if 'datetime' in df_copy.columns:
        df_copy = df_copy.drop(['datetime'], axis=1)
    
    # 只选择数值型列
    numeric_cols = df_copy.select_dtypes(include=['int64', 'float64']).columns
    df_copy = df_copy[numeric_cols]
    
    # 确保目标列在数据框中
    if target_col not in df_copy.columns:
        print(f"目标列 {target_col} 不在数据框中，无法计算特征重要性")
        return pd.DataFrame(columns=['feature', 'importance'])
    
    try:
        # 计算与目标变量的相关性
        correlation = df_copy.corr()[target_col].abs().sort_values(ascending=False)
        
        # 排除目标变量自身
        correlation = correlation.drop(target_col)
        
        # 创建特征重要性数据框
        importance_df = pd.DataFrame({
            'feature': correlation.index,
            'importance': correlation.values
        })
        
        return importance_df
        
    except Exception as e:
        print(f"计算特征重要性时出错: {e}")
        # 返回空数据框
        return pd.DataFrame(columns=['feature', 'importance'])

def select_features(df, importance_df, top_n=20, target_col='PM2.5'):
    """
    基于特征重要性选择特征
    
    Args:
        df: 数据框
        importance_df: 特征重要性数据框
        top_n: 选择的特征数量
        target_col: 目标列名
        
    Returns:
        selected_df: 选择特征后的数据框
    """
    # 选择最重要的特征
    top_features = importance_df[importance_df['feature'] != target_col].head(top_n)['feature'].tolist()
    
    # 始终包含目标列
    if target_col not in top_features:
        top_features.append(target_col)
    
    # 始终包含datetime列
    if 'datetime' in df.columns and 'datetime' not in top_features:
        selected_df = df[['datetime'] + top_features]
    else:
        selected_df = df[top_features]
    
    return selected_df

def create_feature_combinations(df, feature_pairs):
    """
    创建特征组合
    
    Args:
        df: 数据框
        feature_pairs: 要组合的特征对列表
        
    Returns:
        df: 添加特征组合后的数据框
    """
    df_copy = df.copy()
    
    for f1, f2 in feature_pairs:
        if f1 in df_copy.columns and f2 in df_copy.columns:
            # 加法组合
            df_copy[f'{f1}_plus_{f2}'] = df_copy[f1] + df_copy[f2]
            
            # 乘法组合
            df_copy[f'{f1}_mult_{f2}'] = df_copy[f1] * df_copy[f2]
            
            # 比值组合（避免除零）
            df_copy[f'{f1}_div_{f2}'] = df_copy[f1] / (df_copy[f2] + 1e-10)
            
            # 差值组合
            df_copy[f'{f1}_minus_{f2}'] = df_copy[f1] - df_copy[f2]
    
    return df_copy

def prepare_model_data(df, target_col='PM2.5', 
                    include_lagged=True,
                    include_rolling=True,
                    include_weather=True,
                    include_pollutants=True,
                    add_combinations=True,
                    scale=True):
    """
    准备用于模型训练的数据，包括特征工程和数据标准化
    
    Args:
        df: 输入数据框
        target_col: 目标变量列名
        include_lagged: 是否包含滞后特征
        include_rolling: 是否包含滚动特征
        include_weather: 是否包含天气特征
        include_pollutants: 是否包含其他污染物特征
        add_combinations: 是否添加组合特征
        scale: 是否进行特征标准化
        
    Returns:
        processed_df: 处理后的数据框
        scaler: 标准化器对象
        importance_df: 特征重要性数据框
    """
    # 复制数据框，避免修改原始数据
    data = df.copy()
    
    # 确保datetime列是日期时间类型
    if 'datetime' in data.columns:
        data['datetime'] = pd.to_datetime(data['datetime'])
    
    # 所有可用于特征工程的列
    all_cols = data.columns.tolist()
    
    # 排除不用于特征工程的列
    exclude_cols = ['datetime', target_col]
    feature_cols = [col for col in all_cols if col not in exclude_cols]
    
    # 初始化输出数据框
    processed_df = data[exclude_cols].copy()
    
    # 时间特征提取
    if 'datetime' in data.columns:
        print("提取时间特征...")
        # 只保留对模型有价值的时间特征，减少特征数量
        processed_df['month'] = data['datetime'].dt.month
        processed_df['day'] = data['datetime'].dt.day
        processed_df['dayofweek'] = data['datetime'].dt.dayofweek
        processed_df['is_weekend'] = (processed_df['dayofweek'] >= 5).astype(int)
        
        # 季节性特征 - 使用正弦和余弦变换以保持循环特性
        processed_df['month_sin'] = np.sin(2 * np.pi * data['datetime'].dt.month / 12)
        processed_df['month_cos'] = np.cos(2 * np.pi * data['datetime'].dt.month / 12)
        processed_df['hour_sin'] = np.sin(2 * np.pi * data['datetime'].dt.hour / 24)
        processed_df['hour_cos'] = np.cos(2 * np.pi * data['datetime'].dt.hour / 24)
    
    # 天气特征
    weather_cols = ['DEWP', 'HUMI', 'PRES', 'TEMP', 'WSPM', 'is_stagnant']
    if include_weather:
        weather_cols_exist = [col for col in weather_cols if col in data.columns]
        if weather_cols_exist:
            print("添加天气特征...")
            processed_df[weather_cols_exist] = data[weather_cols_exist]
    
    # 污染物特征
    pollutant_cols = ['SO2', 'NO2', 'CO', 'O3']
    if include_pollutants:
        pollutant_cols_exist = [col for col in pollutant_cols if col in data.columns]
        if pollutant_cols_exist:
            print("添加污染物特征...")
            processed_df[pollutant_cols_exist] = data[pollutant_cols_exist]
    
    # 滞后特征 - 使用较短的滞后窗口，避免引入过多相关特征
    if include_lagged:
        print("创建滞后特征...")
        # 对目标变量创建滞后特征，但减少滞后步数，避免过拟合
        for lag in [1, 3, 6]:  # 减少了滞后步数
            processed_df[f'{target_col}_lag_{lag}'] = data[target_col].shift(lag)
        
        # 对天气和污染物特征创建有限的滞后特征
        key_feature_cols = weather_cols_exist + pollutant_cols_exist if 'weather_cols_exist' in locals() and 'pollutant_cols_exist' in locals() else []
        # 只对重要特征创建滞后特征，并限制滞后步数
        for col in key_feature_cols[:3]:  # 限制特征数量
            if col in data.columns and col != target_col:
                processed_df[f'{col}_lag_1'] = data[col].shift(1)
    
    # 滚动特征
    if include_rolling:
        print("创建滚动特征...")
        # 只为PM2.5创建有限的滚动特征
        for window in [3, 6]:  # 减少了窗口大小
            # 计算滚动平均
            processed_df[f'{target_col}_rolling_mean_{window}'] = data[target_col].shift(1).rolling(window=window).mean()
            # 计算滚动标准差
            processed_df[f'{target_col}_rolling_std_{window}'] = data[target_col].shift(1).rolling(window=window).std()
    
    # 添加组合特征 - 减少组合特征的数量，只组合最关键的特征
    if add_combinations and 'weather_cols_exist' in locals() and len(weather_cols_exist) > 0:
        print("创建组合特征...")
        # 只创建物理意义明确的组合特征
        # 温度与湿度组合 (感知温度)
        if 'TEMP' in processed_df.columns and 'HUMI' in processed_df.columns:
            processed_df['temp_humidity'] = processed_df['TEMP'] * processed_df['HUMI']
        
        # 风速与气压组合 (可能影响污染物扩散)
        if 'WSPM' in processed_df.columns and 'PRES' in processed_df.columns:
            processed_df['wind_pressure'] = processed_df['WSPM'] / processed_df['PRES']
    
    # 删除包含NaN的行
    initial_rows = processed_df.shape[0]
    processed_df = processed_df.dropna()
    dropped_rows = initial_rows - processed_df.shape[0]
    print(f"删除了{dropped_rows}行含有NaN值的数据，剩余{processed_df.shape[0]}行")
    
    # 标准化数值特征
    if scale:
        print("标准化特征...")
        # 识别数值特征
        numeric_cols = processed_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col != target_col]
        
        if numeric_cols:
            # 使用更稳健的标准化方法
            scaler = RobustScaler()  # 使用RobustScaler替代StandardScaler，对异常值更稳健
            processed_df[numeric_cols] = scaler.fit_transform(processed_df[numeric_cols])
        else:
            scaler = None
    else:
        scaler = None
    
    # 计算特征重要性（使用相关性）
    print("计算特征重要性...")
    importance_df = compute_feature_importance(processed_df, target_col)
    
    # 根据特征重要性筛选特征，只保留前60%最重要的特征
    if importance_df is not None and len(importance_df) > 10:  # 确保有足够的特征可选
        print("根据重要性筛选特征...")
        top_features = importance_df.sort_values('importance', ascending=False)
        n_features = max(int(len(top_features) * 0.6), 10)  # 取前60%的特征，但至少保留10个
        top_features = top_features.head(n_features)
        
        # 保留目标列和必要的非特征列
        keep_cols = [col for col in processed_df.columns if col == target_col or col == 'datetime' or col == 'is_stagnant']
        # 添加筛选后的重要特征
        keep_cols += [col for col in top_features['feature'].tolist() if col in processed_df.columns and col not in keep_cols]
        
        # 筛选数据框
        processed_df = processed_df[keep_cols]
        print(f"特征筛选后，保留了{len(keep_cols)}个特征")
    
    return processed_df, scaler, importance_df

def visualize_feature_importance(importance_df, top_n=20, output_dir="./processed2/figures"):
    """
    可视化特征重要性
    
    Args:
        importance_df: 特征重要性数据框
        top_n: 显示的特征数量
        output_dir: 输出目录
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 检查数据框是否为空
    if importance_df.empty:
        print("警告: 特征重要性数据框为空，无法生成可视化")
        return
    
    # 取绝对值并限制显示的特征数量
    importance_df_abs = importance_df.copy()
    importance_df_abs['importance'] = importance_df_abs['importance'].abs()
    
    # 确保top_n不超过可用特征数量
    top_n = min(top_n, len(importance_df_abs))
    
    # 绘制特征重要性条形图
    plt.figure(figsize=(12, 8))
    top_features = importance_df_abs.head(top_n).sort_values('importance')
    plt.barh(top_features['feature'], top_features['importance'], color='skyblue')
    plt.title(f'特征重要性 (Top {top_n})')
    plt.xlabel('相关系数绝对值')
    plt.ylabel('特征名称')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/feature_importance.png", dpi=300)
    plt.close()
    
    # 保存特征重要性到CSV
    importance_df.to_csv(f"{output_dir}/feature_importance.csv", index=False)
    print(f"特征重要性已保存至 {output_dir}/feature_importance.csv")

def save_features_info(processed_df, importance_df, output_dir="./processed2"):
    """
    保存特征信息
    
    Args:
        processed_df: 处理后的数据框
        importance_df: 特征重要性数据框
        output_dir: 输出目录
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存特征信息
    features_info = pd.DataFrame({
        'feature': processed_df.columns,
        'type': processed_df.dtypes,
        'non_null_count': processed_df.count(),
        'null_count': processed_df.isnull().sum(),
        'null_percent': (processed_df.isnull().sum() / len(processed_df) * 100).round(2)
    })
    
    # 如果特征重要性数据框不为空，合并特征重要性信息
    if not importance_df.empty and 'feature' in importance_df.columns and 'importance' in importance_df.columns:
        # 合并特征重要性信息
        features_info = features_info.merge(
            importance_df[['feature', 'importance']], 
            on='feature', 
            how='left'
        )
        
        # 如果特征重要性值可以排序，则排序
        if 'importance' in features_info.columns:
            try:
                features_info = features_info.sort_values('importance', ascending=False)
            except:
                # 如果排序失败，保持原顺序
                pass
    
    # 保存到CSV
    features_info.to_csv(f"{output_dir}/features_info.csv", index=False)
    print(f"特征信息已保存至 {output_dir}/features_info.csv")

def main():
    # 加载训练数据
    train_df = pd.read_csv("./processed2/train_data.csv")
    
    # 加载测试数据
    test_df = pd.read_csv("./processed2/test_data.csv")
    
    # 确保datetime列是日期时间类型
    for df in [train_df, test_df]:
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
    
    # 准备训练数据
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
    
    # 准备测试数据（使用训练集的缩放器）
    processed_test, _, _ = prepare_model_data(
        test_df, 
        target_col='PM2.5',
        include_lagged=True,
        include_rolling=True,
        include_weather=True,
        include_pollutants=True,
        add_combinations=True,
        scale=True  # 使用与训练数据相同的缩放
    )
    
    # 可视化特征重要性
    visualize_feature_importance(importance_df.abs().sort_values('importance', ascending=False))
    
    # 保存特征信息
    save_features_info(processed_train, importance_df)
    
    # 保存处理后的数据
    processed_train.to_csv("./processed2/processed_train.csv", index=False)
    processed_test.to_csv("./processed2/processed_test.csv", index=False)
    print("处理后的训练和测试数据已保存")

if __name__ == "__main__":
    main() 