#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据可视化脚本
用于分析附件一中空气质量数据的时空特征
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
import os
from datetime import datetime
import matplotlib.cm as cm

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.rcParams['figure.figsize'] = [14, 8]  # 设置默认图片大小

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
        df = pd.read_csv(file_path)
        print(f"成功使用默认编码读取文件")
    except UnicodeDecodeError:
        # 如果默认编码失败，尝试其他常见编码
        for encoding in ['utf-8', 'gbk', 'gb2312', 'gb18030']:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                print(f"成功使用 {encoding} 编码读取文件")
                break
            except UnicodeDecodeError:
                continue
        else:
            raise UnicodeDecodeError("无法使用任何常见编码读取文件")
    
    # 确保有datetime列，用于时间序列分析
    if 'datetime' in df.columns:
        # 确保datetime列是日期时间类型
        if df['datetime'].dtype == 'object':
            try:
                df['datetime'] = pd.to_datetime(df['datetime'])
                print("已将datetime列转换为日期时间类型")
            except Exception as e:
                print(f"转换datetime列时出错: {e}")
    elif all(col in df.columns for col in ['year', 'month', 'day', 'hour']):
        try:
            # 转换为字符串再连接
            time_cols = df[['year', 'month', 'day', 'hour']].astype(str)
            df['datetime'] = pd.to_datetime(
                time_cols['year'] + '-' + time_cols['month'] + '-' + 
                time_cols['day'] + ' ' + time_cols['hour'] + ':00:00'
            )
            print("已从年月日时列创建datetime列")
        except Exception as e:
            print(f"创建datetime列时出错: {e}")
    
    print(f"数据加载完成，形状: {df.shape}")
    return df

def create_output_dir(output_dir='./figures'):
    """
    创建输出目录
    
    Args:
        output_dir: 输出目录路径
    """
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            print(f"已创建输出目录: {output_dir}")
        else:
            print(f"输出目录已存在: {output_dir}")
    except Exception as e:
        print(f"创建输出目录时出错: {e}")
        # 如果无法创建指定目录，使用当前目录的figures子目录
        fallback_dir = "./figures"
        if not os.path.exists(fallback_dir):
            os.makedirs(fallback_dir, exist_ok=True)
        print(f"使用替代输出目录: {fallback_dir}")
        return fallback_dir
    
    return output_dir

def plot_time_series(df, columns, title, output_dir, filename, freq='D'):
    """
    绘制时间序列图
    
    Args:
        df: 数据框
        columns: 要绘制的列
        title: 图表标题
        output_dir: 输出目录
        filename: 输出文件名
        freq: 重采样频率，'D'表示日均值，'ME'表示月末
    """
    plt.figure(figsize=(16, 8))
    
    try:
        # 设置时间索引
        df_temp = df.copy()
        
        # 确保datetime列是日期时间类型
        if 'datetime' in df_temp.columns:
            if df_temp['datetime'].dtype == 'object':
                df_temp['datetime'] = pd.to_datetime(df_temp['datetime'])
            df_temp.set_index('datetime', inplace=True)
        else:
            # 如果没有datetime列但有日期相关列，尝试创建
            if all(col in df_temp.columns for col in ['year', 'month', 'day', 'hour']):
                time_cols = df_temp[['year', 'month', 'day', 'hour']].astype(str)
                df_temp['datetime'] = pd.to_datetime(
                    time_cols['year'] + '-' + time_cols['month'] + '-' + 
                    time_cols['day'] + ' ' + time_cols['hour'] + ':00:00'
                )
                df_temp.set_index('datetime', inplace=True)
            else:
                raise ValueError("数据中缺少datetime列或必要的日期时间列")
        
        # 修复弃用警告：将'M'替换为'ME'
        if freq == 'M':
            freq = 'ME'  # 使用月末频率替代
            print("注意: 使用'ME'(月末)替代已弃用的'M'")
        
        # 重采样以减少噪声
        df_resampled = df_temp[columns].resample(freq).mean()
        
        # 绘制时间序列
        for col in columns:
            plt.plot(df_resampled.index, df_resampled[col], label=col)
        
        plt.title(title, fontsize=16)
        plt.xlabel('日期', fontsize=12)
        plt.ylabel('浓度/数值', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # 设置x轴日期格式
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=45)
        
        # 保存图表
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()
    
    except Exception as e:
        print(f"绘制时间序列图时出错 ({filename}): {e}")
        plt.close() # 确保关闭图表

def plot_correlation_heatmap(df, columns, title, output_dir, filename):
    """
    绘制相关性热力图
    
    Args:
        df: 数据框
        columns: 要分析的列
        title: 图表标题
        output_dir: 输出目录
        filename: 输出文件名
    """
    plt.figure(figsize=(12, 10))
    
    # 计算相关系数
    corr = df[columns].corr()
    
    # 绘制热力图
    mask = np.triu(np.ones_like(corr, dtype=bool))  # 对角线上方的掩码
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True, fmt=".2f")
    
    plt.title(title, fontsize=16)
    
    # 保存图表
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

def plot_boxplot_by_category(df, value_col, category_col, title, output_dir, filename):
    """
    按类别绘制箱线图
    
    Args:
        df: 数据框
        value_col: 数值列
        category_col: 类别列
        title: 图表标题
        output_dir: 输出目录
        filename: 输出文件名
    """
    plt.figure(figsize=(14, 8))
    
    sns.boxplot(x=category_col, y=value_col, data=df)
    
    plt.title(title, fontsize=16)
    plt.xlabel(category_col, fontsize=12)
    plt.ylabel(value_col, fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 如果类别太多，旋转标签
    if df[category_col].nunique() > 6:
        plt.xticks(rotation=45)
    
    # 保存图表
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

def plot_daily_pattern(df, columns, title, output_dir, filename):
    """
    绘制日内变化模式
    
    Args:
        df: 数据框
        columns: 要分析的列
        title: 图表标题
        output_dir: 输出目录
        filename: 输出文件名
    """
    plt.figure(figsize=(14, 8))
    
    # 计算每小时的平均值
    hourly_avg = df.groupby('hour')[columns].mean()
    
    for col in columns:
        plt.plot(hourly_avg.index, hourly_avg[col], label=col, marker='o')
    
    plt.title(title, fontsize=16)
    plt.xlabel('小时', fontsize=12)
    plt.ylabel('平均浓度/数值', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 设置x轴刻度
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xticks(range(0, 24))
    
    # 保存图表
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

def plot_seasonal_pattern(df, columns, title, output_dir, filename):
    """
    绘制季节性变化模式
    
    Args:
        df: 数据框
        columns: 要分析的列
        title: 图表标题
        output_dir: 输出目录
        filename: 输出文件名
    """
    plt.figure(figsize=(14, 8))
    
    # 计算每月的平均值
    monthly_avg = df.groupby('month')[columns].mean()
    
    for col in columns:
        plt.plot(monthly_avg.index, monthly_avg[col], label=col, marker='o')
    
    plt.title(title, fontsize=16)
    plt.xlabel('月份', fontsize=12)
    plt.ylabel('平均浓度/数值', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 设置x轴刻度
    plt.xticks(range(1, 13))
    
    # 保存图表
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

def plot_wind_rose(df, output_dir, filename):
    """
    绘制风玫瑰图
    
    Args:
        df: 包含风向角度(wd_angle)和风速(WSPM)的数据框
        output_dir: 输出目录
        filename: 输出文件名
    """
    # 确保有风向角度列
    if 'wd_angle' not in df.columns and 'wd' in df.columns:
        # 将风向文本转换为角度
        wind_dir_map = {
            'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5, 
            'E': 90, 'ESE': 112.5, 'SE': 135, 'SSE': 157.5,
            'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5,
            'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5
        }
        
        # 检查wd列的类型
        if df['wd'].dtype == 'object':
            df['wd_angle'] = df['wd'].map(wind_dir_map)
        else:
            df['wd_angle'] = df['wd']
    
    # 风向分组（16个方向）
    dir_bins = np.arange(0, 361, 22.5)
    dir_labels = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 
                 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
    
    # 将风向角度转换为方向类别
    df['wind_dir_cat'] = pd.cut(df['wd_angle'], bins=dir_bins, labels=dir_labels, include_lowest=True)
    
    # 风速分组
    speed_bins = [0, 1, 2, 3, 5, 8, np.inf]
    speed_labels = ['<1', '1-2', '2-3', '3-5', '5-8', '>8']
    
    df['wind_speed_cat'] = pd.cut(df['WSPM'], bins=speed_bins, labels=speed_labels)
    
    # 统计每个风向和风速组合的频率
    wind_freq = df.groupby(['wind_dir_cat', 'wind_speed_cat']).size().unstack(fill_value=0)
    
    # 绘制风玫瑰图
    fig = plt.figure(figsize=(10, 10), dpi=80)
    ax = fig.add_subplot(111, polar=True)
    
    # 计算角度（以北为0度，顺时针方向）
    angles = np.linspace(0, 2*np.pi, len(dir_labels), endpoint=False)
    
    # 添加最后一个点以闭合图形
    angles = np.append(angles, angles[0])
    
    # 为每个风速类别绘制一个区域
    width = 2*np.pi / len(dir_labels)
    
    # 颜色映射
    cmap = cm.get_cmap('Blues')
    colors = [cmap(0.2 + 0.8*i/len(speed_labels)) for i in range(len(speed_labels))]
    
    # 从外到内绘制各风速区域
    for i, speed_cat in enumerate(speed_labels):
        if speed_cat in wind_freq.columns:
            # 获取当前风速类别的频率
            freqs = wind_freq[speed_cat].values
            
            # 添加最后一个点以闭合图形
            freqs = np.append(freqs, freqs[0])
            
            # 绘制区域
            ax.bar(angles, freqs, width=width, bottom=0.0, 
                   color=colors[i], alpha=0.7, label=speed_cat)
    
    # 设置风向标签
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dir_labels)
    
    # 设置图例和标题
    plt.legend(title='风速 (m/s)', loc='upper right', bbox_to_anchor=(1.2, 1.0))
    plt.title('风向玫瑰图 (频率分布)', fontsize=16, y=1.1)
    
    # 保存图表
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

def plot_pm25_vs_weather(df, weather_cols, output_dir):
    """
    绘制PM2.5与气象因素的关系图
    
    Args:
        df: 数据框
        weather_cols: 气象列列表
        output_dir: 输出目录
    """
    for col in weather_cols:
        plt.figure(figsize=(10, 6))
        
        plt.scatter(df[col], df['PM2.5'], alpha=0.3)
        
        # 添加趋势线
        try:
            z = np.polyfit(df[col], df['PM2.5'], 1)
            p = np.poly1d(z)
            plt.plot(df[col], p(df[col]), "r--", linewidth=2)
        except:
            print(f"无法为 {col} 添加趋势线")
        
        plt.title(f'PM2.5 vs {col}', fontsize=16)
        plt.xlabel(col, fontsize=12)
        plt.ylabel('PM2.5', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # 保存图表
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'pm25_vs_{col}.png'))
        plt.close()

def plot_pm25_by_stagnant_weather(df, output_dir, filename='pm25_by_stagnant_weather.png'):
    """
    比较静稳天气与非静稳天气下的PM2.5分布
    
    Args:
        df: 数据框
        output_dir: 输出目录
        filename: 输出文件名
    """
    plt.figure(figsize=(12, 8))
    
    # 确保有静稳天气标记
    if 'is_stagnant' not in df.columns and 'WSPM' in df.columns:
        df['is_stagnant'] = (df['WSPM'] < 1.0).astype(int)
    
    # 创建分类标签
    df['weather_type'] = df['is_stagnant'].map({1: '静稳天气 (风速<1m/s)', 0: '非静稳天气'})
    
    # 绘制分布图
    sns.violinplot(x='weather_type', y='PM2.5', data=df)
    
    plt.title('静稳天气与非静稳天气下的PM2.5分布', fontsize=16)
    plt.xlabel('天气类型', fontsize=12)
    plt.ylabel('PM2.5 浓度 (μg/m³)', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 添加均值线
    means = df.groupby('weather_type')['PM2.5'].mean()
    for i, weather_type in enumerate(['非静稳天气', '静稳天气 (风速<1m/s)']):
        if weather_type in means:
            plt.axhline(y=means[weather_type], xmin=i/2, xmax=(i+1)/2, 
                       color='red', linestyle='--', linewidth=2)
            plt.text(i, means[weather_type]*1.02, f'均值: {means[weather_type]:.1f}', 
                    color='red', ha='center')
    
    # 保存图表
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

def plot_heatmap_hourly_monthly(df, column, title, output_dir, filename):
    """
    绘制小时-月份热力图
    
    Args:
        df: 数据框
        column: 要分析的列
        title: 图表标题
        output_dir: 输出目录
        filename: 输出文件名
    """
    plt.figure(figsize=(14, 10))
    
    # 计算每月每小时的平均值
    pivot_data = df.pivot_table(values=column, index='hour', columns='month', aggfunc='mean')
    
    # 绘制热力图
    sns.heatmap(pivot_data, cmap='YlOrRd', annot=True, fmt='.1f', linewidths=.5)
    
    plt.title(title, fontsize=16)
    plt.xlabel('月份', fontsize=12)
    plt.ylabel('小时', fontsize=12)
    
    # 设置坐标轴刻度
    plt.xticks(np.arange(0.5, 12.5), range(1, 13))
    plt.yticks(np.arange(0.5, 24.5), range(0, 24))
    
    # 保存图表
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

def plot_weekday_weekend_comparison(df, columns, title, output_dir, filename):
    """
    比较工作日和周末的污染物水平
    
    Args:
        df: 数据框
        columns: 要分析的列
        title: 图表标题
        output_dir: 输出目录
        filename: 输出文件名
    """
    plt.figure(figsize=(14, 8))
    
    # 确保有工作日/周末标记
    if 'is_weekend' not in df.columns and 'day_of_week' in df.columns:
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # 创建日类型标签
    df['day_type'] = df['is_weekend'].map({1: '周末', 0: '工作日'})
    
    # 为每种污染物绘制工作日和周末的比较图
    for i, col in enumerate(columns):
        plt.subplot(1, len(columns), i+1)
        
        sns.boxplot(x='day_type', y=col, data=df)
        
        plt.title(f'{col}', fontsize=14)
        plt.xlabel('')
        plt.ylabel(col, fontsize=12)
        
        # 添加均值标注
        means = df.groupby('day_type')[col].mean()
        for j, day_type in enumerate(['工作日', '周末']):
            if day_type in means:
                plt.text(j, means[day_type]*1.02, f'{means[day_type]:.1f}', 
                        color='red', ha='center')
    
    # 添加整体标题
    plt.suptitle(title, fontsize=16, y=1.05)
    
    # 保存图表
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

def plot_pollution_by_season(df, column, title, output_dir, filename):
    """
    按季节绘制污染物箱线图
    
    Args:
        df: 数据框
        column: 要分析的污染物列
        title: 图表标题
        output_dir: 输出目录
        filename: 输出文件名
    """
    plt.figure(figsize=(12, 8))
    
    # 确保有季节列
    if 'season' not in df.columns:
        df['season'] = (df['month'] % 12 + 3) // 3
    
    # 映射季节名称
    season_names = {1: '冬季', 2: '春季', 3: '夏季', 4: '秋季'}
    df['season_name'] = df['season'].map(season_names)
    
    # 绘制箱线图
    order = ['冬季', '春季', '夏季', '秋季']  # 确保季节顺序正确
    sns.boxplot(x='season_name', y=column, data=df, order=order)
    
    plt.title(title, fontsize=16)
    plt.xlabel('季节', fontsize=12)
    plt.ylabel(f'{column} 浓度', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 添加均值标注
    means = df.groupby('season_name')[column].mean()
    for i, season in enumerate(order):
        if season in means:
            plt.text(i, means[season]*1.02, f'{means[season]:.1f}', 
                    color='red', ha='center')
    
    # 保存图表
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

def plot_station_statistics(df, output_dir='../processed/figures'):
    """绘制不同站点的PM2.5统计信息"""
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 按站点分组计算统计量
    station_stats = df.groupby('station').agg({
        'PM2.5': ['mean', 'std', 'min', 'max', 'median']
    }).reset_index()
    
    # 构造DataFrame便于绘图
    station_stats.columns = ['station', 'mean', 'std', 'min', 'max', 'median']
    
    # 保留3位小数
    for col in ['mean', 'std', 'min', 'max', 'median']:
        station_stats[col] = station_stats[col].round(3)
    
    # 绘制均值柱状图
    plt.figure(figsize=(12, 6))
    bars = plt.bar(station_stats['station'], station_stats['mean'], yerr=station_stats['std'], 
            capsize=5, color='skyblue', alpha=0.7)
    
    # 在柱状图上标注均值
    for i, bar in enumerate(bars):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 5, 
                f"{station_stats['mean'].iloc[i]:.3f}", 
                ha='center', va='bottom', rotation=0)
    
    plt.title('不同监测站点PM2.5均值对比', fontsize=14)
    plt.xlabel('监测站点')
    plt.ylabel('PM2.5均值 (μg/m³)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/station_pm25_mean.png", dpi=300)
    plt.close()
    
    # 绘制箱线图
    plt.figure(figsize=(14, 8))
    station_data = {station: df[df['station'] == station]['PM2.5'] for station in df['station'].unique()}
    plt.boxplot([station_data[station] for station in df['station'].unique()], 
                labels=df['station'].unique(), showfliers=False)
    
    # 添加中位数标注
    for i, station in enumerate(df['station'].unique(), 1):
        median_val = station_data[station].median().round(3)
        plt.text(i, median_val + 5, f"{median_val:.3f}", 
                ha='center', va='bottom', fontsize=9)
    
    plt.title('不同监测站点PM2.5分布箱线图', fontsize=14)
    plt.ylabel('PM2.5浓度 (μg/m³)')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/station_pm25_boxplot.png", dpi=300)
    plt.close()
    
    # 保存统计信息到CSV
    station_stats.to_csv(f"{output_dir}/station_pm25_stats.csv", index=False)
    print(f"已保存站点统计图表到 {output_dir}")


def plot_temporal_patterns(df, output_dir='../processed/figures'):
    """绘制PM2.5的时间变化模式"""
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 按小时统计PM2.5均值
    hourly_avg = df.groupby('hour')['PM2.5'].mean().round(3).reset_index()
    
    plt.figure(figsize=(12, 6))
    plt.plot(hourly_avg['hour'], hourly_avg['PM2.5'], 'o-', linewidth=2, markersize=8)
    
    # 添加数值标签
    for i, val in enumerate(hourly_avg['PM2.5']):
        plt.text(hourly_avg['hour'][i], val + 2, f"{val:.3f}", ha='center', va='bottom', fontsize=9)
    
    plt.title('PM2.5浓度的小时变化模式', fontsize=14)
    plt.xlabel('小时 (时)')
    plt.ylabel('PM2.5平均浓度 (μg/m³)')
    plt.xticks(range(0, 24, 2))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/pm25_hourly_pattern.png", dpi=300)
    plt.close()
    
    # 2. 按月统计PM2.5均值
    monthly_avg = df.groupby('month')['PM2.5'].mean().round(3).reset_index()
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(monthly_avg['month'], monthly_avg['PM2.5'], color='salmon', alpha=0.7)
    
    # 添加数值标签
    for i, bar in enumerate(bars):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 2, 
                f"{monthly_avg['PM2.5'].iloc[i]:.3f}", 
                ha='center', va='bottom')
    
    plt.title('PM2.5浓度的月度变化', fontsize=14)
    plt.xlabel('月份')
    plt.ylabel('PM2.5平均浓度 (μg/m³)')
    plt.xticks(range(1, 13))
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/pm25_monthly_pattern.png", dpi=300)
    plt.close()
    
    # 3. 按星期统计PM2.5均值
    weekly_avg = df.groupby('day_of_week')['PM2.5'].mean().round(3).reset_index()
    day_names = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
    weekly_avg['day_name'] = weekly_avg['day_of_week'].apply(lambda x: day_names[x])
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(weekly_avg['day_name'], weekly_avg['PM2.5'], color='lightgreen', alpha=0.7)
    
    # 添加数值标签
    for i, bar in enumerate(bars):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 2, 
                f"{weekly_avg['PM2.5'].iloc[i]:.3f}", 
                ha='center', va='bottom')
    
    plt.title('PM2.5浓度的星期变化', fontsize=14)
    plt.xlabel('星期')
    plt.ylabel('PM2.5平均浓度 (μg/m³)')
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/pm25_weekly_pattern.png", dpi=300)
    plt.close()
    
    # 4. 按季节统计PM2.5均值
    season_avg = df.groupby('season')['PM2.5'].mean().round(3).reset_index()
    season_names = ['春', '夏', '秋', '冬']
    season_avg['season_name'] = season_avg['season'].apply(lambda x: season_names[x-1])
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(season_avg['season_name'], season_avg['PM2.5'], color='lightblue', alpha=0.7)
    
    # 添加数值标签
    for i, bar in enumerate(bars):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 2, 
                f"{season_avg['PM2.5'].iloc[i]:.3f}", 
                ha='center', va='bottom')
    
    plt.title('PM2.5浓度的季节变化', fontsize=14)
    plt.xlabel('季节')
    plt.ylabel('PM2.5平均浓度 (μg/m³)')
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/pm25_seasonal_pattern.png", dpi=300)
    plt.close()
    
    # 保存时间模式统计数据
    temporal_stats = pd.DataFrame({
        '小时': hourly_avg['hour'].tolist() + [None] * (12 - len(hourly_avg)),
        '小时均值': hourly_avg['PM2.5'].tolist() + [None] * (12 - len(hourly_avg)),
        '月份': monthly_avg['month'].tolist(),
        '月均值': monthly_avg['PM2.5'].tolist(),
        '星期': weekly_avg['day_name'].tolist() + [None] * (12 - len(weekly_avg)),
        '星期均值': weekly_avg['PM2.5'].tolist() + [None] * (12 - len(weekly_avg)),
        '季节': season_avg['season_name'].tolist() + [None] * (12 - len(season_avg)),
        '季节均值': season_avg['PM2.5'].tolist() + [None] * (12 - len(season_avg))
    })
    
    temporal_stats.to_csv(f"{output_dir}/temporal_pattern_stats.csv", index=False)
    print(f"已保存时间模式图表到 {output_dir}")


def plot_meteorological_influence(df, output_dir='../processed/figures'):
    """绘制气象因素对PM2.5的影响"""
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. TEMP与PM2.5关系的散点图
    plt.figure(figsize=(10, 6))
    plt.scatter(df['TEMP'], df['PM2.5'], alpha=0.3, s=10)
    
    # 添加趋势线
    z = np.polyfit(df['TEMP'], df['PM2.5'], 1)
    p = np.poly1d(z)
    trend_x = np.linspace(df['TEMP'].min(), df['TEMP'].max(), 100)
    plt.plot(trend_x, p(trend_x), "r--", linewidth=2)
    
    # 添加相关系数
    corr = df[['TEMP', 'PM2.5']].corr().iloc[0, 1].round(3)
    plt.annotate(f"相关系数: {corr:.3f}", xy=(0.05, 0.95), xycoords='axes fraction',
                 fontsize=12, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    plt.title('温度与PM2.5的关系', fontsize=14)
    plt.xlabel('温度 (°C)')
    plt.ylabel('PM2.5浓度 (μg/m³)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/pm25_vs_temperature.png", dpi=300)
    plt.close()
    
    # 2. WSPM（风速）与PM2.5关系的散点图
    plt.figure(figsize=(10, 6))
    plt.scatter(df['WSPM'], df['PM2.5'], alpha=0.3, s=10)
    
    # 添加趋势线
    z = np.polyfit(df['WSPM'], df['PM2.5'], 1)
    p = np.poly1d(z)
    trend_x = np.linspace(df['WSPM'].min(), df['WSPM'].max(), 100)
    plt.plot(trend_x, p(trend_x), "r--", linewidth=2)
    
    # 添加相关系数
    corr = df[['WSPM', 'PM2.5']].corr().iloc[0, 1].round(3)
    plt.annotate(f"相关系数: {corr:.3f}", xy=(0.05, 0.95), xycoords='axes fraction',
                 fontsize=12, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    plt.title('风速与PM2.5的关系', fontsize=14)
    plt.xlabel('风速 (m/s)')
    plt.ylabel('PM2.5浓度 (μg/m³)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/pm25_vs_wind_speed.png", dpi=300)
    plt.close()
    
    # 3. 湿度（DEWP）与PM2.5关系的散点图
    plt.figure(figsize=(10, 6))
    plt.scatter(df['DEWP'], df['PM2.5'], alpha=0.3, s=10)
    
    # 添加趋势线
    z = np.polyfit(df['DEWP'], df['PM2.5'], 1)
    p = np.poly1d(z)
    trend_x = np.linspace(df['DEWP'].min(), df['DEWP'].max(), 100)
    plt.plot(trend_x, p(trend_x), "r--", linewidth=2)
    
    # 添加相关系数
    corr = df[['DEWP', 'PM2.5']].corr().iloc[0, 1].round(3)
    plt.annotate(f"相关系数: {corr:.3f}", xy=(0.05, 0.95), xycoords='axes fraction',
                 fontsize=12, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    plt.title('露点温度与PM2.5的关系', fontsize=14)
    plt.xlabel('露点温度 (°C)')
    plt.ylabel('PM2.5浓度 (μg/m³)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/pm25_vs_dew_point.png", dpi=300)
    plt.close()
    
    # 4. 气压（PRES）与PM2.5关系的散点图
    plt.figure(figsize=(10, 6))
    plt.scatter(df['PRES'], df['PM2.5'], alpha=0.3, s=10)
    
    # 添加趋势线
    z = np.polyfit(df['PRES'], df['PM2.5'], 1)
    p = np.poly1d(z)
    trend_x = np.linspace(df['PRES'].min(), df['PRES'].max(), 100)
    plt.plot(trend_x, p(trend_x), "r--", linewidth=2)
    
    # 添加相关系数
    corr = df[['PRES', 'PM2.5']].corr().iloc[0, 1].round(3)
    plt.annotate(f"相关系数: {corr:.3f}", xy=(0.05, 0.95), xycoords='axes fraction',
                 fontsize=12, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    plt.title('气压与PM2.5的关系', fontsize=14)
    plt.xlabel('气压 (hPa)')
    plt.ylabel('PM2.5浓度 (μg/m³)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/pm25_vs_pressure.png", dpi=300)
    plt.close()
    
    # 保存气象因素相关性统计
    meteo_corr = df[['PM2.5', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']].corr()['PM2.5'].round(3)
    meteo_corr.to_csv(f"{output_dir}/meteorological_correlations.csv")
    print(f"已保存气象因素影响图表到 {output_dir}")


def plot_pollution_rose(df, output_dir='../processed/figures'):
    """绘制污染玫瑰图（PM2.5随风向的变化）"""
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 检查是否有风向角度列
    if 'wd_angle' not in df.columns:
        print("缺少风向角度数据，无法绘制污染玫瑰图")
        return
    
    # 创建风向bins（按16个方位划分）
    bins = np.arange(0, 361, 22.5)
    labels = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 
              'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
    
    # 添加风向分类
    df_wind = df.copy()
    df_wind['wind_dir_cat'] = pd.cut(df_wind['wd_angle'], bins=bins, labels=labels, include_lowest=True)
    
    # 按风向计算PM2.5的平均值
    wind_pm25 = df_wind.groupby('wind_dir_cat')['PM2.5'].mean().round(3)
    
    # 转换为径向图的数据格式
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    values = wind_pm25.tolist()
    # 闭合图形
    angles.append(angles[0])
    values.append(values[0])
    
    # 创建污染玫瑰图
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
    ax.plot(angles, values, linewidth=2)
    ax.fill(angles, values, alpha=0.25)
    
    # 添加数值标签
    for i, (angle, value) in enumerate(zip(angles[:-1], values[:-1])):
        ha = 'left' if 0 <= angle < np.pi else 'right'
        ax.annotate(f"{value:.3f}", xy=(angle, value + max(values)*0.05), 
                    ha=ha, va='center', fontsize=8)
    
    # 设置图表属性
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_title('风向与PM2.5浓度玫瑰图', fontsize=14, pad=20)
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/pm25_wind_rose.png", dpi=300)
    plt.close()
    
    # 保存风向统计
    wind_stats = pd.DataFrame({
        '风向': labels,
        'PM2.5均值': wind_pm25.values.round(3)
    })
    wind_stats.to_csv(f"{output_dir}/wind_direction_pm25.csv", index=False)
    print(f"已保存污染玫瑰图到 {output_dir}")


def visualize_station_distribution(df, output_dir='../processed/figures'):
    """可视化不同站点的空间分布"""
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 站点经纬度数据 (模拟数据，实际应使用真实经纬度)
    # 由于题目中没有给出站点经纬度，我们使用模拟数据进行演示
    station_locations = {
        'Dongsi': {'lat': 39.929, 'lon': 116.417},
        'Dongsihuan': {'lat': 39.915, 'lon': 116.434},
        'Nongzhanguan': {'lat': 39.937, 'lon': 116.461},
        'US Embassy': {'lat': 39.954, 'lon': 116.466},
        'Wanliu': {'lat': 39.987, 'lon': 116.287},
        'Wanshouxigong': {'lat': 39.878, 'lon': 116.352},
        'Xizhimenbei': {'lat': 39.954, 'lon': 116.349},
        'Yungang': {'lat': 39.824, 'lon': 116.146},
        'Zhiwuyuan': {'lat': 39.941, 'lon': 116.207},
        'Changping': {'lat': 40.217, 'lon': 116.230},
        'Dingling': {'lat': 40.290, 'lon': 116.220},
        'Shunyi': {'lat': 40.125, 'lon': 116.655},
        'Huairou': {'lat': 40.330, 'lon': 116.628}
    }
    
    # 计算每个站点的PM2.5平均值
    station_pm25 = df.groupby('station')['PM2.5'].mean().round(3)
    
    # 创建站点数据框
    stations_df = pd.DataFrame(station_locations).T.reset_index()
    stations_df.columns = ['station', 'lat', 'lon']
    stations_df = stations_df.merge(station_pm25.reset_index(), on='station')
    
    # 绘制站点分布图
    plt.figure(figsize=(12, 10))
    
    # 背景地图（简化版，实际应使用地图API）
    plt.scatter(stations_df['lon'], stations_df['lat'], 
                c=stations_df['PM2.5'], cmap='YlOrRd', 
                s=200, alpha=0.7, edgecolors='black', zorder=3)
    
    # 添加站点名称标签
    for i, row in stations_df.iterrows():
        plt.annotate(f"{row['station']}\n({row['PM2.5']:.3f})", 
                     (row['lon'], row['lat']),
                     xytext=(5, 5), textcoords='offset points',
                     fontsize=8, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    plt.colorbar(label='PM2.5平均浓度 (μg/m³)')
    plt.title('北京市监测站点PM2.5空间分布', fontsize=14)
    plt.xlabel('经度')
    plt.ylabel('纬度')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/station_spatial_distribution.png", dpi=300)
    plt.close()
    
    # 保存站点位置和PM2.5数据
    stations_df.to_csv(f"{output_dir}/station_locations_pm25.csv", index=False)
    print(f"已保存站点分布图到 {output_dir}")


def run_visualization(data_path='../processed/processed_data.csv', output_dir='../processed/figures'):
    """运行所有可视化函数"""
    
    print("开始数据可视化...")
    
    # 检查输入文件是否存在，如果不存在尝试其他位置
    if not os.path.exists(data_path):
        print(f"输入文件不存在: {data_path}")
        # 尝试在当前目录查找
        alt_data_path = "./processed_data.csv"
        if os.path.exists(alt_data_path):
            data_path = alt_data_path
            print(f"使用替代输入文件: {data_path}")
        else:
            print("无法找到输入文件，请确保已运行数据预处理脚本")
            return None
    
    # 确保输出目录存在
    output_dir = create_output_dir(output_dir)
    
    try:
        # 加载处理后的数据
        df = pd.read_csv(data_path)
        print(f"数据已加载，形状: {df.shape}")
        
        # 如果数据中包含datetime列但格式是字符串，转换为日期时间类型
        if 'datetime' in df.columns and df['datetime'].dtype == 'object':
            df['datetime'] = pd.to_datetime(df['datetime'])
            print("已将datetime列转换为日期时间类型")
        elif 'datetime' not in df.columns and all(col in df.columns for col in ['year', 'month', 'day', 'hour']):
            # 创建datetime列
            time_cols = df[['year', 'month', 'day', 'hour']].astype(str)
            df['datetime'] = pd.to_datetime(
                time_cols['year'] + '-' + time_cols['month'] + '-' + 
                time_cols['day'] + ' ' + time_cols['hour'] + ':00:00'
            )
            print("已从年月日时列创建datetime列")
        
        # 运行各个可视化函数，每个函数都加入异常处理
        try:
            if 'station' in df.columns:
                plot_station_statistics(df, output_dir)
                visualize_station_distribution(df, output_dir)
            else:
                print("数据中缺少station列，无法绘制站点相关图表")
        except Exception as e:
            print(f"绘制站点统计图表时出错: {e}")
        
        try:
            plot_temporal_patterns(df, output_dir)
        except Exception as e:
            print(f"绘制时间模式图表时出错: {e}")
        
        try:
            plot_meteorological_influence(df, output_dir)
        except Exception as e:
            print(f"绘制气象影响图表时出错: {e}")
        
        try:
            # 使用更详细的相关性热力图函数，传递所有主要特征
            columns = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 
                      'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']
            # 添加可能存在的周期性特征
            if 'hour_sin' in df.columns and 'hour_cos' in df.columns:
                columns.extend(['hour_sin', 'hour_cos', 'month_sin', 'month_cos'])
            plot_correlation_heatmap(df, columns, "特征相关性热力图", 
                                    output_dir, "correlation_heatmap.png")
        except Exception as e:
            print(f"绘制相关性热力图时出错: {e}")
        
        try:
            if 'wd_angle' in df.columns:
                plot_pollution_rose(df, output_dir)
            else:
                print("数据中缺少wd_angle列，无法绘制污染玫瑰图")
        except Exception as e:
            print(f"绘制污染玫瑰图时出错: {e}")
        
        print("数据可视化完成！所有图表已保存到", output_dir)
        
        return df
    except Exception as e:
        print(f"运行可视化过程中发生错误: {e}")
        return None

def main():
    # 定义输入和输出路径
    input_file = "../processed/processed_data.csv"  # 从processed文件夹读取预处理后的数据
    
    # 检查输入文件是否存在，如果不存在尝试其他位置
    if not os.path.exists(input_file):
        print(f"输入文件不存在: {input_file}")
        # 尝试在当前目录查找
        alt_input_file = "./processed_data.csv"
        if os.path.exists(alt_input_file):
            input_file = alt_input_file
            print(f"使用替代输入文件: {input_file}")
        else:
            print("无法找到输入文件，请确保已运行数据预处理脚本")
            return
    
    output_dir = create_output_dir('../processed/figures')  # 将图表保存到processed/figures目录下
    
    # 1. 加载数据
    try:
        df = load_data(input_file)
    except Exception as e:
        print(f"加载数据失败: {e}")
        return
    
    # 显示数据基本信息
    print("\n数据基本信息:")
    print(df.info())
    
    # 2. 时间序列分析
    print("\n绘制污染物时间序列图...")
    plot_time_series(df, ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3'], 
                    '北京东四站点主要污染物浓度时间序列 (日均值)', 
                    output_dir, 'pollutants_time_series.png')
    
    # 3. 绘制PM2.5时间序列的月均值和日均值
    plot_time_series(df, ['PM2.5'], 'PM2.5浓度时间序列 (月均值)', 
                    output_dir, 'pm25_monthly_time_series.png', freq='ME')
    
    plot_time_series(df, ['PM2.5'], 'PM2.5浓度时间序列 (日均值)', 
                    output_dir, 'pm25_daily_time_series.png', freq='D')
    
    # 4. 相关性分析
    print("\n绘制相关性热力图...")
    # 污染物之间的相关性
    plot_correlation_heatmap(df, ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3'], 
                            '污染物之间的相关性', output_dir, 'pollutants_correlation.png')
    
    # 污染物与气象因子的相关性
    plot_correlation_heatmap(df, ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM'], 
                            '污染物与气象因子的相关性', output_dir, 'pollutants_weather_correlation.png')
    
    # 5. 日内变化模式
    print("\n分析日内变化模式...")
    plot_daily_pattern(df, ['PM2.5', 'PM10', 'NO2', 'CO'], 
                      '主要污染物日内变化模式', output_dir, 'daily_pattern_pollutants.png')
    
    plot_daily_pattern(df, ['TEMP', 'WSPM'], 
                      '温度和风速日内变化模式', output_dir, 'daily_pattern_weather.png')
    
    # 6. 季节性变化
    print("\n分析季节性变化...")
    plot_seasonal_pattern(df, ['PM2.5', 'PM10', 'SO2', 'NO2'], 
                         '主要污染物月度变化模式', output_dir, 'monthly_pattern_pollutants.png')
    
    plot_seasonal_pattern(df, ['TEMP', 'PRES', 'RAIN'], 
                         '主要气象要素月度变化模式', output_dir, 'monthly_pattern_weather.png')
    
    # 7. 风玫瑰图
    print("\n绘制风玫瑰图...")
    plot_wind_rose(df, output_dir, 'wind_rose.png')
    
    # 8. PM2.5与气象因素的关系
    print("\n分析PM2.5与气象因素的关系...")
    plot_pm25_vs_weather(df, ['TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM'], output_dir)
    
    # 9. 静稳天气分析
    print("\n分析静稳天气对PM2.5的影响...")
    plot_pm25_by_stagnant_weather(df, output_dir)
    
    # 10. 小时-月份热力图
    print("\n绘制小时-月份热力图...")
    plot_heatmap_hourly_monthly(df, 'PM2.5', 'PM2.5小时-月份分布热力图', 
                               output_dir, 'pm25_hourly_monthly_heatmap.png')
    
    # 11. 工作日与周末比较
    print("\n比较工作日和周末的污染水平...")
    plot_weekday_weekend_comparison(df, ['PM2.5', 'PM10', 'NO2', 'CO'], 
                                  '工作日与周末污染物水平比较', 
                                  output_dir, 'weekday_weekend_comparison.png')
    
    # 12. 季节性分析
    print("\n按季节分析污染物水平...")
    plot_pollution_by_season(df, 'PM2.5', 'PM2.5季节分布', 
                            output_dir, 'pm25_by_season.png')
    
    print(f"\n数据可视化完成！所有图表已保存至 {output_dir} 目录")

if __name__ == "__main__":
    main() 