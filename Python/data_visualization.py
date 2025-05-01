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
    except UnicodeDecodeError:
        # 如果默认编码失败，尝试其他常见编码
        for encoding in ['utf-8', 'gbk', 'gb2312', 'gb18030']:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                print(f"成功使用 {encoding} 编码读取文件")
                break
            except UnicodeDecodeError:
                continue
    
    # 确保有datetime列，用于时间序列分析
    if 'datetime' not in df.columns and all(col in df.columns for col in ['year', 'month', 'day', 'hour']):
        # df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']].assign(
        #     hour=lambda x: x['hour'].astype(str) + ':00:00'
        # ).agg(' '.join, axis=1), format='%Y %m %d %H:%M:%S')
        
        # 修复：先将所有列转换为字符串，然后再连接
        time_cols = df[['year', 'month', 'day', 'hour']].astype(str)
        df['datetime'] = pd.to_datetime(
            time_cols['year'] + ' ' + time_cols['month'] + ' ' + 
            time_cols['day'] + ' ' + time_cols['hour'] + ':00:00',
            format='%Y %m %d %H:%M:%S'
        )
    
    print(f"数据加载完成，形状: {df.shape}")
    return df

def create_output_dir(output_dir='./figures'):
    """
    创建输出目录
    
    Args:
        output_dir: 输出目录路径
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
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
        freq: 重采样频率，'D'表示日均值，'M'表示月均值
    """
    plt.figure(figsize=(16, 8))
    
    # 设置时间索引
    df_temp = df.copy()
    df_temp.set_index('datetime', inplace=True)
    
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

def main():
    # 定义输入和输出路径
    input_file = "../processed/processed_data.csv"  # 从processed文件夹读取预处理后的数据
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
                    output_dir, 'pm25_monthly_time_series.png', freq='M')
    
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