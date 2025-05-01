# PM2.5预测与静稳天气分析

## 项目概述

本项目旨在构建PM2.5预测模型，并研究预测模型在静稳天气（风速<1m/s）下的预测误差是否显著增大，同时提出改进策略。

主要功能包括：
1. 数据加载与预处理
2. 特征工程
3. 多种预测模型训练与评估
4. 静稳天气条件下的模型性能分析
5. 针对静稳天气的专门模型与集成优化

## 项目结构

```
Python2/                  # 代码目录
├── README.md             # 项目说明文件
├── data_loader.py        # 数据加载与预处理模块
├── feature_engineering.py # 特征工程模块
├── model_training.py     # 模型训练与评估模块
├── model_improvement.py  # 模型改进模块
└── main.py               # 主程序入口

processed2/               # 处理结果目录
├── train_data.csv        # 训练数据
├── test_data.csv         # 测试数据
├── processed_train.csv   # 特征工程后的训练数据
├── processed_test.csv    # 特征工程后的测试数据
├── data_info.txt         # 数据基本信息
├── features_info.csv     # 特征信息
├── descriptive_stats.csv # 数据描述性统计
├── conclusion_report.txt # 结论报告
├── model_improvement_report.txt # 模型改进报告
├── figures/              # 图表目录
│   ├── feature_importance.png # 特征重要性图
│   ├── pm25_by_stagnant_boxplot.png # 静稳天气PM2.5箱线图
│   └── ... 其他图表
└── models/               # 模型保存目录
    ├── RF_model.pkl      # 随机森林模型
    ├── XGB_model.pkl     # XGBoost模型
    ├── hybrid_model.pkl  # 混合模型
    └── ... 其他模型文件
```

## 模块说明

### 1. data_loader.py

数据加载与预处理模块，主要功能：
- 加载原始数据集
- 分析静稳天气情况
- 计算基本统计信息
- 创建训练集和测试集

### 2. feature_engineering.py

特征工程模块，主要功能：
- 添加时间特征（小时、星期、月份、季节等）
- 添加滞后特征（历史PM2.5水平）
- 添加滚动特征（移动平均、标准差等）
- 添加气象特征与交互特征
- 特征重要性分析与选择

### 3. model_training.py

模型训练与评估模块，主要功能：
- 训练多种预测模型（线性回归、随机森林、XGBoost等）
- 超参数调优
- 模型性能评估
- 静稳天气条件下的性能分析
- 预测结果可视化

### 4. model_improvement.py

模型改进模块，主要功能：
- 为静稳天气和非静稳天气分别构建专门模型
- 构建集成模型
- 开发混合预测策略
- 评估改进效果并生成报告

### 5. main.py

主程序入口，集成各模块功能：
- 支持通过命令行参数控制执行流程
- 按顺序执行数据加载、特征工程、模型训练和评估

## 使用方法

### 环境要求

```
Python 3.8+
pandas
numpy
scikit-learn
matplotlib
seaborn
xgboost
lightgbm
joblib
tqdm
```

### 运行完整流程

```bash
python main.py
```

### 运行部分流程

```bash
# 跳过数据加载步骤
python main.py --skip-data-loading

# 跳过特征工程步骤
python main.py --skip-feature-engineering

# 跳过模型训练步骤
python main.py --skip-model-training
```

### 单独运行某一模块

```bash
# 数据加载与预处理
python data_loader.py

# 特征工程
python feature_engineering.py

# 模型训练与评估
python model_training.py

# 模型改进
python model_improvement.py
```

## 模型改进策略

针对静稳天气下预测误差较大的问题，本项目提出以下改进策略：

1. **专门化建模**：为静稳天气和非静稳天气分别构建专门的预测模型
2. **集成学习**：结合多种基础模型，提高预测稳定性
3. **混合预测器**：根据天气条件动态结合专门模型和集成模型的预测结果
4. **加权策略**：对静稳天气样本，赋予专门模型更高的权重；对非静稳天气样本，赋予集成模型更高的权重

## 结果分析

模型评估报告和可视化结果位于`processed2/`目录，包括：

- 各模型性能指标对比
- 静稳天气条件下的预测误差分析
- 改进前后的模型性能对比
- 预测结果可视化图表

## 未来工作

- 引入更多描述静稳天气的特征变量
- 探索深度学习方法（LSTM、Transformer等）
- 引入外部数据源（边界层高度、卫星数据等）
- 开发自适应权重的混合预测策略 