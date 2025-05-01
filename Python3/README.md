# 问题三：PM2.5减排策略优化

本目录包含问题三的实现代码，用于基于问题二构建的PM2.5预测模型，实现减排策略的优化。

## 项目结构

- `main.py`: 主程序，整合模型加载和减排策略优化的完整流程
- `model_loading.py`: 模型加载模块，用于加载问题二训练好的模型并预测目标日期的PM2.5浓度
- `emission_reduction_optimizer.py`: 减排策略优化模块，构建和求解线性规划模型，生成可视化结果和报告

## 运行环境要求

- Python 3.6+
- 依赖库：
  - pandas
  - numpy
  - matplotlib
  - pulp (线性规划求解器)
  - joblib (模型加载)

## 安装依赖

```bash
pip install pandas numpy matplotlib pulp joblib
```

## 运行方法

### 完整运行流程（包括预测和优化）

```bash
python main.py
```

### 仅运行减排策略优化（跳过预测步骤）

如果已有预测结果，可以跳过预测步骤，直接进行减排策略优化：

```bash
python main.py --skip-prediction
```

### 单独运行各模块

1. 仅预测2016年3月1日PM2.5浓度：

```bash
python model_loading.py
```

2. 仅进行减排策略优化（需要先有预测结果）：

```bash
python emission_reduction_optimizer.py
```

## 输出结果

所有结果文件将保存在 `../processed3/` 目录下：

- `predicted_pm25.txt`: 2016年3月1日PM2.5日均浓度预测结果
- `emission_reduction_report.md`: 减排策略优化报告（包含优化结果和建议）
- `figures/`: 各种可视化图表
  - `target_date_pm25_prediction.png`: 2016年3月1日PM2.5小时浓度预测与实际值对比图
  - `measures_implementation_hours.png`: 各减排措施实施时长条形图
  - `cost_distribution_pie.png`: 总成本分布饼图
  - `reduction_distribution_pie.png`: 减排效果分布饼图
  - `implementation_schedule.png`: 减排措施实施时间计划甘特图
  - `cost_efficiency_comparison.png`: 各减排措施成本效益比较图

## 模型说明

### 减排优化模型

使用线性规划模型进行减排策略优化，具有以下特点：

1. **决策变量**：
   - x₁: 限行措施实施小时数
   - x₂: 工厂限产措施实施小时数
   - x₃: 洒水抑尘措施实施小时数

2. **目标函数**：
   - 最小化总成本：min Z = 5000x₁ + 8000x₂ + 3000x₃

3. **约束条件**：
   - 减排效果约束：3x₁ + 5x₂ + 2x₃ ≥ 目标减排量
   - 预算约束：5000x₁ + 8000x₂ + 3000x₃ ≤ 100000
   - 时间约束：0 ≤ x₁, x₂, x₃ ≤ 16
   - 整数约束：x₁, x₂, x₃ 为非负整数

## 注意事项

1. 该代码依赖于问题二的模型和数据，需要确保问题二的代码已正确运行
2. 模型优化基于以下减排措施参数：
   - 限行：成本5000元/小时，减排效率3μg/m³·h
   - 工厂限产：成本8000元/小时，减排效率5μg/m³·h
   - 洒水抑尘：成本3000元/小时，减排效率2μg/m³·h 