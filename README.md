# Wenqing_W
code for 2025 machine learning course

本项目基于时间序列建模，预测家庭每日 `Global Active Power`（总有功功率）消耗，采用多种模型（LSTM、Transformer、趋势-季节性分解模型）进行短期与长期预测，结合可视化和误差评估分析。

在提出自己方案阶段，我们主要想要提升长期预测的性能，使用对原始特征进行趋势（trend）与季节性（seasonal）分解后建模，结合滑动平均方式进行趋势分解，加入可学习的融合参数 alpha，实现趋势与季节性的自适应加权预测。
实验证明我们的方案在长期预测上有了性能提升。

下面是我们的项目结构，可以执行main.py运行，同时各个文件中包含训练好的模型。

##  项目结构

```bash
.
├── Data_Processing.ipynb              # 数据预处理与探索性分析
├── Processed_train.csv                # 已预处理训练集
├── Processed_test.csv                 # 已预处理测试集

├── LSTM/                              # LSTM 模型实现
│   ├── lstm_model.py                  # LSTM 模型结构
│   ├── preprocess.py                  # 特征工程与归一化
│   ├── train.py                       # 训练入口
│   ├── visualize.py                   # 可视化函数
│   ├── data/                          # 原始数据目录
│   ├── models/                        # 模型保存路径
│   ├── processed_data/                # 预处理后的数据
│   └── results/                       # 可视化结果输出目录

├── Transformer/                       # Transformer 模型实现
│   ├── dataset.py                     # 数据集类
│   ├── model.py                       # Transformer 模型结构
│   ├── main.py                        # 训练和评估入口
│   ├── utils.py                       # 工具函数
│   ├── model_check/                   # 保存模型 checkpoint
│   └── Result_Figure/                 # 可视化图保存目录

├── OurMethod/                         # 改进模型（趋势+季节性）
│   ├── data_utils.py                  # 数据加载与归一化
│   ├── main.py                        # 主程序入口
│   ├── models.py                      # 模型定义（含DecomposedTransformer）
│   ├── plot_utils.py                  # 拆解可视化函数
│   ├── train_eval.py                  # 训练与评估函数
│   ├── model_check/                   # 模型保存目录
│   └── Result_Figure/                 # 可视化图输出目录
