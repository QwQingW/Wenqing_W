import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 路径设置
processed_dir = 'processed_data'
model_dir = 'models'
result_dir = 'results'
os.makedirs(result_dir, exist_ok=True)

# 加载测试数据
testX_short = np.load(os.path.join(processed_dir, 'testX_short.npy'))
testY_short = np.load(os.path.join(processed_dir, 'testY_short.npy'))
testX_long = np.load(os.path.join(processed_dir, 'testX_long.npy'))
testY_long = np.load(os.path.join(processed_dir, 'testY_long.npy'))

# 加载 scaler 参数
scaler_min = np.load(os.path.join(processed_dir, 'scaler_min.npy'))
scaler_max = np.load(os.path.join(processed_dir, 'scaler_max.npy'))

def denormalize(data):
    return data * (scaler_max - scaler_min) + scaler_min

def calculate_metrics(pred, true):
    pred_denorm = denormalize(pred)
    true_denorm = denormalize(true)
    mse = mean_squared_error(true_denorm, pred_denorm)
    mae = mean_absolute_error(true_denorm, pred_denorm)
    return mse, mae

# 多轮评估函数
def run_experiments(rounds=5):
    short_mse_list, short_mae_list = [], []
    long_mse_list, long_mae_list = [], []

    print("开始进行 5 轮实验评估：")
    for i in range(rounds):
        # 加载模型
        model = load_model(os.path.join(model_dir, f'lstm_short_term_round_{i}.keras'), compile=False)
        model.compile(optimizer='adam', loss='mse')

        # 短期预测
        pred_short = model.predict(testX_short).flatten()
        pred_short_denorm = denormalize(pred_short)
        true_short_denorm = denormalize(testY_short)
        mse_short, mae_short = calculate_metrics(pred_short, testY_short)

        # 长期预测
        pred_long = model.predict(testX_long).flatten()
        pred_long_denorm = denormalize(pred_long)
        true_long_denorm = denormalize(testY_long)
        mse_long, mae_long = calculate_metrics(pred_long, testY_long)

        # 存储指标
        short_mse_list.append(mse_short)
        short_mae_list.append(mae_short)
        long_mse_list.append(mse_long)
        long_mae_list.append(mae_long)

        # 打印每轮结果
        print(f"Round {i+1:2d} | 短期预测 MSE: {mse_short:.4f}, MAE: {mae_short:.4f} | "
              f"长期预测 MSE: {mse_long:.4f}, MAE: {mae_long:.4f}")

    # 计算均值和标准差
    results = {
        "short": {
            "mse_avg": np.mean(short_mse_list),
            "mse_std": np.std(short_mse_list),
            "mae_avg": np.mean(short_mae_list),
            "mae_std": np.std(short_mae_list)
        },
        "long": {
            "mse_avg": np.mean(long_mse_list),
            "mse_std": np.std(long_mse_list),
            "mae_avg": np.mean(long_mae_list),
            "mae_std": np.std(long_mae_list)
        }
    }

    # 打印汇总结果
    print("\n五轮实验平均结果：")
    print(f"短期预测平均 MSE: {results['short']['mse_avg']:.4f} ± {results['short']['mse_std']:.4f}, "
          f"MAE: {results['short']['mae_avg']:.4f} ± {results['short']['mae_std']:.4f}")
    print(f"长期预测平均 MSE: {results['long']['mse_avg']:.4f} ± {results['long']['mse_std']:.4f}, "
          f"MAE: {results['long']['mae_avg']:.4f} ± {results['long']['mae_std']:.4f}\n")

    # 保存结果到文件
    result_file = os.path.join(result_dir, 'evaluation_results.txt')
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write("五轮实验详细结果报告\n\n")
        f.write("每轮实验结果：\n")
        for i in range(rounds):
            f.write(f"Round {i+1:2d} | 短期预测 MSE: {short_mse_list[i]:.4f}, MAE: {short_mae_list[i]:.4f} | "
                    f"长期预测 MSE: {long_mse_list[i]:.4f}, MAE: {long_mae_list[i]:.4f}\n")
        f.write("\n五轮实验平均结果：\n")
        f.write(f"短期预测平均 MSE: {results['short']['mse_avg']:.4f} ± {results['short']['mse_std']:.4f}, "
                f"MAE: {results['short']['mae_avg']:.4f} ± {results['short']['mae_std']:.4f}\n")
        f.write(f"长期预测平均 MSE: {results['long']['mse_avg']:.4f} ± {results['long']['mse_std']:.4f}, "
                f"MAE: {results['long']['mae_avg']:.4f} ± {results['long']['mae_std']:.4f}\n")

    print(f"评估结果已保存至 {result_file}")
    return results


def plot_prediction(pred, true, title, filename, length=90):
    plt.figure(figsize=(14, 6))
    plt.plot(pred[:length], label='预测值')
    plt.plot(true[:length], label='真实值')
    plt.title(title)
    plt.xlabel('时间步')
    plt.ylabel('总有功功率 (kW)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(result_dir, filename))
    plt.close()

if __name__ == '__main__':
    # 运行评估
    results = run_experiments()

    #绘图
    last_model = load_model(os.path.join(model_dir, 'lstm_short_term_round_4.keras'), compile=False)
    last_model.compile(optimizer='adam', loss='mse')

    pred_short = last_model.predict(testX_short).flatten()
    pred_short_denorm = denormalize(pred_short)
    testY_short_denorm = denormalize(testY_short)

    pred_long = last_model.predict(testX_long).flatten()
    pred_long_denorm = denormalize(pred_long)
    testY_long_denorm = denormalize(testY_long)

    plot_prediction(pred_short_denorm, testY_short_denorm,
                    '短期预测（未来90天）- 基于原始单位',
                    'short_term_prediction_real.png')

    plot_prediction(pred_long_denorm, testY_long_denorm,
                    '长期预测（未来365天）- 基于原始单位',
                    'long_term_prediction_real.png',
                    length=365)

    print("可视化结果已生成并保存至 results/ 目录")