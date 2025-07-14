import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

data_dir = 'data'
processed_dir = 'processed_data'
os.makedirs(processed_dir, exist_ok=True)

train_df = pd.read_csv(os.path.join(data_dir, 'Processed_train.csv'))
test_df = pd.read_csv(os.path.join(data_dir, 'Processed_test.csv'))

features = ['global_active_power', 'global_reactive_power', 'sub_metering_1',
            'sub_metering_2', 'sub_metering_3', 'sub_metering_remainder',
            'voltage', 'global_intensity', 'rr', 'nbjrr1', 'nbjrr5', 'nbjrr10', 'nbjbrou']

# 处理缺失值
train_df[features] = train_df[features].fillna(method='ffill')
test_df[features] = test_df[features].fillna(method='ffill')

# 归一化
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_train = scaler.fit_transform(train_df[features])
scaled_test = scaler.transform(test_df[features])

np.save(os.path.join(processed_dir, 'scaler_min.npy'), scaler.data_min_[0])  # global_active_power 最小值
np.save(os.path.join(processed_dir, 'scaler_max.npy'), scaler.data_max_[0])  # global_active_power 最大值

#构建 look_back 时间步的样本
def create_dataset(data, look_back=90):
    X, Y = [], []
    for i in range(len(data) - look_back - 1):
        X.append(data[i:(i + look_back), :])     # 所有特征
        Y.append(data[i + look_back, 0])         # 只预测 global_active_power
    return np.array(X), np.array(Y)

# 短期预测 (90天)
look_back_short = 90
trainX_short, trainY_short = create_dataset(scaled_train, look_back_short)
testX_short, testY_short = create_dataset(scaled_test, look_back_short)

# 长期预测 (365天)
look_back_long = 365
testX_long, testY_long = create_dataset(scaled_test, look_back_long)

np.save(os.path.join(processed_dir, 'trainX.npy'), trainX_short)
np.save(os.path.join(processed_dir, 'trainY.npy'), trainY_short)
np.save(os.path.join(processed_dir, 'testX_short.npy'), testX_short)
np.save(os.path.join(processed_dir, 'testY_short.npy'), testY_short)
np.save(os.path.join(processed_dir, 'testX_long.npy'), testX_long)
np.save(os.path.join(processed_dir, 'testY_long.npy'), testY_long)

print("数据预处理完成，并已保存至 processed_data/")