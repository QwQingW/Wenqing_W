import numpy as np
import os
import tensorflow as tf
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# 设置路径
processed_dir = 'processed_data'
model_dir = 'models'
os.makedirs(model_dir, exist_ok=True)

# 加载数据
trainX = np.load(os.path.join(processed_dir, 'trainX.npy'))
trainY = np.load(os.path.join(processed_dir, 'trainY.npy'))

testX_short = np.load(os.path.join(processed_dir, 'testX_short.npy'))
testY_short = np.load(os.path.join(processed_dir, 'testY_short.npy'))
testX_long = np.load(os.path.join(processed_dir, 'testX_long.npy'))
testY_long = np.load(os.path.join(processed_dir, 'testY_long.npy'))

# 设置随机种子以确保可复现性
def set_seed(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)

# 构建 LSTM 模型
def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(64, input_shape=input_shape, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

# 回调函数
callbacks = [
    EarlyStopping(patience=15, restore_best_weights=True),
    ReduceLROnPlateau(factor=0.5, patience=5)
]

# 进行五轮训练
rounds = 5
short_model_list = []

print("开始五轮训练：")
for i in range(rounds):
    set_seed(i * 100)  # 每轮不同种子
    print(f"\n第 {i+1} 轮训练开始")
    model = build_model((trainX.shape[1], trainX.shape[2]))
    model.fit(trainX, trainY,
              validation_split=0.1,
              epochs=100,
              batch_size=32,
              callbacks=callbacks,
              verbose=1)
    model.save(os.path.join(model_dir, f'lstm_short_term_round_{i}.keras'))
    short_model_list.append(model)

print("五轮训练完成，模型已保存")