from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(look_back, 1)))
model.add(LSTM(50))
model.add(Dense(1))

opt = Adam(learning_rate=0.001)
model.compile(loss='mean_squared_error', optimizer=opt)
history = model.fit(trainX, trainY, epochs=20, batch_size=70, validation_data=(testX, testY), verbose=2)