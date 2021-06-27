from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential

model = Sequential()
model.add(Embedding(96815, 300, input_length=480))
model.add(Conv1D(32, kernel_size=5, padding='same', activation='relu'))
model.add(MaxPool1D(pool_size=1))
model.add(LSTM(128, activation='tanh', return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(64, activation='tanh', return_sequences=True))
model.add(Dropout(0.1))
model.add(LSTM(64, activation='tanh'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.05))
model.add(Dense(12, activation='softmax'))
print(model.summary())
print(model.summary(line_length=None,
                    positions=[.33, .60, .80, 1.]))
