import numpy as np
from keras.callbacks import EarlyStopping    # 과적합을 막기 위한 함수
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
# Import mlcompute module to use the optional set_mlc_device API for device selection with ML Compute.
from tensorflow.python.compiler.mlcompute import mlcompute

# Select GPU device.
mlcompute.set_mlc_device(device_name='gpu') # Available options are 'cpu', 'gpu', and ‘any'.

X_train, X_test, Y_train, Y_test = np.load(
    '../data/book_data_max_6125_size_218318.npy',
    allow_pickle=True)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

model = Sequential()
model.add(Embedding(218318, 100, input_length=6125))
model.add(Conv1D(32, kernel_size=5,
            padding='same', activation='relu'))
model.add(MaxPool1D(pool_size=1))
model.add(LSTM(128, activation='tanh',
               return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(64, activation='tanh',
               return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(64, activation='tanh'))
model.add(Dropout(0.1))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(12, activation='softmax'))
print(model.summary())

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy']
              )
fit_hist = model.fit(X_train, Y_train,
                     batch_size=600,
                     epochs=5,
                     validation_data=(X_test, Y_test)
                     )

score = model.evaluate(X_test, Y_test)
print(score[1])

model.save('../models/book_classfication_{}.h5'.format(score[1]))