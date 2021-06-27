import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
# Import mlcompute module to use the optional set_mlc_device API for device selection with ML Compute.
from tensorflow.python.compiler.mlcompute import mlcompute
from keras.utils.vis_utils import plot_model
from keras.callbacks import EarlyStopping

# Select GPU device.
mlcompute.set_mlc_device(device_name='gpu') # Available options are 'cpu', 'gpu', and ‘any'.

X_train, X_test, Y_train, Y_test = np.load(
    '../data/book_data_max_102_wordsize_68747.npy',
    allow_pickle=True)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

file_name = 'DM999_L1_DN2'
model = Sequential()
model.add(Embedding(57323, 999, input_length=104))
model.add(Conv1D(32, kernel_size=5, padding='same', activation='relu'))
model.add(MaxPool1D(pool_size=1))
# model.add(LSTM(128, activation='tanh', return_sequences=True))
# model.add(Dropout(0.5))
# model.add(LSTM(64, activation='tanh', return_sequences=True))
# model.add(Dropout(0.5))
model.add(LSTM(64, activation='tanh'))
model.add(Dropout(0.5))
model.add(Flatten())
# model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(12, activation='softmax'))
print(model.summary())

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy']
              )
file_name = file_name + '_B7000_E1500_P30'
es = EarlyStopping(monitor='val_loss', mode='min', patience=60)
fit_hist = model.fit(X_train, Y_train,
                     batch_size=7000,
                     epochs=1500,
                     validation_data=(X_test, Y_test),
                     callbacks=[es]
                     )

score = model.evaluate(X_test, Y_test)
print('Evaluation loss :', score[0])
print('Evaluation accuracy :', score[1])


plt.figure(figsize=(10,4), facecolor="white")
ax = plt.subplot(1,2,1)
ax.plot(fit_hist.history['loss'], label='loss')
ax.plot(fit_hist.history['val_loss'], label='val_loss')
ax.set_title('Model loss')
ax.set_ylabel('Loss')
ax.set_xlabel('Epoch')
ax.legend()
ax = plt.subplot(1,2,2)
ax.plot(fit_hist.history['accuracy'], label='accuracy')
ax.plot(fit_hist.history['val_accuracy'], label='val_accuracy')
ax.set_title('Model accuracy')
ax.set_ylabel('Accuracy')
ax.set_xlabel('Epoch')
ax.legend()
plt.savefig('/content/drive/MyDrive/result/accuracy_{}_{}.png'.format(score[1],file_name))
plt.show()

model.save('/content/drive/MyDrive/model/{}_{}.h5'.format(score[1],file_name))
plot_model(model, to_file='/content/drive/MyDrive/model/{}_{}.png'.format(score[1],file_name), show_shapes=True, show_layer_names=True)