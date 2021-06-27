#!/usr/bin/env python
# coding: utf-8

# ## Model learning

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras import regularizers
# Import mlcompute module to use the optional set_mlc_device API for device selection with ML Compute.
from tensorflow.python.compiler.mlcompute import mlcompute


# In[2]:


# Select GPU device.
mlcompute.set_mlc_device(device_name='any') # Available options are 'cpu', 'gpu', and ‘any'.


# In[3]:


X_train, X_test, Y_train, Y_test = np.load('../data/book_data_max_179_wordsize_123913.npy', allow_pickle=True)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


# In[4]:


model_LSTM_dict = {
    'E':100, 
    'C':[16,5], 
    'M':1, 
    'L1':128, 
    'DO1':0.5, 
#     'L2':64, 
#     'DO2':0.3, 
#     'L3':64, 
#     'DO3':0.5, 
}
model_Dense_dict = {
    'D1':256,
    'DO1':0.5,
    'D2':128,
    'DO2':0.5,
    'D3':64,
    'DO3':0.1,
    'D4':12
}

model = Sequential()
model.add(Embedding(123913, model_LSTM_dict['E'], input_length=179)) 
model.add(Conv1D(model_LSTM_dict['C'][0], kernel_size=model_LSTM_dict['C'][1], padding='same', activation='relu'))
model.add(MaxPool1D(pool_size=model_LSTM_dict['M']))
model.add(LSTM(model_LSTM_dict['L1'], activation='tanh'))#, return_sequences=True))  
model.add(Dropout(model_LSTM_dict['DO1']))
# model.add(LSTM(model_LSTM_dict['L2'], activation='tanh', return_sequences=True))  
# model.add(Dropout(model_LSTM_dict['DO2']))
# model.add(LSTM(model_LSTM_dict['L3'], activation='tanh')) 
# model.add(Dropout(model_LSTM_dict['DO3']))
model.add(Flatten())
model.add(Dense(model_Dense_dict['D1'], activation='relu'))
model.add(Dropout(model_Dense_dict['DO1']))
model.add(Dense(model_Dense_dict['D2'], activation='relu'))
model.add(Dropout(model_Dense_dict['DO2']))
model.add(Dense(model_Dense_dict['D3'], activation='relu'))
model.add(Dropout(model_Dense_dict['DO3']))
model.add(Dense(model_Dense_dict['D4'], activation='softmax'))
print(model.summary())


# In[5]:


# early_stopping = tf.keras.callbacks.EarlyStopping(moniter='val_accuracy', patience=5)
fit_dict = {
    'B':2048,
    'E':50
}
model.compile(loss='categorical_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])
fit_hist = model.fit(X_train, Y_train, 
                     batch_size=fit_dict['B'], 
                     epochs=fit_dict['E'], 
                     validation_data=(X_test, Y_test))


# In[6]:


name = []
for k,v in model_LSTM_dict.items():
    name.append(k)
    if str(type(v)) == "<class 'list'>":
        for n in v:
            name.append(str(n))
    else:
        name.append(str(v))

for k,v in model_Dense_dict.items():
    name.append(k)
    name.append(str(v))

for k,v in fit_dict.items():
    name.append(k)
    name.append(str(v))
    
file_name = '_'.join(name)
print(file_name)


# In[7]:


score = model.evaluate(X_test, Y_test)
print('Evaluation loss :', score[0])
print('Evaluation accuracy :', score[1])


# In[8]:


# 그래프
plt.figure(facecolor="white")
plt.plot(fit_hist.history['loss'], label='loss')
plt.plot(fit_hist.history['val_loss'], label='val_loss')
plt.legend()
plt.savefig('../result/loss_{}_{}.png'.format(score[0],file_name))
plt.show()


# In[9]:


plt.figure(facecolor="white")
plt.plot(fit_hist.history['accuracy'], label='accuracy')
plt.plot(fit_hist.history['val_accuracy'], label='val_accuracy')
plt.legend()
plt.savefig('../result/accuracy_{}_{}.png'.format(score[1],file_name))
plt.show()


# In[10]:


# 모델 저장
model.save('./model/{}_{}.h5'.format(score[1],file_name))


# In[ ]:




