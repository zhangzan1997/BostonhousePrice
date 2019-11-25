import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

#导入数据集 分离标签
train_data = pd.read_csv('train.csv')
train_y = train_data['MEDV']
train_x = train_data.drop('MEDV', axis = 1)
test_data = pd.read_csv('test.csv')
test_y = test_data['id']
test_x = test_data.drop('id', axis = 1) 
print(train_x.shape)
print(test_x.shape)

mean = train_x.mean(axis=0)
std = train_x.std(axis=0)
#print(mean)
#print(std)
train_x = (train_x - mean) / std
test_x = (test_x - mean) / std
#print(train_x)

def build_model():
    model = keras.Sequential([
        keras.layers.Dense(64,activation=tf.nn.relu,
                          input_shape=(train_x.shape[1],)),
        keras.layers.Dense(64,activation=tf.nn.relu),
        keras.layers.Dense(1)
    ])
    optimizer = tf.train.AdamOptimizer(0.001)
    model.compile(loss = 'mse',
                 optimizer = optimizer,
                 metrics = ['mae'])
    return model
model = build_model()
model.summary()
#model = build_model()
history = model.fit(train_x, train_y, epochs=1000, batch_size=4, verbose = 1, callbacks = None, validation_split = 0.1 , shuffle = True)
print(history.history.keys())
#test_mse_score, test_mae_score = model.evaluate(test_x, test_y) #评估模型
#print('mse = ',test_mse_score, 'mae = ', test_mae_score)

plt.subplot(1,2,1)
plt.plot(history.history['loss'], 'r')
plt.plot(history.history['val_loss'], 'b')
plt.title('model loss')
plt.xlabel('epoch')
plt.ylabel('mse loss')
#plt.show()

plt.subplot(1,2,2)
plt.plot(history.history['mean_absolute_error'],'r')
plt.plot(history.history['val_mean_absolute_error'],'b')
plt.title('model mae')
plt.xlabel('epoch')
plt.ylabel('mae')
#plt.savefig('Adam,0.001,1000,4.png')
plt.show()
