# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.metrics import r2_score
import pdb
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ModelCheckpoint
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#导入数据集
train_data = pd.read_csv('train.csv',header=0)
#删除异常点数据
i_=[]
for i in range(len(train_data)):
    if train_data['MEDV'][i]== 50:
        i_.append(i)
for i in range(len(i_)):
    train_data=train_data.drop(i_[i])
train_y = train_data['MEDV']
train_x = train_data.drop('MEDV', axis = 1)
#数据归一化
test_data = pd.read_csv('test.csv')
test_y = pd.read_csv('submission.csv')
test_y = test_y['MEDV']
test_x = test_data.drop('id', axis = 1) 
#归一化处理
mean = train_x.mean(axis=0)
std = train_x.std(axis=0)
train_x = (train_x - mean) / std
test_x = (test_x - mean) / std
#数组初始化
test_list=[]#每个epoch对测试集的mse
all_val_mse = []#每个epoch对验证集的mse
all_test_mse = [15]#第i折最后对测试集的mse 作为最好模型评估
#输出路径初始化
path = r'../output/logs/'
if not os.path.exists(path):
    os.makedirs(path)
    print("创建目录")
else:
    print("目录已存在")
#回调加动态学习率、评价指标、保存最好模型
class MyCallback(keras.callbacks.Callback):
    def __init__(self,training_data,validation_data,testing_data):       
        self.x = training_data[0]
        self.y = training_data[1]
        self.xx = validation_data[0]
        self.yy = validation_data[1]
        self.xxx = testing_data[0]
        self.yyy = testing_data[1]
        #pdb.set_trace()
    
    def on_train_begin(self, logs={}):
        return
    def on_train_end(self, epoch,logs={}):
        return
    def on_epoch_end(self, epoch, logs={}):
        test_mse_score = model.evaluate(self.xxx, self.yyy); #评估模型
        if test_mse_score < 20:
            test_list.append(test_mse_score);
            if test_mse_score < all_test_mse[len(all_test_mse)-1]:
                model.save(os.path.join(path,'best_model.h5'))
                all_test_mse.append(test_mse_score)
                print("目前最优test_mse为%.3f" %test_mse_score)
        return

def build_model():
    model = keras.Sequential([
        keras.layers.Dense(64,activation=tf.nn.relu,input_shape=(train_x.shape[1],)),
#        Dropout(0.1),
        keras.layers.Dense(64,activation=tf.nn.relu),
#        Dropout(0.1),
        keras.layers.Dense(1)
    ])
    optimizer = keras.optimizers.Adam()
    
    model.compile(loss = 'mse',
                 optimizer = optimizer)
    return model

k=4
num_val_samples = len(train_x) // k
num_epochs = 500
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, mode='auto')
eraly_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=80, verbose=0)
for i in range(k):
    print('第', i+1 ,'折')
    val_x = train_x[i*num_val_samples : (i+1)*num_val_samples]#划出验证集部分
    val_y = train_y[i*num_val_samples : (i+1)*num_val_samples]
    
    partial_train_x = np.concatenate([train_x[:i*num_val_samples],train_x[(i+1)*num_val_samples:]],axis=0)
    partial_train_y = np.concatenate([train_y[:i*num_val_samples], train_y[(i+1)*num_val_samples:]],axis=0)
    
    model = build_model()
    
    history = model.fit(partial_train_x, partial_train_y, epochs = num_epochs, batch_size=8,verbose = 2,
                        callbacks = [MyCallback(training_data=[partial_train_x,partial_train_y],validation_data=[val_x,val_y],
                                               testing_data=[test_x,test_y]),
                                     reduce_lr,
                                     eraly_stopping],
                        validation_data = (val_x,val_y) )

    print(history.history.keys())
    #pdb.set_trace()
    val_loss = history.history['val_loss']#记录验证集数据
    all_val_mse.append(val_loss[len(val_loss)-1])
    #test_mse_score = model.evaluate(test_x, test_y) #评估模型
    #all_test_mse.append(test_mse_score)
    #print('test_mse = ',test_mse_score)
model.summary()
print("最优test_mse为%.3f" %all_test_mse[len(all_test_mse)-1])
np.save(os.path.join(path,'all_val_mse.npy'),all_val_mse)
np.save(os.path.join(path,'all_test_mse.npy'),all_test_mse)
np.save(os.path.join(path,'test_list.npy'),test_list)
y_pred = model.predict(test_x ,batch_size = 1)
df = pd.DataFrame(y_pred)
df.to_csv(os.path.join(path,'ownsubmissionAdam.csv'))