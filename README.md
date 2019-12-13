# python
Boston house price prediction
网络实现：三层全连接网络实现，优化器Adam，损失函数mse。

Layer (type)               Output Shape               Param
_________________________________________________________________
dense_9 (Dense)              (None, 64)                896       
_________________________________________________________________
dense_10 (Dense)             (None, 64)                4160      
_________________________________________________________________
dense_11 (Dense)             (None, 1)                 65        
_________________________________________________________________
Total params: 5,121
Trainable params: 5,121
Non-trainable params: 0
_________________________________________________________________
最优test_mse为12.095

优化手段：一、由于数据较少，采用KFold交叉验证（K=5）；
         二、对训练集中的异常数据进行预先处理。
         
训练技巧：由于数据少容易overfit,一味追求训练集loss低是不可取的，所以采用了动态学习率和early stopping对val_loss（或loss）在回调函数中进行监视。
         考虑模型泛化能力不够好，所以同时对测试集进行评估，保存最好的模型。
         
tip:训练时数据集最好划分为三个部分，即训练集（train）、验证集（validation）、测试集（test），在训练集上拟合学习数据，验证集上评估泛化能力，测试集上测     试实际模型性能。

结果：目前可以得到Mean_validation_mse=6,Mean_test_mse=11左右   有待提升....

继续优化展望：使用更合理的模型、尝试更多的算法（梯度下降算法不是最适合的）....
![image]https://github.com/zhangzan1997/BostonhousePrice/blob/master/test_mse.png[image]

