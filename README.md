# python
Boston house price prediction
实现：三层全连接网络实现，优化器RMS/Adam，损失函数mse，错误率mae。 训练/验证 9:1

Adam结果：![image](https://github.com/zhangzan1997/python/blob/master/Adam%2C0.001%2C1000%2C4.png)

RMS结果：![image](https://github.com/zhangzan1997/python/blob/master/Rms%2C0.001%2C1000%2C4.png)

结果训练Loss精度可以实现较好，但是验证集loss较高。在10-15反复跌宕。

优化提升：由于数据集较小，可以考虑用k折法增加数据强度。
