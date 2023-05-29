import pandas as pd
import math
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import randint


if __name__ == '__main__':
    
    train_CSV_FILE_PATH = 'D:\\comma2k19\\Chunk_01\\b0c9d2329ad1606b_2018-08-02--08-34-47.csv'
    test_CSV_FILE_PATH = 'D:\\comma2k19\\Chunk_01\\b0c9d2329ad1606b_2018-08-01--21-13-49.csv'
    train_df = pd.read_csv(train_CSV_FILE_PATH)
    test_df = pd.read_csv(test_CSV_FILE_PATH)
    train_values = train_df.to_numpy()
    train_times = train_values[:, -1]#所有行最后一列--时间 此处并没有用到
    train_distance = train_values[:, -2]#所有行的倒数第二列--距离
    test_values = test_df.to_numpy()
    test_times = test_values[:, -1]
    test_distance = test_values[:, -2]
    # 将输入特征归一化
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_X, train_y = scaler.fit_transform(train_values[:, :-2]), train_distance#提取can线数据，角度，加速度进行归一化 12000*3
    test_X, test_y = scaler.fit_transform(test_values[:, :-2]), test_distance
    # # 将四分之三作为训练集
    # train_len = len(times)
    # train = values[:train_len, :]
    # test = values[train_len:, :]
    # 划分输入（CAN_speed,steering_angel, acceleration_forward）输出（distance)
    # train_X, train_y = train, distance[:train_len]
    # test_X, test_y = test, distance[train_len:]
    # 将输入（X）改造为LSTM的输入格式，即[samples, timesteps, features]
    #x_train_2D = (x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2]))
    #x_test_2D = (x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2]))
    #train_X = train_X.reshape((train_X.shape[0], train_X.shape[1]))
    #test_X = test_X.reshape((test_X.shape[0], test_X.shape[1])) #二维不需要reshape
    forest_reg = RandomForestRegressor()
    # 让模型对训练集和结果进行拟合
    forest_reg.fit(train_X, train_y)
    print(forest_reg.score(train_X, train_y)) 
    result=forest_reg.score( test_X, test_y)
    print(result)
    print(forest_reg.feature_importances_)
    x_pre=forest_reg.predict(test_X)
    plt.plot(x_pre, label='train')
    plt.plot(test_y, label='test')
    plt.legend()
    plt.show()


   