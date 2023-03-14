import torch
from numpy import array
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
import math


# plt.rcParams['font.sans-serif'] = ['SimHei'] # 步骤一（替换sans-serif字体）
# plt.rcParams['axes.unicode_minus'] = False   # 步骤二（解决坐标轴负数的负号显示问题）

def read_data(path):
    """
    读取数据集
    :param path:文件路径
    :return:处理后数据
    """
    datasets_train = pd.read_csv(path)
    datasets_train = datasets_train.iloc[1:, 1:]
    # datasets_train = datasets_train.tail(2818).reset_index(drop=True)
    datasets = datasets_train.apply(pd.to_numeric, errors='ignore')
    n_train_hours = int(len(datasets) * 0.7)
    train = datasets.head(n_train_hours)
    test = datasets.tail(len(datasets) - n_train_hours)
    print(train.shape, test.shape, datasets.shape)

    x_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    train = x_scaler.fit_transform(train)
    test = x_scaler.fit_transform(test)

    return datasets,train,test,x_scaler

def data_processing(values,n_train,n_test,n_feature,TIME_STEP):
    """
    数据集划分训练和测试，标准化
    :param values:输入数据集，pa.array格式
    :param n_train:训练集样本数
    :param n_feature:输入特征数量
    :return:划分后的训练集和测试集
    """
    train = pd.DataFrame(values[:n_train*TIME_STEP, :])
    test = pd.DataFrame(values[n_train*TIME_STEP:n_train*TIME_STEP + n_test*TIME_STEP, :])
    print(train.shape, test.shape)
    train_X, train_y = np.array(train.iloc[:, :-n_feature]), np.array(train.iloc[:, -n_feature:])
    test_X, test_y = np.array(test.iloc[:, :-n_feature]), np.array(test.iloc[:, -n_feature:])
    x_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    y_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))

    train_X = x_scaler.fit_transform(train_X)
    train_y = y_scaler.fit_transform(train_y)
    test_X = x_scaler.fit_transform(test_X)
    test_y = y_scaler.fit_transform(test_y)
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
    # 转换成tensor,分成batch
    train_X = torch.tensor(train_X)
    train_X = train_X.reshape(n_train, TIME_STEP, n_feature)
    train_y = torch.tensor(train_y)
    train_y = train_y.reshape(n_train, TIME_STEP, n_feature)
    test_X = torch.tensor(test_X)
    test_X = test_X.reshape(n_test, TIME_STEP, n_feature)
    test_y = torch.tensor(test_y)
    test_y = test_y.reshape(n_test, TIME_STEP, n_feature)

    return train_X, train_y, test_X, test_y, x_scaler, y_scaler


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
    数据转换
    :param data:数据矩阵
    :param n_in:输入步长
    :param n_out:输出步长
    :return:划分后数据集
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

# 三维数据划分
def split_sequences(sequences, n_steps_in, n_steps_out):
    """
    :param sequences:输入数据
    :param n_steps_in:输入步长
    :param n_steps_out:预测步长
    :return:处理后的输出和输出
    """
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        # check if we are beyond the dataset
        if out_end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix:out_end_ix, :]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


def reg_metric(y_True, y_pred):
    """
    回归算法评估
    :param y_True:真实数据
    :param y_pred:预测数据
    :return:无
    """
    mae = mean_absolute_error(y_True, y_pred)
    mse = mean_squared_error(y_True, y_pred)
    rmse = math.sqrt(mean_squared_error(y_True, y_pred))
    r2 = r2_score(y_True, y_pred)
    mape = np.mean(np.abs((np.array(y_True) -np.array(y_pred)) / np.array(y_True))) /len(np.array(y_True))* 100
    # print("MAE:",mae)
    #     # print("MSE:",mse)
    #     # print("RMSE:",rmse)
    #     # print("R Square:",r2)
    # print("MAPE:",mape)
    result=pd.DataFrame([mae,mse,rmse,r2,mape])
    # result.columns=['MAE','MSE','RMSE','R_Square']
    return result



# # 绘制预测前后图形
# def plot_sensor(s,datasets,all_datasets,all_datasets_upper,all_datasets_lower,N=5000):
#     """
#     :param s:列名
#     :param datasets:训练数据
#     :param all_datasets:预测数据
#     :return:无
#     """
#     x = np.linspace(0,1,len(all_datasets))
#     x1=x[0:len(datasets)]
#     x2=x[len(datasets):len(all_datasets)]
#     plt.figure(figsize=(8,4))
#     plt.plot(x1,all_datasets[s].iloc[0:len(datasets)],'b', label='样本集')
#     plt.plot(x2,all_datasets[s].iloc[len(datasets):],'r', label='预测6小时数据')
#     plt.plot(x2, all_datasets_upper[s].iloc[len(datasets):], 'y', label='预测6小时数据-上限')
#     plt.plot(x2, all_datasets_lower[s].iloc[len(datasets):], 'c', label='预测6小时数据-下限')
#     # plt.ylim(-100, N)
#     plt.ylabel(s)
#     plt.xlabel('time')
#     plt.legend()
#     plt.show()
#     # plt.show(block=False)
#     # plt.pause(2)  # 显示秒数
#     # plt.close()

# 绘制预测前后图形 (values[:n_train*TIME_STEP, :])
# def plot_sensor_time(s,datasets,all_datasets,N=5000):
#     """
#     :param s:列名
#     :param datasets:训练数据
#     :param all_datasets:预测数据
#     :return:无
#     """
#     import matplotlib.dates as mdates
#     from dateutil import parser
#
#     # list(map(parser.parse, data_date_str))
#     x1 = list(map(parser.parse, datasets[['start_time']]))
#     x2=list(map(parser.parse, all_datasets[['start_time']].iloc[len(x1):]))
#     # print(x2)
#     plt.figure(figsize=(8,4))
#     plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('Y%%m%d %HH%MM'))  # 設置x軸主刻度顯示格式（日期）
#     plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())  # 設置x軸主刻度間距
#
#     plt.plot(x1,all_datasets[s].iloc[0:len(x1)],'b', label='样本集')
#     plt.plot(x2,all_datasets[s].iloc[len(x1):],'r', label='预测6小时数据')
#     plt.gcf().autofmt_xdate()  # 自动旋转日期标记
#     # plt.ylim(-100, N)
#     plt.ylabel(s)
#     plt.xlabel('time')
    # plt.show()
    # plt.show(block=False)
    # plt.pause(2)  # 显示秒数
    # plt.close()



 # 绘制测试集拟合图形
# def loss_plot(l):
#     plt.plot(l, 'r')
#     plt.xlabel('训练次数')
#     plt.ylabel('loss')
#     plt.title('损失函数下降曲线')
#     # plt.show()
#     plt.show(block=False)
#     plt.pause(2)  # 显示秒数
#     plt.close()
#
# # 绘制测试集拟合图形
# def plot_sensor_test(s,datasets,all_datasets,N=5000):
#     """
#     :param s:列名
#     :param datasets:原始数据
#     :param all_datasets:预测数据
#     :return:无
#     """
#     x = np.linspace(0,1,len(all_datasets))
#     x1=x[0:len(datasets)]
#     x2=x[0:len(all_datasets)]
#     plt.figure(figsize=(8,4))
#     plt.plot(x1,datasets[s].iloc[0:len(datasets)],'b', label='原始测试集')
#     plt.plot(x2,all_datasets[s].iloc[0:len(all_datasets):],'r', label='预测测试集')
#     # plt.ylim(-100, N)
#     plt.ylabel(s)
#     plt.xlabel('time')
#     plt.legend()
#     plt.show()
    #
    # plt.show(block=False)
    # plt.pause(2)  # 显示秒数
    # plt.close()


# 损失模型曲线
# def plot_loss(fit_history):
#     """
#     :param fit_history: 损失曲线
#     :return:无
#     """
#     plt.figure(figsize=(13,5))
#     plt.plot(range(1, len(fit_history.history['loss'])+1), fit_history.history['loss'], label='train')
#     plt.plot(range(1, len(fit_history.history['val_loss'])+1), fit_history.history['val_loss'], label='val')
#     plt.xlabel('epoch')
#     plt.ylabel('mse')
#     plt.legend()
#     plt.show()
    # plt.show(block=False)
    # plt.pause(2)  # 显示秒数
    # plt.close()

# 预测数据处理
def data_restore(test_y,x_scaler,s):
    """
    预测数据处理
    :param test_y:预测的数据集
    :param x_scaler:归一化函数
    :param s:列名
    :return:处理后数据集
    """
    y_test_orignal = []
    for i in range(test_y.shape[0]):
        tmp_test = test_y[i][0]
        y_test_orignal.append(tmp_test)
    y_test_orignal = pd.DataFrame(y_test_orignal)
    y_test_orignal.columns = s
    y_test_orignal = pd.DataFrame(x_scaler.inverse_transform(y_test_orignal))  # 反归一化
    y_test_orignal.columns = s
    return y_test_orignal