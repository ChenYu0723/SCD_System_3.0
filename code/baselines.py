# -*- coding: utf-8 -*-
# @Time    : 2019/12/9 15:19
# @Author  : Chen Yu

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
from code.utils.input_data import load_data, preprocess_data
from code.utils.utils import evaluation
import warnings
warnings.filterwarnings("ignore")


def my_svr(X_train, Y_train, X_test):
    reg = SVR()
    reg.fit(X_train, Y_train)
    Y_pred = reg.predict(X_test)
    return Y_pred


def my_rdf(X_train, Y_train, X_test):
    reg = RandomForestRegressor()
    reg.fit(X_train, Y_train)
    Y_pred = reg.predict(X_test)
    return Y_pred


def my_gbm(X_train, Y_train, X_test):
    reg = lgb.LGBMRegressor()
    reg.fit(X_train, Y_train)
    Y_pred = reg.predict(X_test)
    return Y_pred


if __name__ == '__main__':
    # ==== Hyper Parameters
    SEQ_LEN = 9
    PRE_LEN = 1
    TRAIN_RATE = .8

    # ==== read data
    data, adj = load_data('sh')

    time_len = data.shape[0]
    num_nodes = data.shape[1]

    # ==== predict every station
    all_rmse, all_mae, all_mape, all_acc, all_r2_score, all_var_score = [], [], [], [], [], []
    for node in range(num_nodes):
        if node % 10 == 0:
            print('dealed:%s' % node, 'all:%s' % num_nodes)
        station_data = data[:, node]
        # ==== normalization
        mms = MinMaxScaler()
        station_data = mms.fit_transform(station_data)

        X_train, Y_train, X_test, Y_test = preprocess_data(station_data, TRAIN_RATE, SEQ_LEN, PRE_LEN)
        X_train = X_train.reshape(-1, SEQ_LEN)
        X_test = X_test.reshape(-1, SEQ_LEN)
        Y_train = Y_train.reshape(-1)

        # ==== choose method
        # Y_pred = my_svr(X_train, Y_train, X_test)
        # Y_pred = my_rdf(X_train, Y_train, X_test)
        Y_pred = my_gbm(X_train, Y_train, X_test)

        Y_pred = Y_pred.reshape(-1, 1)
        Y_test = Y_test.reshape(-1, 1)

        Y_test = mms.inverse_transform(Y_test)
        Y_pred = mms.inverse_transform(Y_pred)

        rmse, mae, mape, acc, r2_score, var_score = evaluation(Y_test, Y_pred)
        all_rmse.append(rmse)
        all_mae.append(mae)
        all_mape.append(mape)
        all_acc.append(acc)
        all_r2_score.append(r2_score)
        all_var_score.append(var_score)

    print('All score:',
          'min_rmse:{:.4}'.format(np.mean(all_rmse)),
          'min_mae:{:.4}'.format(np.mean(all_mae)),
          'max_acc:{:.4}'.format(np.nanmean(all_acc)),
          'min_mape:{:.4}'.format(np.mean(all_mape)),
          'r2:{:.4}'.format(np.nanmean(all_r2_score)),
          'var:{:.4}'.format(np.nanmean(all_var_score)))
