# -*- coding: utf-8 -*-
# @Time    : 2019/12/5 19:17
# @Author  : Chen Yu

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

os.chdir('../..')
os.system('pwd')


def load_data(filter_hour=None):

    inFlow_path = "data/station_flow/station_inFlow.csv"
    outFlow_path = "data/station_flow/station_outFlow.csv"

    adj_path = "data/adj_matrix/station_adj.csv"
    # adj_path = "data/adj_matrix/station_adj_dis.csv"

    # od_path = "data/od_matrix/od_matrix_all.csv"
    # od_path = "data/od_matrix/od_matrix_all_nor.csv"
    od_path = "data/od_matrix/od_matrix_all_nor_0_1.csv"

    if filter_hour is not None:
        inFlow_path = "data/station_flow/station_flow_filtered/station_inFlow_%d.csv" % filter_hour
        od_path = "data/od_matrix/od_matrix_filtered/od_matrix_%d_nor.csv" % filter_hour

    inFlow_data = pd.read_csv(inFlow_path)
    inFlow_data = np.mat(inFlow_data, dtype=np.float32)

    outFlow_data = pd.read_csv(outFlow_path)
    outFlow_data = np.mat(outFlow_data, dtype=np.float32)

    station_adj = pd.read_csv(adj_path, header=None)
    station_adj = np.mat(station_adj, dtype=np.float32)

    station_od = pd.read_csv(od_path, header=None)
    station_od = np.mat(station_od, dtype=np.float32)
    return inFlow_data, outFlow_data, station_adj, station_od


def preprocess_data(data, time_len, train_rate, seq_len, pre_len):
    data_train, data_test = train_test_split(data, train_size=train_rate, shuffle=False)
    X_train, Y_train, X_test, Y_test = [], [], [], []
    for i in range(len(data_train) - seq_len - pre_len):
        X_train.append(data_train[i: i + seq_len])
        Y_train.append(data_train[i+seq_len: i + seq_len + pre_len])
    for i in range(len(data_test) - seq_len - pre_len):
        X_test.append(data_test[i: i + seq_len])
        Y_test.append(data_test[i+seq_len: i + seq_len + pre_len])

    return np.array(X_train), np.array(Y_train), np.array(X_test), np.array(Y_test)


if __name__ == '__main__':
    SEQ_LEN = 9
    PRE_LEN = 1
    TRAIN_RATE = .8

    inFlow_data, outFlow_data, adj, od = load_data()
    # print(adj.shape)
    # print(od.shape)
    time_len = inFlow_data.shape[0]

    X_train_in, Y_train_in, X_test_in, Y_test_in = preprocess_data(inFlow_data, time_len, TRAIN_RATE, SEQ_LEN, PRE_LEN)
    X_train_out, Y_train_out, X_test_out, Y_test_out = preprocess_data(outFlow_data, time_len, TRAIN_RATE, SEQ_LEN, PRE_LEN)
    # print(X_train)
    # print(X_train_in.shape)
    # print(X_train_out.shape)
    X_train = np.concatenate([X_train_in, X_train_out], axis=1)
    X_train = X_train.reshape([-1, 2, 9, 322])
    print(X_train.shape)

    # print(X_train_in[0])
    # print(X_train_out[0])
    # print(X_train[0])
    # print(X_train[0].shape)

    # print(Y_train_in.shape)
    # print(Y_train_out.shape)
    # Y_train = np.concatenate([Y_train_in, Y_train_out], axis=1)
    # Y_train = Y_train.reshape([-1, 2, 1, 322])
    # print(Y_train.shape)
