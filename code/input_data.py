# -*- coding: utf-8 -*-
# @Time    : 2019/12/5 19:17
# @Author  : Chen Yu

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

os.chdir('..')


def load_data(name='sh', filter_hour=None):
    if name == 'sh':
        flow_path = "data/station_inFlow.csv"

        adj_path = "data/station_adj.csv"
        # adj_path = "data/station_adj_dis.csv"

        # od_path = "data/od_matrix_all.csv"
        # od_path = "data/od_matrix_all_nor.csv"
        od_path = "data/od_matrix_all_nor_0_1.csv"
    elif name == 'sz':
        flow_path = "data/sz_speed.csv"
        adj_path = "data/sz_adj.csv"

    if filter_hour is not None:
        flow_path = "data/station_flow_filtered/station_inFlow_%d.csv" % filter_hour
        od_path = "data/od_matrix_filtered/od_matrix_%d_nor.csv" % filter_hour

    station_flow = pd.read_csv(flow_path)
    station_flow = np.mat(station_flow, dtype=np.float32)

    station_adj = pd.read_csv(adj_path, header=None)
    station_adj = np.mat(station_adj, dtype=np.float32)

    station_od = pd.read_csv(od_path, header=None)
    station_od = np.mat(station_od, dtype=np.float32)
    return station_flow, station_adj, station_od


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
    # ==== test
    data, adj, od = load_data()
    print(adj.shape)
    print(od.shape)
