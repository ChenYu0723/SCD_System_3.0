# -*- coding: utf-8 -*-
# @Time    : 2019/12/5 21:45
# @Author  : Chen Yu

import math
import numpy as np
import numpy.linalg as la
import scipy.sparse as sp
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from code.utils.input_data import load_data


def normalized_adj(adj):
    """
    将adj归一化
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)  # adj转为坐标对和值
    rowsum = np.array(adj.sum(1))  # adj每一行的和 ==> 度矩阵
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)  # 变为对角阵
    normalized_adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()  # adj*对角阵.转置*对角阵
    normalized_adj = normalized_adj.astype(np.float32)
    return normalized_adj


def sparse_to_tuple(mx):
    mx = mx.tocoo()
    coords = np.vstack((mx.row, mx.col)).transpose()  # 提取数值的坐标
    L = tf.SparseTensor(coords, mx.data, mx.shape)
    return tf.sparse_reorder(L)


def calculate_laplacian(adj, lambda_max=1):
    """
    计算adj的拉普拉斯算子并表示为稀疏张量
    :param adj:
    :param lambda_max:
    :return:
    """
    adj = normalized_adj(adj + sp.eye(adj.shape[0]))
    adj = sp.csr_matrix(adj)
    adj = adj.astype(np.float32)
    return sparse_to_tuple(adj)


def weight_variable_glorot(input_dim, output_dim, name=""):
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = tf.random_uniform([input_dim, output_dim], minval=-init_range,
                            maxval=init_range, dtype=tf.float32)

    return tf.Variable(initial, name=name)


def evaluation(y_true, y_pred):
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred))/(y_true + 1e-3)) * 100
    F_norm = la.norm(y_true - y_pred, 'fro') / (la.norm(y_true, 'fro') + 1e-3)
    r2 = 1 - ((y_true - y_pred) ** 2).sum() / ((y_true - y_true.mean()) ** 2).sum()
    var = 1 - (np.var(y_true - y_pred)) / np.var(y_true)
    return rmse, mae, mape, 1-F_norm, r2, var


def mape(y_pred, y_true):
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    m = np.mean(np.abs((y_true - y_pred)/y_true)) * 100
    return m


if __name__ == '__main__':
    # ==== test
    sess = tf.InteractiveSession()

    data, adj, od = load_data()
    # print(calculate_laplacian(adj).eval())
    print(calculate_laplacian(od).eval())
    # print(normalized_adj(adj))
    '''
    (317, 321)	0.0071363198
  (318, 321)	0.01757145
  (319, 321)	0.020485977
  (320, 321)	0.0064470195
    '''
    # a = [1,2,3]
    # b = sp.diags(a)  # 变为对角阵
    # print(b)
