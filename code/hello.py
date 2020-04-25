# -*- coding: utf-8 -*-
# @Time    : 2019/12/6 19:56
# @Author  : Chen Yu

import os
import numpy as np
import pandas as pd
import tensorflow as tf
os.chdir('..')

# sess = tf.InteractiveSession()
#
# a = np.array([[1,2], [3,4]])
# b = np.array([[5,6], [7,8]])
# print(a.dot(b))
# print(np.dot(a, b))
# print(np.matmul(a, b))
# print(tf.tensordot(a, b, 1).eval())
# print(tf.matmul(a, b).eval())

# for i in range(5,-1,-1):
#     print(i)

# df = pd.read_csv('data/station_inFlow.csv').head()
# col = list(df.columns.astype(int))
# print(col)

a = np.array(
    [[[1,2], [3,4]], [[5,6], [7,8]]]
)
b = np.array(
    [[0], [1]]
)
print(a*b)