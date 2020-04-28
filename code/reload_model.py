# -*- coding: utf-8 -*-
# File Name:  reload_model.py
# Date:       2020/2/12 12:50 下午
# Author:     yuchen

import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from code.utils.input_data import load_data, preprocess_data
from code.main import my_nn
from code.utils.utils import evaluation
from code.utils.visualization import plot_flow


# ==== Hyper Parameters
EPOCH = 500
SEQ_LEN = 9
PRE_LEN = 1
NUM_UNITS = 64
TRAIN_RATE = .8
LR = .01
BATCH_SIZE = 32
MODEL = 'tgcn'  # tgcn gcn gru

# ==== read data
data, adj, od = load_data('sh')

time_len = data.shape[0]
num_nodes = data.shape[1]

# ==== normalization
mms = MinMaxScaler()
data = mms.fit_transform(data)

X_train, Y_train, X_test, Y_test = preprocess_data(data, time_len, TRAIN_RATE, SEQ_LEN, PRE_LEN)
total_batch = int(X_train.shape[0] / BATCH_SIZE)
training_data_count = len(X_train)

# ==== placeholders
inputs = tf.placeholder(tf.float32, shape=[None, SEQ_LEN, num_nodes])
labels = tf.placeholder(tf.float32, shape=[None, PRE_LEN, num_nodes])

# ==== graph weights
weights = {
    'out': tf.Variable(tf.random_normal([NUM_UNITS, PRE_LEN], mean=1.0), name='weight_o')
}
biases = {
    'out': tf.Variable(tf.random_normal([PRE_LEN]), name='bias_o')
}

# ==== model
Y_pred, ttts, ttto = my_nn(MODEL, inputs, weights, biases, adj, od, NUM_UNITS, num_nodes, PRE_LEN)

# ==== reload model
saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, "model/adj_1000/tgcn_net.ckpt")
    # print("weights:", sess.run(weights))
    # print("biases:", sess.run(biases))
    test_out = sess.run(Y_pred, feed_dict={inputs: X_test})
    test_label = np.reshape(Y_test, [-1, num_nodes])

    test_label = mms.inverse_transform(test_label)
    test_out = mms.inverse_transform(test_out)

    rmse, mae, mape, acc, r2_score, var_score = evaluation(test_label, test_out)

# ==== score
print('Last score:',
      'rmse:{:.4}'.format(rmse),
      'mae:{:.4}'.format(mae),
      'acc:{:.4}'.format(acc),
      'mape:{:.4}'.format(mape),
      'r2:{:.4}'.format(r2_score),
      'var:{:.4}'.format(var_score))

# ==== visualization
path = '/output/flow_fig'
plot_flow(247, test_out, test_label, path)

'''
113：莲花路
247:陆家嘴
2010：世纪大道
2011：莘庄
2035：人民广场
2040：东方体育中心 注意修改test ts 从100开始
'''
