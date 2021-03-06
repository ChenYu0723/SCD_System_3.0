# -*- coding: utf-8 -*-
# @Time    : 2019/12/5 19:01
# @Author  : Chen Yu

import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from code.utils.input_data import load_data, preprocess_data
from code.tgcn import TGCN_Cell
from code.gcn import GCN
from code.gru import GRUCell
from code.utils.utils import evaluation
from datetime import datetime

os.chdir('..')


def my_nn(model, _X, _weights, _biases, adj, od, NUM_UNITS, num_nodes, PRE_LEN):
    if model == 'tgcn':
        cell_1 = TGCN_Cell(adj, od, NUM_UNITS, num_nodes)
    elif model == 'gcn':
        cell_1 = GCN(NUM_UNITS, adj, _X, 1)
    elif model == 'gru':
        cell_1 = GRUCell(NUM_UNITS, num_nodes)

    cell = tf.nn.rnn_cell.MultiRNNCell([cell_1], state_is_tuple=True)
    _X = tf.unstack(_X, axis=1)  # ==> (32, 9*322)
    outputs, states = tf.nn.static_rnn(cell, _X, dtype=tf.float32)
    m = []
    for i in outputs:
        o = tf.reshape(i, shape=[-1, num_nodes, NUM_UNITS])
        o = tf.reshape(o, shape=[-1, NUM_UNITS])
        m.append(o)
    last_output = m[-1]
    output = tf.matmul(last_output, _weights['out']) + _biases['out']
    output = tf.reshape(output, shape=[-1, num_nodes, PRE_LEN])
    output = tf.transpose(output, perm=[0, 2, 1])
    output = tf.reshape(output, shape=[-1, num_nodes])
    return output, m, states


if __name__ == '__main__':
    start_time = datetime.now()

    # ==== Hyper Parameters
    EPOCH = 3
    SEQ_LEN = 9 * 2
    PRE_LEN = 1 * 2
    NUM_UNITS = 64
    TRAIN_RATE = .8
    LR = .01
    BATCH_SIZE = 32
    MODEL = 'tgcn'  # tgcn gcn gru

    # ==== read data
    inFlow_data, outFlow_data, adj, od = load_data()

    time_len = inFlow_data.shape[0]
    num_nodes = inFlow_data.shape[1]

    # ==== normalization
    mms = MinMaxScaler()
    inFlow_data = mms.fit_transform(inFlow_data)
    outFlow_data = mms.fit_transform(outFlow_data)

    X_train_in, Y_train_in, X_test_in, Y_test_in = preprocess_data(inFlow_data, TRAIN_RATE, int(SEQ_LEN / 2), int(PRE_LEN / 2))
    X_train_out, Y_train_out, X_test_out, Y_test_out = preprocess_data(outFlow_data, TRAIN_RATE, int(SEQ_LEN / 2), int(PRE_LEN / 2))

    X_train = np.concatenate([X_train_in, X_train_out], axis=1)
    # print('X_train.shape:', X_train.shape)
    # X_train = X_train.reshape([-1, 2, 9, 322])
    Y_train = np.concatenate([Y_train_in, Y_train_out], axis=1)
    # Y_train = Y_train.reshape([-1, 2, 1, 322])
    X_test = np.concatenate([X_test_in, X_test_out], axis=1)
    # X_test = X_test.reshape([-1, 2, 9, 322])
    Y_test = np.concatenate([Y_test_in, Y_test_out], axis=1)
    # Y_test = Y_test.reshape([-1, 2, 1, 322])

    total_batch = int(X_train.shape[0]/BATCH_SIZE)
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

    # ==== optimizer
    lambda_loss = .0015
    Lreg = lambda_loss * sum(tf.nn.l2_loss(tf.cast(tf_var, dtype=tf.float32)) for tf_var in tf.trainable_variables())
    label = tf.reshape(labels, [-1, num_nodes])

    # ==== loss
    loss = tf.reduce_mean(tf.nn.l2_loss(Y_pred - label) + Lreg)

    # ==== rmse
    error = tf.sqrt(tf.reduce_mean(tf.square(Y_pred - label)))
    optimizer = tf.train.AdamOptimizer(LR).minimize(loss)

    # ==== initialize session
    variables = tf.global_variables()
    saver = tf.train.Saver(tf.global_variables())

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # ==== start
    print('start training...')
    batch_loss, batch_rmse = [], []
    test_loss, test_rmse_or, test_rmse, test_mae, test_mape, test_acc, test_r2, test_var, test_pred = [], [], [], [], [], [], [], [], []
    for epoch in range(EPOCH):
        if epoch / EPOCH > .66:
            LR = .001
        for m in range(total_batch):
            mini_batch = X_train[m * BATCH_SIZE:(m+1) * BATCH_SIZE]
            mini_label = Y_train[m * BATCH_SIZE:(m+1) * BATCH_SIZE]

            _, loss1, rmse1, train_output = sess.run([optimizer, loss, error, Y_pred],
                                                     feed_dict={inputs: mini_batch, labels: mini_label})
            batch_loss.append(loss1)

            train_label = np.reshape(mini_label, [-1, num_nodes])
            train_label = mms.inverse_transform(train_label)
            train_output = mms.inverse_transform(train_output)
            rmse_train_inver, _, _, _, _, _ = evaluation(train_label, train_output)
            batch_rmse.append(rmse_train_inver)

        # ==== Test completely at every epoch
        loss2, rmse2, test_out = sess.run([loss, error, Y_pred],
                                          feed_dict={inputs: X_test, labels: Y_test})
        test_label = np.reshape(Y_test, [-1, num_nodes])

        test_label = mms.inverse_transform(test_label)
        test_out = mms.inverse_transform(test_out)
        rmse, mae, mape, acc, r2_score, var_score = evaluation(test_label, test_out)

        test_loss.append(loss2)
        test_rmse_or.append(rmse2)
        test_rmse.append(rmse)
        test_mae.append(mae)
        test_mape.append(mape)
        test_acc.append(acc)
        test_r2.append(r2_score)
        test_var.append(var_score)
        test_pred.append(test_out)

        # ==== output
        print('Epoch:{}'.format(epoch),
              'train_rmse:{:.4}'.format(batch_rmse[-1]),
              'test_loss:{:.4}'.format(loss2),
              'test_rmse:{:.4}'.format(rmse),
              'test_acc:{:.4}'.format(acc),
              'test_mape:{:.4}'.format(mape))

    # ==== save model
    saver = tf.train.Saver()
    save_path = saver.save(sess, 'model/tgcn_net.ckpt')
    print(("Save to path: ", save_path))

    # ==== best score
    index = test_rmse.index(np.min(test_rmse))
    print('Best score epoch:', index)
    print('Best score:',
          'min_rmse:{:.4}'.format(test_rmse[index]),
          'min_mae:{:.4}'.format(test_mae[index]),
          'max_acc:{:.4}'.format(test_acc[index]),
          'min_mape:{:.4}'.format(test_mape[index]),
          'r2:{:.4}'.format(test_r2[index]),
          'var:{:.4}'.format(test_var[index]))

    # ==== all score
    b = int(len(batch_rmse) / total_batch)
    batch_rmse1 = [i for i in batch_rmse]
    train_rmse = [(sum(batch_rmse1[i * total_batch:(i + 1) * total_batch]) / total_batch) for i in range(b)]
    batch_loss1 = [i for i in batch_loss]
    train_loss = [(sum(batch_loss1[i * total_batch:(i + 1) * total_batch]) / total_batch) for i in range(b)]

    print('All score:')
    print(train_rmse)
    print(train_loss)
    print(test_rmse)
    print(test_acc)
    print(test_mae)
    print(test_mape)

    end_time = datetime.now()
    print(end_time - start_time)

