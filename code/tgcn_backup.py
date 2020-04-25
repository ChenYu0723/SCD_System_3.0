# -*- coding: utf-8 -*-
# @Time    : 2019/12/5 21:48
# @Author  : Chen Yu

import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell, LSTMCell, BasicRNNCell
from utils import calculate_laplacian


class TGCN_Cell(RNNCell):
# class TGCN_Cell(LSTMCell):
    def __init__(self, adj, num_units, num_nodes, input_size=None,
                 act=tf.nn.tanh, reuse=None, norm=False):
        super(TGCN_Cell, self).__init__(_reuse=reuse)
        self._adj = []
        self._adj.append(calculate_laplacian(adj))
        self._units = num_units
        self._nodes = num_nodes
        self._act = act
        self._norm = norm

    @property
    def state_size(self):
        return self._nodes * self._units

    @property
    def output_size(self):
        return self._units

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or "tgcn"):
            with tf.variable_scope("gates"):
                value = tf.nn.sigmoid(
                    self._gc(inputs, state, 2 * self._units, bias=1.0, scope=scope))
                r, u = tf.split(value=value, num_or_size_splits=2, axis=1)
            with tf.variable_scope("candidate"):
                r_state = r * state
                c = self._act(self._gc(inputs, r_state, self._units, scope=scope))
            new_h = u * state + (1 - u) * c
        return new_h, new_h

    def _gc(self, inputs, state, output_size, bias=.0, scope=None):
        # ==== inputs:(-1, num_nodes)
        inputs = tf.expand_dims(inputs, 2)
        # ==== state:(batch,num_node,gru_units)
        state = tf.reshape(state, (-1, self._nodes, self._units))
        # ==== concat
        x_s = tf.concat([inputs, state], axis=2)
        input_size = x_s.get_shape()[2].value
        # ==== (num_node,input_size,-1)
        x0 = tf.transpose(x_s, perm=[1, 2, 0])
        x0 = tf.reshape(x0, shape=[self._nodes, -1])

        scope = tf.get_variable_scope()
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            for m in self._adj:
                x1 = tf.sparse_tensor_dense_matmul(m, x0)
            x = tf.reshape(x1, shape=[self._nodes, input_size, -1])
            x = tf.transpose(x, perm=[2, 0, 1])
            x = tf.reshape(x, shape=[-1, input_size])
            weights = tf.get_variable(
                'weights', [input_size, output_size], initializer=tf.contrib.layers.xavier_initializer()
            )
            x = tf.matmul(x, weights)  # (batch_size * self._nodes, output_size)
            biases = tf.get_variable(
                'biases', [output_size], initializer=tf.constant_initializer(bias, dtype=tf.float32)
            )
            x = tf.nn.bias_add(x, biases)
            # ==== normlization
            if self._norm:
                fc_mean = tf.Variable(initial_value=1)
                fc_var = tf.Variable(initial_value=1)
                fc_mean, fc_var = tf.nn.moments(x, axes=[0])
                scale = tf.Variable(tf.ones([output_size]))
                shift = tf.Variable(tf.zeros([output_size]))
                epsilon = 0.001

                # apply moving average for mean and var when train on batch
                # ema = tf.train.ExponentialMovingAverage(decay=0.5)
                #
                # def mean_var_with_update():
                #     ema_apply_op = ema.apply([fc_mean, fc_var])
                #     with tf.control_dependencies([ema_apply_op]):
                #         return tf.identity(fc_mean), tf.identity(fc_var)

                # mean, var = mean_var_with_update()
                mean, var = fc_mean, fc_var

                x = tf.nn.batch_normalization(x, mean, var, shift, scale, epsilon)

            x = tf.reshape(x, shape=[-1, self._nodes, output_size])
            x = tf.reshape(x, shape=[-1, self._nodes * output_size])
        return x


def my_tgcn(_X, _weights, _biases, adj, od, NUM_UNITS, num_nodes, PRE_LEN):
    cell_1 = TGCN_Cell(adj, NUM_UNITS, num_nodes)  # GCN-adj
    # cell_2 = TGCN_Cell(od, NUM_UNITS, num_nodes)  # GCN-od
    print(cell_1)
    cell = tf.nn.rnn_cell.MultiRNNCell([cell_1], state_is_tuple=True)
    _X = tf.unstack(_X, axis=1)
    # print('!!!!!')
    # print(cell)
    # print(_X.shape)
    outputs, states = tf.nn.static_rnn(cell, _X, dtype=tf.float32)  # GRU
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
