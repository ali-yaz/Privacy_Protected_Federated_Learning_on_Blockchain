#!/usr/bin/env python
"""
Mode Detection with Deep Neural Network
Implemented in Tensorflow Library (Version 1.6 installed with Anaconda on Windows 10)
The code read the data files from PostgreSQL database
Please find the 'points.csv' and 'labels.csv' on Github and import them into a PostgreSQL db, or
modify the code to read all the data from csv files directly.
"""
# ==============================================================================
__author__ = "Ali Yazdizadeh"
__date__ = "February 2018"
__email__ = "ali.yazdizadeh@mail.concordia.ca"
__python_version__ = "3.5.4"
# ==============================================================================

import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import time

start_time = time.time()
minibatch_size = 16
seg_size = 70
# Number of channels
num_channels = 5
# Number of classes
num_classes = 4
#Network architecture

num_channels_ensemble = [5]
num_filters_ensemble = []
filters_size_ensemble = []
num_stride_maxpool_ensemble = []
num_stride_conv2d_ensemble = []
maxpool_size_ensemble = []

num_layers_ensemble = [5]
num_networks = len(num_layers_ensemble)
filters_size_ensemble.append([8,8,8,8,8])

num_filters_ensemble.append([96,256,384,384,256])

maxpool_size_ensemble.append([8,8,8,8,8])
for i in range(len(num_layers_ensemble)):
    num_stride_conv2d_ensemble.append([2 for k in range(0, num_layers_ensemble[i])])

for i in range(len(num_layers_ensemble)):
    num_stride_maxpool_ensemble.append([2 for k in range(0, num_layers_ensemble[i])])

weights_ensemble = []
for i in range(len(filters_size_ensemble)):

    filters_size = filters_size_ensemble[i]
    num_filters = num_filters_ensemble[i]

    weights = []
    for index, f in enumerate(filters_size):
        if index == 0:
            weights.append([f, num_channels, num_filters[index]])
        else:
            weights.append([f, num_filters[index - 1], num_filters[index]])

    weights_ensemble.append(weights)

def parameters_weights():
    num_layers_ensemble = [5]

    filters_size_ensemble.append([8, 8, 8, 8, 8])

    num_filters_ensemble.append([96, 256, 384, 384, 256])

    maxpool_size_ensemble.append([8, 8, 8, 8, 8])
    for i in range(len(num_layers_ensemble)):
        num_stride_conv2d_ensemble.append([2 for k in range(0, num_layers_ensemble[i])])

    for i in range(len(num_layers_ensemble)):
        num_stride_maxpool_ensemble.append([2 for k in range(0, num_layers_ensemble[i])])


    weights_ensemble = []
    for i in range(len(filters_size_ensemble)):

        filters_size = filters_size_ensemble[i]
        num_filters = num_filters_ensemble[i]

        weights = []
        for index, f in enumerate(filters_size):
            if index == 0:
                weights.append([f, num_channels, num_filters[index]])
            else:
                weights.append([f, num_filters[index - 1], num_filters[index]])

        weights_ensemble.append(weights)

    return num_layers_ensemble, filters_size_ensemble, num_filters_ensemble, maxpool_size_ensemble, num_stride_conv2d_ensemble, num_stride_maxpool_ensemble, weights_ensemble


######################split data to train-test######################
def split_train_test(X_origin, Y_orig):
    X_train_orig, X_test_orig, Y_train_orig, Y_test_orig = train_test_split(X_origin, Y_orig, test_size=0.20,
                                                                            random_state=None)

    return (X_train_orig, X_test_orig, Y_train_orig, Y_test_orig)


######################Convert labes vector to one-hot######################
def convert_to_one_hot(Y, C):
    Y_onehot = np.zeros(Y.shape[0], dtype=[('uuid', 'S64'), ('trip_id', 'int8'), ('segment_id', 'int8'),
                                           ('class_label', '(4,)int8')])
    Y_onehot = np.rec.array(Y_onehot)
    Y_onehot.uuid = Y.uuid
    Y_onehot.trip_id = Y.trip_id
    Y_onehot.segment_id = Y.segment_id
    Y_onehot.class_label = np.eye(C)[Y.class_label.reshape(-1)]

    return Y_onehot

################initialize the parameters#################
def initialize_parameters(weights):
    """
    Initializes weight parameters to build a neural network with tensorflow. For example, the shapes could be:
                        W1 : [4, 4, 3, 8]
                        W2 : [2, 2, 8, 16]
                        W3 : ....
    Returns:
    parameters -- a dictionary of tensors containing W1, W2, W3 , ...
        """

    # define the parameters for conv layers
    parameters = {}

    for index, current_layer in enumerate(weights):
        # declare 'W's
        globals()['W{}'.format(index + 1)] = tf.get_variable('W{}'.format(index + 1),
                                                             current_layer,
                                                             initializer=tf.contrib.layers.xavier_initializer())

        parameters['W{}'.format(index + 1)] = globals()['W{}'.format(index + 1)]

    return parameters

####################Forward propagation in tensorflow#########################
def forward_propagation(X, parameters, num_stride_conv2d, maxpool_size, num_stride_maxpool):
    # Retrieve the parameters from the dictionary "parameters"
    for index, param in enumerate(parameters):
        # print(param)
        # print(index, 'index is')
        # print('num_stride_conv2d:',num_stride_conv2d[index])
        # print('num_stride_maxpool:', num_stride_maxpool[index])
        # Retrieve the parameters from the dictionary "parameters"
        if index == 0:
            globals()['W{}'.format(index + 1)] = parameters['W{}'.format(index + 1)]

            # CONV2D: stride from num_stride_conv2d, padding 'SAME'
            globals()['Z{}'.format(index + 1)] = tf.nn.conv1d(X, filters=globals()['W{}'.format(index + 1)]
                                                              , stride=num_stride_conv2d[index],
                                                              padding='SAME')

            # RELU
            globals()['A{}'.format(index + 1)] = tf.nn.leaky_relu(globals()['Z{}'.format(index + 1)], alpha=0.02)
            # tf.nn.relu(globals()['Z{}'.format(index + 1)])

            # filter = tf.get_variable('weights', [5, 5, 1, 64],
            #                          initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32),
            #                          dtype=tf.float32)

            # MAXPOOL: window size form stride from num_stride_maxpool, sride is the same size as window size, padding 'SAME'
            globals()['P{}'.format(index + 1)] = tf.layers.max_pooling1d(globals()['A{}'.format(index + 1)],
                                                                         pool_size=maxpool_size[index],
                                                                         strides=num_stride_maxpool[index],
                                                                         padding='SAME')
        else:
            globals()['W{}'.format(index + 1)] = parameters['W{}'.format(index + 1)]

            # CONV2D: stride from num_stride_conv2d, padding 'SAME'
            globals()['Z{}'.format(index + 1)] = tf.nn.conv1d(globals()['P{}'.format(index)],
                                                              filters=globals()['W{}'.format(index + 1)]
                                                              , stride=num_stride_conv2d[index], padding='SAME')

            # RELU
            globals()['A{}'.format(index + 1)] = tf.nn.leaky_relu(globals()['Z{}'.format(index + 1)], alpha=0.02)
            # tf.nn.relu(globals()['Z{}'.format(index + 1)])

            # MAXPOOL: window size form stride from num_stride_maxpool, sride is the same size as window size, padding 'SAME'
            globals()['P{}'.format(index + 1)] = tf.layers.max_pooling1d(globals()['A{}'.format(index + 1)],
                                                                         pool_size=maxpool_size[index],
                                                                         strides=num_stride_maxpool[index],
                                                                         padding='SAME')

    # FLATTEN
    globals()['P{}'.format(len(parameters))] = tf.contrib.layers.flatten(globals()['P{}'.format(len(parameters))])

    # one fully connected layer
    globals()['Z{}'.format(len(parameters) + 1)] = tf.contrib.layers.fully_connected(
        globals()['P{}'.format(len(parameters))], num_classes, activation_fn=None)

    for index, param in enumerate(parameters):
        print(globals()['Z{}'.format(index + 1)])

    print(globals()['Z{}'.format(len(parameters) + 1)])
    print(globals()['P{}'.format(len(parameters))])

    final_Z = globals()['Z{}'.format(len(parameters) + 1)]
    return final_Z


####################Computing Cost with softmax_cross_entropy in tensorflow#########################
def compute_cost(final_Z, Y, cl_weights):
    """
    class_weights
    Computes the cost

    Arguments:
    Final_Z -- output of forward propagation (output of the last LINEAR unit), of shape (6, number of examples)
    Y -- "true" labels vector placeholder, same shape as final_Z

    Returns:
    cost - Tensor of the cost function
    """
    # without weights
    # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=final_Z, labels=Y))
    # with weights
    cost = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=Y, logits=final_Z, weights=cl_weights))
    return cost


