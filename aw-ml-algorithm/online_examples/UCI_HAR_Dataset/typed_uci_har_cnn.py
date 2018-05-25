# Human Activity Dataset CNN example
#https://burakhimmetoglu.com/2017/08/22/time-series-classification-with-tensorflow/

#Imports
import tensorflow as tf
import numpy as np 
import os 
from utils.utilities import *
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt 
%matplotlib inline


#Hyperparameters
batch_size = 600
seq_len = 128
learning_rate = 0.0001
epochs = 1000

n_channels = 9
n_classes = 6

graph = tf.Graph()

# The labels will be once-hot encoded for the 6 possible
# classifications.
with graph.as_default():
    inputs_ = tf.placeholder(tf.float32, 
                                [None, seq_len, n_channels],
                                name='inputs')
    labels_ = tf.placeholder(tf.float32,
                                [None, n_classes],
                                name='labels')
    keep_prob_ = tf.placeholder(tf.float32, 
                                name='keep_prob')
    learning_rate_ = tf.placeholder(tf.float32,
                                    name='learning_rate')

#Making the convolution layers
# Unlike image processing with a CNN when we move across
# the image with a 2d window. As this is a sequence we
# move across it with a 1d "kernel".
# Note: strides=2, which is why the seq_len halves each
#       convolution and max pooling layer.
#       padding='SAME' and strides = 1 in the convolution
#       layer and so same number of points come out as in
with graph.as_default():
    # Eg: (batch_size, seq_len, n_channels)
    # (batch, 128, 9) -> (batch, 64, 18)
    conv1 = tf.layers.conv1d(inputs=inputs_,
                                filters=18,
                                kernel_size=2,
                                strides=1,
                                padding='SAME',
                                activation=tf.nn.relu)
    max_pool_1 = tf.layers.max_pooling1d(inputs=conv1,
                                            pool_size=2,
                                            strides=2,
                                            padding='SAME')
    # (batch, 64, 18) -> (batch, 32, 36)
    conv2 = tf.layers.conv1d(inputs=max_pool_1,
                                filters=36,
                                kernel_size=2,
                                strides=1,
                                padding='SAME',
                                activation=tf.nn.relu)
    max_pool_2 = tf.layers.max_pooling1d(inputs=conv2,
                                            pool_size=2,
                                            strides=2,
                                            padding='SAME')

    # (batch, 32, 36) -> (batch, 16, 72)
    conv3 = tf.layers.conv1d(inputs=max_pool_2,
                                filters = 72,
                                kernel_size=2,
                                strides=1,
                                padding='SAME',
                                activation=tf.nn.relu)
    max_pool_3 = tf.layers.max_pooling1d(inputs=conv3,
                                            pool_size=2,
                                            strides=2,
                                            padding='SAME')

    # (batch, 16, 72) -> (batch, 8, 144)
    conv4 = tf.layers.conv1d(inputs=max_pool_3,
                                filters=144,
                                kernel_size=2,
                                strides=1,
                                padding='SAME',
                                activation=tf.nn.relu)
    max_pool_4 = tf.layers.max_pooling1d(inputs=conv4,
                                            pool_size=2,
                                            strides=2,
                                            padding='SAME')

# Adding a fully connected layer for classification, must
# flatten from 144 x 8 layers in one layer of (8 x 144)
with graph.as_default():
    # flattena nd add dropout
    flat = tf.reshape(max_pool_4, (-1,8*144))
    flat = tf.nn.drouput(flat, keep_prob=keep_prob_)

    # predictions
    logits = tf,layers.dense(flat, n_classes)

    # cost function and optimiser
    cost = tf.reduce_mean(
                tf.nn.softmas_cross_entropy_with_logits(
                    logits=logits,
                    labels=labels_))
    optimiser = tf.train.AdamOptimser(learning_rate_).minimise(cost)

    # accuracy
    correct_pred = tf.equal(
                        tf.argmax(logits, 1),
                        tf.argmas(labels_, 1))
    accuracy = tf.reduce_mean(
                        tf.cast(correct_pred,
                                tf.float32),
                        name='accuracy')