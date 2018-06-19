# Recurrent Neural Network example
#http://adventuresinmachinelearning.com/recurrent-neural-networks-lstm-tutorial-tensorflow/ 

import tensorflow as tf
import numpy as np 
import collections
import os 
import argparse
import datetime as dt 

data_path = "C:\\Users\\apsw\\ThirdYearGroupProjectGithubRepo\\olympian-biotech\\aw-ml-algorithm\\online_examples\\aiml\\recurrent_nn_tutorial\\data"

parser = argparse.ArgumentParser()
parser.add_argument('run_opt', type=int, default=1, help='An integer: 1 to train, 2 to test')
parser.add_argument('--data_path', type=str, default=data_path, help='The full path of the training data')
args = parser.parse_args()

def read_words(filename):
    with tf.gfile.GFile(filename, "r") as f:
        return f.read().decode("utf-8").replace("\n", "<eos>").split()


def build_vocab(filename):
    data = read_words(filename)

    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))

    return word_to_id

def file_to_word_ids(filename, word_to_id):
    data = read_words(filename)
    return [word_to_id[word] for word in data if word in word_to_id]


def load_data():
    # get the data paths
    train_path = os.path.join(data_path, "ptb.train.txt")
    valid_path = os.path.join(data_path, "ptb.valid.txt")
    test_path = os.path.join(data_path, "ptb.test.txt")

    # Build the complete vocabulary, then convert the 
    # test data to a list of integers
    word_to_id = build_vocab(train_path)
    train_data = file_to_word_ids(train_path, word_to_id)
    valid_data = file_to_word_ids(valid_path, word_to_id)
    test_data = file_to_word_ids(test_path, word_to_id)
    vocabulary = len(word_to_id)
    reversed_dictionary = dict(
                            zip(
                                word_to_id.values(),
                                word_to_id.keys()))

    print(train_data[:5])
    print(word_to_id)
    print(vocabulary)
    print(" ".join([reversed_dictionary[x] for x in train_data[:10]]))
    return train_data, valid_data, test_data, vocabulary, reversed_dictionary


def batch_producer(raw_data, batch_size, num_steps):
    raw_data = tf.convert_to_tensor(raw_data, 
                                    name="raw_data",
                                    dtype=tf.int32)
    data_len = tf.size(raw_data)
    batch_len = data_len // batch_size
    data = tf.reshape(raw_data[0: batch_size * batch_len],
                        [batch_size, batch_len])
    epoch_size = (batch_len - 1) // num_steps
    i = tf.train.range_input_producer(epoch_size, 
                                        shuffle=False).dequeue()

    x = data[:, i*num_steps: (i+1)*num_steps]
    x.set_shape([batch_size, num_steps])
    y = data[:, i*num_steps +1: (i+1)*num_steps + 1]
    y.set_shape([batch_size, num_steps])
    return x, y

# We pass this object the data and important info like
# batch_size and num_steps, it returns a batch of x 
# inputs and their associated y outputs.
class Input(object):
    def __init__(self, batch_size, num_steps, data):
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.epoch_size = ((len(data) // batch_size) -1\
                            // num_steps)
        self.input_data, self.targets = batch_producer(
                                                data,
                                                batch_size,
                                                num_steps)

# is_training is a boolean allowing us to both train the
# model and use it for testing/validation.
# This class also takes an Input object so can extract 
# it's member data. 
class Model(object):
    def __init__(self, input, is_training, hidden_size,
                    vocab_size, num_layers, dropout= 0.5,
                    init_scale=0.05):
        self.is_training = is_training
        self.input_obj = input
        self.batch_size = input.batch_size
        self.num_steps = input.num_steps 

with tf.device("/cpu:0")
    embedding = tf.Variable(
                    tf.random_uniform(
                        [vocab_size, self.hidden_size],
                        -init_scale,
                        init_scale))
    inputs = tf.nn.embedding_lookup(embedding,
                                    self.input_obj.input_data)

# Adds dropout regularisation which leaves out some 
# inputs every iteration which helps prevent overfitting
if is_training and dropout < 1:
    inputs = tf.nn.dropout(inputs, dropout)

# set up the state storage / extraction
self.init_state = tf.placeholder(tf.float32, 
                                [num_layers, 2, 
                                self.batch_size, 
                                self.hidden_size])

# Set up the state data in the format needed to feed it
# into the tensorflow LSTM built in functions
# Tensorflow's LSTM cell can accept the state as a tuple
# if a flag is set to True.
# tf.unstack creates a number of tensors, all of shape
# (2, batch_size, hidden_size) from the init_state tensor. 
# It makes one for each stacked LSTM layer (num_layer).
# These can then be loaded into LSTMStateTuple.
state_per_layer_list = tf.unstack(self.init_state, axis=0)
rnn_tuple_state = tuple(
                        [tf.contrub.rnn.LSTMStateTuple(
                            state_per_layer_list[idx][0],
                            state_per_layer_list[idx][0])]
                        for idx in range(num_layers))

# create an LSTM cell to be unrolled
cell = tf.contrub.rnn.LSTMCell(hidden_size, 
                                forget_bias=1.0)
# add a dropout wrapper if training 
if is_training and dropout < 1:
    cell = tf.contrib.rnn.DropoutWrapper(cell,
                                            output_keep_prob=dropout)
# as we have many layers of stacked LSTM cell in the 
# model we need to use another TensorFlow object called
# MultiRNNCell which performs the cell stacking.
# Note that we can tell MultiRNNCell to expect the state
# variable in the form of an LSTMStateTuple by setting 
# the flag state_is_tuple to True.
if num_layer > 1:
    cell = tf.contrib.rnn.MultiRNNCell([cell for _ in range(num_layers)], 
                                        state_is_tuple=True)

# Lastly we a dynamic RNN object in Tensorflow to 
# dynamically which will unroll the LSTM cell over each
# time step.
output, self.state = tf.nn.dynamic_rnn(cell, inputs, 
                                        dtype=tf.float32,
                                        initial_state=rnn_tuple_state)

#Adding the softmax, loss and optimiser operations
# reshape to (batch_size * num_steps, hidden_size)
output = tf.reshape(output, [-1, hidden_size])
# do the standard y = xw + b bit
softmax_w = tf.Variable(
                tf.random_uniform(
                    [hidden_size, vocab_size],
                    -init_scale,
                    init_scale))
softmax_b = tf.Variable(
                tf.random_uniform(
                    [vocab_size], -init_scale, init_scale))
logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b) 

# reshape logits to be a 3-D tensor for sequence loss
logits = tf.reshape(logits, [self.batch_size,
                                self.num_steps,
                                vocab_size])       
# use the contrib sequence loss and average over the 
# batches
# average_across_batch sums the cost across the batch
# dimension, whereas average_across_time sums the cost
# across the time dimension.
loss = tf.contrib.seq2seq.sequence_loss(
            logits,
            self.input_obj.targets,
            tf.ones([self.batch_size, self.num_steps],
                dtype=tf.float32),
            average_across_timesteps=False,
            average_across_batch=True)
 # update the cost
 self.const = tf.reduce_sum(loss)


# get the prediction accuracy
self.softmax_out = tf.nn.softmax(
                        tf.reshape(
                            logits, 
                            [-1, vocab_size]))
self.predict = tf.cast(
                    tf.argmax(
                        self.softmax_out, axis=1)
                    tf.int32)
correct_prediction = tf.equal(
                        self.predict,
                        tf.reshape(
                            self.input_obj.targets,
                            [-1]))
self.accuracy = tf.reduce_mean(
                    tf.cast(
                        correct_prediction,
                        tf.float32))

