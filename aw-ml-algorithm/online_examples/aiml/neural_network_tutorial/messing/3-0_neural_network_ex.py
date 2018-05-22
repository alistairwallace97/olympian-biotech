#---------------------------------------------------------
# 3.0 A Neural Network Example
# http://adventuresinmachinelearning.com/python-tensorflow-tutorial/

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
#NOTE_: one_hot=True specifies the labels associated with
#       each image, eg "4" = [0,0,0,0,1,0,0,0,0,0]
mnist = input_data.read_data_sets("MNIST_data/",\
                                    one_hot=True)
# The mnist data set is a set of 28x28 greyscale images 
# which represent hand-written digits. It has 55,000 
# training rows, 10,000 testing rows and 5,000 validation
# rows.

#---------------------------------------------------------
# 3.1 Setting things up
# Python optimisation variables 
learning_rate = 0.5
epochs = 10
batch_size = 100

# declare the training data placeholders 
# input x - for 28 x 28 pixels = 784
x = tf.placeholder(tf.float32, [None, 784])
# now declare the output data placeholder - 10 digits
y = tf.placeholder(tf.float32, [None, 10])
#NOTE_: The input x is an n x 784 vector, where n is the
#       unspecified number of input samples we are 
#       inputting. 


# For an L layer neural network there are always L-1 
# number of weights/bias tensors. In this case 3 layer->2

# For each neuron the output is defined by
#Eq(1): output = sum(inputs*weights) + bias
# Therefore the number of biases is the number of outputs 
# for each neuron and the number weights is the number of
# inputs for that layer.

# now declare the weights connecting the input to the 
# hidden layer
W1 = tf.Variable(tf.random_normal([784, 300], stddev=0.03),\
                    name='W1')
b1 = tf.Variable(tf.random_normal([300]), name='b1')
# and weights connecting the hidden layer to the output
# layer
W2 = tf.Variable(tf.random_normal([300, 10], stddev=0.03),\
                    name='W2')
b2 = tf.Variable(tf.random_normal([10]), name='b2')

# calculate the output of the hidden layer 
# first line implements Eq(1) from above
hidden_out = tf.add(tf.matmul(x, W1), b1)
# second line implements a "rectified linear unit", 
# basically if x<0 -> y=0, x>0 -> y=x
hidden_out = tf.nn.relu(hidden_out)

# now calculate the hidden layer output - in this case
# let's use a softmax activated output layer
y_ = tf.nn.softmax(tf.add(tf.matmul(hidden_out, W2), b2))

# We need a loss function, in this case he's gone for the
# one on this link:
# http://adventuresinmachinelearning.com/python-tensorflow-tutorial/
# This is a cross entropy loss function, implemented below.
# This line limits the output_y to being between 1e-10 
# and 0.9999999, this ensures we never get a log(0) 
# operation which would produce NaN and so break the 
# training process.
y_clipped = tf.clip_by_value(y_, 1e-10, 0.9999999)
# Reduce_sum sums over an axis, but it is zero indexed
# so axis=1 means we sum over the second axis.
# Remember y and y_clipped are m x 10 tensors
# Reduce_mean just takes the mean of whatever vector you
# give it.
cross_entropy = -tf.reduce_mean(tf.reduce_sum(\
                    y*tf.log(y_clipped)\
                    + (1-y)*tf.log(1-y_clipped), axis=1))


# add an optimiser
optimiser = tf.train.GradientDescentOptimizer(\
                learning_rate=learning_rate)\
                .minimize(cross_entropy)

# finally set up the initialisation operator
init_op = tf.global_variables_initializer()

# define an accuracy assessment operation 
# tf.argmax returns the 
correct_prediction = tf.equal(tf.argmax(y,1),\
                                tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,\
                                    tf.float32))


#---------------------------------------------------------
# 3.2 Setting up the training

# start the session
with tf.Session() as sess:
    # initialise the variables
    sess.run(init_op)
    # The number of batches we are going to loop through
    # for each epoch
    total_batch = int(len(mnist.train.labels)/batch_size)
    for epoch in range(epochs):
        # initialising a variable for the average error
        # of all the epochs, where our error function is
        # the cross entropy loss of each epoch.
        avg_cost = 0
        for i in range(total_batch):
            # selects a randomised batch of sample pairs
            # from the MNIST training data set.
            # the next_batch function is specific to the
            # MNIST dataset
            batch_x, batch_y = mnist.train.next_batch(\
                                    batch_size=batch_size)
            #NOTE_: sess.run can take two operations as 
            #       it's inputs. (optimiser and cross_...)
            _, c = sess.run([optimiser,\
                            cross_entropy],\
                            feed_dict={x: batch_x,\
                                        y: batch_y})
            # use c to calculate the average cost per epoch
            avg_cost += c/total_batch
        print("Epoch:", (epoch +1), "cost =", "{:.3f}".format(avg_cost))
    print("Training Complete\nTest Error:\n", sess.run(accuracy, feed_dict={x: mnist.test.images, y:mnist.test.labels}))

