# Convolutional Neural Network example
#http://adventuresinmachinelearning.com/convolutional-neural-networks-tutorial-tensorflow/

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets(
                                    "MNIST_data/",
                                    one_hot=True)

# python optimisation variables
learning_rate = 0.0001
epochs = 10
batch_size = 50

# declare the training data placeholders
# input x - for 28 x 28 pixels = 784 - this is the 
#           flattened input data that is drawn from
#           mnist.train.nextbatch()
x = tf.placeholder(tf.float32, [None, 784])
# dynamically reshape the input
# We do this because the input is 28x28 (2d), but the built
# in functions such as conv2d() and max_pool() take 4d
# inputs.
# The format of the input data is [i, j, k, l]
#   i - the number of input samples
#   j - the height of the image
#   k - the width of the image
#   l - the number of channels, (for greyscale 1, for
#       for RGB would be 3)
# As i is unknown we can put -1 and it will dynamically
# adapt to the number of samples.
x_shaped = tf.reshape(x, [-1, 28, 28, 1])
# now declare the output data placeholder - 10 digits
# It is None by 10 because we are using one-hot encoding
# of the output.
y = tf.placeholder(tf.float32, [None, 10])


#A function defining a convolution layer
def create_new_conv_layer(
                    input_data, num_input_channels,
                    num_filters, filter_shape, 
                    pool_shape, name):
    # The filter shape is 5x5, the number of input 
    # channels is 1 as the image is greyscale, the block
    # diagram defines that the first convolution layer
    # will have a 32 channel outputs, the second 64.
    conv_filt_shape = [filter_shape[0], filter_shape[1],
                        num_input_channels, num_filters]

    # initialise weights and bias for the filter
    weights = tf.Variable(
                    tf.truncated_normal(
                                conv_filt_shape,
                                stddev=0.03),
                    name=name+'_W')
    bias = tf.Variable(
                tf.truncated_normal(
                        [num_filters]),
                name=name+'_b')

    # set up the convolutional layer operation
    # [1,1,1,1] is the strides input. The first and last
    # parameters are both 1 so that we don't move between
    # samples or channels. The padding determines the 
    # output size of each channel. If is it set to same
    # and the strides are set to one then it produces
    # the same size output as input.
    out_layer = tf.nn.conv2d(input_data, weights, 
                                [1, 1, 1, 1],
                                padding='SAME')
    # add the bias
    out_layer += bias
    # apply a ReLU non-linear activation
    out_layer = tf.nn.relu(out_layer)

    # perform max pooling
    # in this case we are performing a 2x2 max pooling
    # window so pool_shape is [2, 2]
    ksize = [1, pool_shape[0], pool_shape[1], 1]
    # the first and last elements will always be 1. The
    # other two are 2 as we want to do 2x down sampling.
    # This will half the input size (x, y) dimensions. 
    strides = [1, 2, 2, 1]
    out_layer = tf.nn.max_pool(out_layer, 
                                    ksize=ksize,
                                    strides=strides,
                                    padding='SAME')
    return out_layer


# create some convolutional layers
layer1 = create_new_conv_layer(x_shaped, 1, 32, [5, 5], 
                                [2, 2], name='layer1')
layer2 = create_new_conv_layer(layer1, 32, 64, [5, 5],
                                [2, 2], name='layer2')


#Make the fully connected layers
# Again we use -1 for the first dimension so it will be
# dynamically calculated
flattened = tf.reshape(layer2, [-1, 7*7*64])

# Set up some weights and bias values for this layer,
# then activate with ReLU
wd1 = tf.Variable(
            tf.truncated_normal(
                    [7*7*64, 1000],
                    stddev=0.03,),
            name='wd1')
bd1 = tf.Variable(
            tf.truncated_normal(
                    [1000],
                    stddev=0.01),
            name='bd1')
dense_layer1 = tf.matmul(flattened, wd1) + bd1
dense_layer1 = tf.nn.relu(dense_layer1)


wd2 = tf.Variable(
            tf.truncated_normal(
                        [1000, 10],
                        stddev=0.03),
            name='wd2')
bd2 = tf.Variable(
            tf.truncated_normal(
                        [10],
                        stddev=0.03),
            name='bd1')
dense_layer2 = tf.matmul(dense_layer1, wd2) + bd2
y_ = tf.nn.softmax(dense_layer2)


# define a cost/loss function
# Reduce_mean turns the vector of cross entropy of the 
# output and labels and reduces it to a scalar by taking 
# the mean 
cross_entropy = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(
                        logits=dense_layer2,
                        labels=y))

# add an optimiser 
optimiser = tf.train.AdamOptimizer(
                learning_rate=learning_rate).minimize(cross_entropy)

# define an acuracy assessment operation
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# set up the initialisation operator
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    # initialise the variables
    sess.run(init_op)
    total_batch = int(len(mnist.train.labels)/batch_size)
    for epoch in range(epochs):
        avg_cost = 0
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size=batch_size)
            _, c = sess.run([optimiser, cross_entropy],
                                feed_dict={x: batch_x, y: batch_y})
            avg_cost += c/total_batch
        test_acc = sess.run(accuracy,
                            feed_dict={x: mnist.test.images,
                                y: mnist.test.labels})
        print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost),
                    "test accuracy: {:.3f}".format(test_acc))
    print("\nTraining complete")
    print(sess.run(accuracy, feed_dict={
                                x: mnist.test.images,
                                y: mnist.test.labels}))

