#2.0 - A simple example with constants and Variables
# http://adventuresinmachinelearning.com/python-tensorflow-tutorial/

import tensorflow as tf

# first, create a TensorFlow constant
const = tf.constant(2.0, name="const")

# create TensorFlow variables
b = tf.Variable(2.0, name='b')
c = tf.Variable(1.0, name='c')

# now create some operations
d = tf.add(b, c, name='d')
e = tf.add(c, const, name='e')
a = tf.multiply(d, e, name='a')

# setup the variable initialisation
#NOTE_: The variables and constants do not exist until
#       this step.
init_op = tf.global_variables_initializer()

# start the session
with tf.Session() as sess:
    # initialise the variables
    sess.run(init_op)
    # compute the output of the graph
    #NOTE_: a is an operation not a variable and so it can
    #       be run
    #NOTE_: We don't have to run d and e which are both
    #       operation needed to calculate a. Tensor flow
    #       works out everything required for a using the 
    #       data flow graph.
    a_out = sess.run(a)
    print("Variable a is {}".format(a_out))




