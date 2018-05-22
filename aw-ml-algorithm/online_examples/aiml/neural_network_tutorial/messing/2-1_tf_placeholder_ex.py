#2.1 - A simple example with placeholders
# http://adventuresinmachinelearning.com/python-tensorflow-tutorial/

import tensorflow as tf
import numpy as np

# create TensorFlow variables 
#NOTE_: None means that the vector can have any value in
#       that dimension. Ie [None, 1] is a n x 1 vector.
b = tf.placeholder(tf.float32, [None, 1], name='b')

# other initialisations from previous example
const = tf.constant(2.0, name="const")
c = tf.Variable(1.0, name="c")
d = tf.add(b, c, name="d")
e = tf.add(c, const, name="e")
a = tf.multiply(e, d, name="a")

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    # here we are using feed_dict to specify the previously
    # unspecified input vector.
    # Feed_dict makes each input a python dictionary, with
    # each key being the name of the placeholder that we 
    # are filling.
    a_out = sess.run(a, feed_dict={b: np.arange(0, 10)[:,np.newaxis]})
    print("Variable a is {}".format(a_out))
