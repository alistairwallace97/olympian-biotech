#Tensorflow Test Python Script 
print("Initial test")
import tensorflow as tf

#Definitions for Code snippet 1
seq_len = 128



#Code snippet 1
graph = tf.Graph()

with graph.as_default():
    inputs_ = tf.placeholder(tf.float32, [None, seq_len, nchannels],
                            name = 'inputs')
    labels_ = tf.placeholder(tf.float32, [None,n_classes], name = 'labels')
    keep_prob_ = tf.placeholder(tf.float32, name = 'keep')
    learning_rate = tf.placeholder(tf.float32, name = 'learning_rate')
  