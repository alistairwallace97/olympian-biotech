
# coding: utf-8

# # HAR CNN training 

# In[1]:


# Imports
import numpy as np
import os
from utils.utilities import *
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
plt.ion()


# ## Prepare data

# In[2]:


X_train, labels_train, list_ch_train = read_data(data_path="./data/", split="train") # train
X_test, labels_test, list_ch_test = read_data(data_path="./data/", split="test") # test


# In[3]:


# Normalize? 
X_train, X_test = standardize(X_train, X_test)


# Train/Validation Split

# In[4]:


X_tr, X_vld, lab_tr, lab_vld = train_test_split(X_train, labels_train, 
                                                stratify = labels_train, random_state = 183)


# One-hot encoding:

# In[5]:


y_tr = one_hot(lab_tr)
y_vld = one_hot(lab_vld)
y_test = one_hot(labels_test)


# In[6]:


# Imports
import tensorflow as tf


# ### Hyperparameters

# In[7]:


batch_size = 100        #525#600       # Batch size  
seq_len = 100          # Number of steps
learning_rate = 0.0001#0.0001
epochs = 1000

n_classes = 4
n_channels = 10


# ### Construct the graph
# Placeholders

# In[8]:


graph = tf.Graph()

# Construct placeholders
with graph.as_default():
    inputs_ = tf.placeholder(tf.float32, [None, seq_len, n_channels], name = 'inputs')
    labels_ = tf.placeholder(tf.float32, [None, n_classes], name = 'labels')
    keep_prob_ = tf.placeholder(tf.float32, name = 'keep')
    learning_rate_ = tf.placeholder(tf.float32, name = 'learning_rate')


# Build Convolutional Layers
# 
# Note: Should we use a different activation? Like tf.nn.tanh?,tf.nn.relu

# In[9]:


with graph.as_default():
    # (batch, 100, 10) --> (batch, 50, 20)
    conv1 = tf.layers.conv1d(inputs=inputs_, filters=20, kernel_size=2, strides=1, 
                             padding='same', activation = tf.sigmoid)#100,20
    max_pool_1 = tf.layers.max_pooling1d(inputs=conv1, pool_size=2, strides=2, padding='valid')#50,20
    
    # (batch, 50, 20) --> (batch, 24, 40)
    conv2 = tf.layers.conv1d(inputs=max_pool_1, filters=40, kernel_size=2, strides=1, 
                             padding='same', activation = tf.sigmoid)#50,40
    max_pool_2 = tf.layers.max_pooling1d(inputs=conv2, pool_size=2, strides=2, padding='valid')#24,40?
    
    # (batch, 24, 40) --> (batch, 12, 80)
    conv3 = tf.layers.conv1d(inputs=max_pool_2, filters=80, kernel_size=2, strides=1, 
                             padding='same', activation = tf.sigmoid)#24,80
    max_pool_3 = tf.layers.max_pooling1d(inputs=conv3, pool_size=2, strides=2, padding='valid')#12,80
    
    # (batch, 12, 80) --> (batch, 6, 160)
    conv4 = tf.layers.conv1d(inputs=max_pool_3, filters=160, kernel_size=2, strides=1, 
                             padding='same', activation = tf.sigmoid)#12,160
    max_pool_4 = tf.layers.max_pooling1d(inputs=conv4, pool_size=2, strides=2, padding='valid')#6,160


# Now, flatten and pass to the classifier

# In[10]:


with graph.as_default():
    # Flatten and add dropout
    flat = tf.reshape(max_pool_4, (-1,6*160))
    flat = tf.nn.dropout(flat, keep_prob=keep_prob_)
    
    # Predictions
    logits = tf.layers.dense(flat, n_classes)
    
    # Cost function and optimizer
   # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels_))
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels_))
    optimizer = tf.train.AdamOptimizer(learning_rate_).minimize(cost)
    
    # Accuracy
    correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(labels_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')


# ### Train the network

# In[11]:


# In[12]:


validation_acc = []
validation_loss = []

train_acc = []
train_loss = []

with graph.as_default():
    saver = tf.train.Saver()

with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())
    iteration = 1
   
    # Loop over epochs
    for e in range(epochs):
        
        # Loop over batches
        for x,y in get_batches(X_tr, y_tr, batch_size):
            
            # Feed dictionary
            feed = {inputs_ : x, labels_ : y, keep_prob_ : 0.5, learning_rate_ : learning_rate}
            
            # Loss
            loss, _ , acc = sess.run([cost, optimizer, accuracy], feed_dict = feed)
            train_acc.append(acc)
            train_loss.append(loss)
            
            # Print at each 5 iters
            if (iteration % 5 == 0):
                print("Epoch: {}/{}".format(e, epochs),
                      "Iteration: {:d}".format(iteration),
                      "Train loss: {:6f}".format(loss),
                      "Train acc: {:.6f}".format(acc))
            
            # Compute validation loss at every 10 iterations
            if (iteration%10 == 0):                
                val_acc_ = []
                val_loss_ = []
                
                for x_v, y_v in get_batches(X_vld, y_vld, batch_size):
                    # Feed
                    feed = {inputs_ : x_v, labels_ : y_v, keep_prob_ : 1.0}  
                    
                    # Loss
                    loss_v, acc_v = sess.run([cost, accuracy], feed_dict = feed)                    
                    val_acc_.append(acc_v)
                    val_loss_.append(loss_v)
                
                # Print info
                print("Epoch: {}/{}".format(e, epochs),
                      "Iteration: {:d}".format(iteration),
                      "Validation loss: {:6f}".format(np.mean(val_loss_)),
                      "Validation acc: {:.6f}".format(np.mean(val_acc_)))
                
                # Store
                validation_acc.append(np.mean(val_acc_))
                validation_loss.append(np.mean(val_loss_))
            
            # Iterate 
            iteration += 1
    
    saver.save(sess,"checkpoints-cnn/har.ckpt")


# In[13]:


# Plot training and test loss
t = np.arange(iteration-1)

fig1 = plt.figure(figsize = (6,6))
plt.plot(t, np.array(train_loss), 'r-', t[t % 10 == 0], np.array(validation_loss), 'b*')
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.legend(['train', 'validation'], loc='upper right')
plt.show()
fig1.savefig('loss_vs_iterations.pdf')


# In[14]:


# Plot Accuracies
fig2 = plt.figure(figsize = (6,6))
plt.plot(t, np.array(train_acc), 'r-', t[t % 10 == 0], validation_acc, 'b*')
plt.xlabel("Iteration")
plt.ylabel("Accuracy")
plt.legend(['train', 'validation'], loc='upper right')
plt.show()
fig2.savefig('acc_vs_iterations.pdf')


# ## Evaluate on test set

# In[15]:


test_acc = []

with tf.Session(graph=graph) as sess:
    # Restore
    saver.restore(sess, tf.train.latest_checkpoint('checkpoints-cnn'))

    test_counter = 0

    for x_t, y_t in get_batches(X_test[:-1], y_test[:-1], batch_size):
        feed = {inputs_: x_t,
                labels_: y_t,
                keep_prob_: 1}
        
        batch_acc = sess.run(accuracy, feed_dict=feed)
        test_counter = test_counter + 1
        test_acc.append(batch_acc)
    print("Test accuracy: {:.6f}".format(np.mean(test_acc)))

