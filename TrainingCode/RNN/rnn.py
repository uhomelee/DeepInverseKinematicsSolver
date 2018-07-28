""" Recurrent Neural Network.
A Recurrent Neural Network (LSTM) implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits (http://yann.lecun.com/exdb/mnist/)
Links:
    [Long Short Term Memory](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf)
    [MNIST Dataset](http://yann.lecun.com/exdb/mnist/).
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""

from __future__ import print_function

import datetime

import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
# Import MNIST data
#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

'''
To classify images using a recurrent neural network, we consider every image
row as a sequence of pixels. Because MNIST image shape is 28*28px, we will then
handle 28 sequences of 28 steps for every sample.
'''
X_train = np.loadtxt('train_x_5_10f.txt')
y_train = np.loadtxt('train_y_5_10f.txt')
X_test = np.loadtxt('test_x_5_10f.txt')
y_test = np.loadtxt('test_y_5_10f.txt')

# Training Parameters
learning_rate = 0.001
training_steps = 100000
batch_size = 128
display_step = 200

# Network Parameters
num_input = 3 # MNIST data input (img shape: 28*28) # todo=3
timesteps = 5 # timesteps #todo=5
num_hidden = 512 # hidden layer num of features
num_classes = 6 # # MNIST total classes (0-9 digits) # todo=6

# tf Graph input
X = tf.placeholder("float", [None, timesteps, num_input])
Y = tf.placeholder("float", [None, num_classes])
#keep_prob = tf.placeholder(tf.float32)
sigma = 1
weight_initializer = tf.variance_scaling_initializer(mode="fan_avg", distribution="uniform", scale=sigma)
bias_initializer = tf.zeros_initializer()
# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([num_hidden, num_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([num_classes]))
}

def RNN(x, weights, biases):

    #x = tf.transpose(x)
    x = tf.unstack(x, timesteps, 1)# todo


    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0, state_is_tuple=True)
    #lstm_cell = rnn.DropoutWrapper(cell=lstm_cell, input_keep_prob=1.0, output_keep_prob=keep_prob)
    #mlstm_cell = rnn.MultiRNNCell([lstm_cell() for _ in range(layer_num)], state_is_tuple=True)
    #init_state = mlstm_cell.zero_state(batch_size, dtype=tf.float32)
    # Get lstm cell output
    #outputs, state = tf.nn.dynamic_rnn(mlstm_cell, x, initial_state=init_state,time_major=False)
    #output=tf.reshape(outputs,[-1,num_hidden*timesteps])
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    #W_hidden = tf.Variable(weight_initializer([128,6]))
    #bias_hidden = tf.Variable(bias_initializer([6]))
    #layer = tf.nn.relu(tf.add(tf.matmul(outputs[-1], W_hidden), bias_hidden))
    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']
    #W_out = tf.Variable(weight_initializer([6, 6]))
    #bias_out = tf.Variable(bias_initializer([6]))
    #return tf.add(tf.matmul(layer, W_out), bias_out)


logits = RNN(X, weights, biases)
#prediction = tf.nn.softmax(out)
# Define loss and optimizer

#loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    #logits=logits, labels=Y))

loss_op = tf.reduce_mean(tf.squared_difference(logits, Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model (with test logits, for dropout to be disabled)
#correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
#accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)
    min_loss=1000
    saver = tf.train.Saver(max_to_keep=None)
    starttime = datetime.datetime.now()
    for step in range(1, training_steps+1):
        total_loss=0
        #batch_x, batch_y = mnist.train.next_batch(batch_size)
        shuffle_indices = np.random.permutation(np.arange(len(y_train)))
        X_train = X_train[shuffle_indices]
        y_train = y_train[shuffle_indices]
        for i in range(0, len(y_train) // batch_size):
            start = i * batch_size
            batch_x = X_train[start:start + batch_size]
            batch_y = y_train[start:start + batch_size]
            # Reshape data to get 28 seq of 28 elements
            batch_x = batch_x.reshape((batch_size, timesteps, num_input))
            # Run optimization op (backprop)
            sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
            if np.mod(i, 5) == 0:
                loss = sess.run(loss_op, feed_dict={X: batch_x, Y: batch_y})
                #print(mse_final)
                total_loss+=loss
        loss_test=0
        for i in range(0, len(y_test) // batch_size):
            start = i * batch_size
            batch_x = X_test[start:start + batch_size]
            batch_y = y_test[start:start + batch_size]
            # Reshape data to get 28 seq of 28 elements
            batch_x = batch_x.reshape((batch_size, timesteps, num_input))
            # Run optimization op (backprop)
            sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
            if np.mod(i, 5) == 0:
                loss = sess.run(loss_op, feed_dict={X: batch_x, Y: batch_y})
                #print(mse_final)
                loss_test+=loss
        print(str(step)+" loss:"+str(total_loss)+" test:"+str(loss_test))
        if total_loss < min_loss:
            with open("epoch_rnn_512.txt", 'a') as fopen:
                path_temp = "./model/model_final"
                save_path = saver.save(sess, path_temp)
                min_loss = total_loss
                string = str(step) + "|" + str(total_loss) + '\n'
                fopen.write(string)
        """
        print("Step " + str(step) +", Loss = "+str(total_loss) )
        with open("loss_rnn.txt", 'a') as f:
            endtime = datetime.datetime.now()
            runtime = endtime - starttime
            f.write(str(step) + " " + str(runtime)+ ' '+ str(total_loss) + '\n')
        """

