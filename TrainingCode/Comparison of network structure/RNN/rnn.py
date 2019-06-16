
from __future__ import print_function

import datetime

import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

# Import MNIST data
X_train = np.loadtxt('train_x.txt')
y_train = np.loadtxt('train_y.txt')
X_test = np.loadtxt('test_x.txt')
y_test = np.loadtxt('test_y.txt')

# Training Parameters
learning_rate = 0.001
training_steps = 100000
batch_size = 128
display_step = 200

# Network Parameters
num_input = 3
timesteps = 5
num_hidden = 512
num_classes = 6

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
    x = tf.unstack(x, timesteps, 1)
    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0, state_is_tuple=True)
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    return tf.matmul(outputs[-1], weights['out']) + biases['out']


logits = RNN(X, weights, biases)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.squared_difference(logits, Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

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
            batch_x = batch_x.reshape((batch_size, timesteps, num_input))
            sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
            if np.mod(i, 5) == 0:
                loss = sess.run(loss_op, feed_dict={X: batch_x, Y: batch_y})
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
            with open("epoch.txt", 'a') as fopen:
                path_temp = "./model/model_final"
                save_path = saver.save(sess, path_temp)
                min_loss = total_loss
                string = str(step) + "|" + str(total_loss) + '\n'
                fopen.write(string)

