from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import matplotlib.pyplot as plt

X_train = np.loadtxt('train_x.txt')
y_train = np.loadtxt('train_y.txt')
X_test = np.loadtxt('test_x.txt')
y_test = np.loadtxt('test_y.txt')

# Training Parameters
learning_rate = 0.001
training_steps = 10000
batch_size = 256
display_step = 200

# Network Parameters
num_input = 3
timesteps = 5
num_hidden = 512
num_classes = 6

# tf Graph input
X = tf.placeholder("float", [None, timesteps, num_input])
Y = tf.placeholder("float", [None, num_classes])
# keep_prob = tf.placeholder(tf.float32)
sigma = 1
weight_initializer = tf.variance_scaling_initializer(mode="fan_avg", distribution="uniform", scale=sigma)
bias_initializer = tf.zeros_initializer()
# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([num_hidden * timesteps, num_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([num_classes]))
}


def RNN(x, weights, biases):
    # multilayer
    def lstm_cell():
        lstm = tf.contrib.rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
        return lstm

    # todo=?
    layer_num = 3
    keep_prob = 1.0
    mlstm_cell = rnn.MultiRNNCell([lstm_cell() for _ in range(layer_num)], state_is_tuple=True)
    init_state = mlstm_cell.zero_state(batch_size, dtype=tf.float32)
    # Get lstm cell output
    outputs, state = tf.nn.dynamic_rnn(mlstm_cell, x, initial_state=init_state, time_major=False)
    output = tf.reshape(outputs, [-1, num_hidden * timesteps])
    return tf.matmul(output, weights['out']) + biases['out']


logits = RNN(X, weights, biases)
# prediction = tf.nn.softmax(out)
# Define loss and optimizer

# loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
# logits=logits, labels=Y))

loss_op = tf.reduce_mean(tf.squared_difference(logits, Y))
global_step = tf.Variable(0, trainable=False)
# start_learning_rate=0.001
# learning_rate=tf.train.exponential_decay(learning_rate=start_learning_rate,global_step=global_step,\
# decay_steps=80000,decay_rate=0.9,staircase=True)
# optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
# train_op = optimizer.minimize(loss_op,global_step=global_step)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model (with test logits, for dropout to be disabled)
# correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:
    # Run the initializer
    sess.run(init)
    min_loss = 10000
    saver = tf.train.Saver(max_to_keep=None)
    for step in range(1, training_steps + 1):
        total_loss = 0
        count = 0
        # batch_x, batch_y = mnist.train.next_batch(batch_size)
        shuffle_indices = np.random.permutation(np.arange(len(y_train)))
        X_train = X_train[shuffle_indices]
        y_train = y_train[shuffle_indices]
        for i in range(0, len(y_train) // batch_size):
            start = i * batch_size
            batch_x = X_train[start:start + batch_size]
            batch_y = y_train[start:start + batch_size]
            batch_x = batch_x.reshape((batch_size, timesteps, num_input))
            # Run optimization op (backprop)
            sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
            if np.mod(i, 5) == 0:
                loss = sess.run(loss_op, feed_dict={X: batch_x, Y: batch_y})
                # print(mse_final)
                total_loss += loss
                count += 1
        # test todo
        loss_test = 0
        count_test = 0
        for i in range(0, len(y_test) // batch_size):
            start = i * batch_size
            batch_x = X_test[start:start + batch_size]
            batch_y = y_test[start:start + batch_size]
            # Reshape data to get 28 seq of 28 elements
            batch_x = batch_x.reshape((batch_size, timesteps, num_input))
            loss = sess.run(loss_op, feed_dict={X: batch_x, Y: batch_y})
            loss_test += loss
            count_test += 1
        if total_loss < min_loss:
            with open("epoch_rnn.txt", 'a') as fopen:
                path_temp = "./model/model_final"
                save_path = saver.save(sess, path_temp)
                min_loss = total_loss
                string = str(step) + "|" + str(total_loss / count) + '\n'
                fopen.write(string)
        print("Step " + str(step) + " Loss_training = " + str(total_loss) + " Loss_test:" + str(loss_test))
        with open("loss_rnn.txt", 'a') as f:
            f.write(str(step) + " " + str(total_loss / count) + ' ' + str(loss_test / count_test) + '\n')
        if step > 3:
            draw_data = np.loadtxt('./loss_rnn.txt')
            plt.plot(draw_data[:, 1])
            plt.plot(draw_data[:, 2])
            plt.savefig('./rnn.jpg')
