import tensorflow as tf
import numpy as np
import os
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

#X_test = np.loadtxt('xyz_0101.txt')
sess = tf.Session()

# Model architecture parameters
learning_rate = 0.001
training_steps = 10000
batch_size = 256
display_step = 200

# Network Parameters
num_input = 3 
timesteps = 5 
num_hidden = 2048 
num_classes = 6 

# tf Graph input
X = tf.placeholder("float", [None, timesteps, num_input])
Y = tf.placeholder("float", [None, num_classes])
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

    x = tf.unstack(x, timesteps, 1)# todo=0

    lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    return tf.matmul(outputs[-1], weights['out']) + biases['out']


logits = RNN(X, weights, biases)
loss_op = tf.reduce_mean(tf.squared_difference(logits, Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)
sess.run(tf.global_variables_initializer())


saver = tf.train.Saver()

#todo
saver.restore(sess, "../TrainingCode/model/model_final")

path='./positionData'
files= os.listdir(path)
for file in files:
    path='./positionData/'+file
    X_test=np.loadtxt(path)
    resultpath='./angleData/'+file
    print(X_test.shape)
    length=X_test.shape[0]
    X_test = X_test.reshape((length, timesteps, num_input))
    result = sess.run(logits, feed_dict={X: X_test})
    print(result.shape)
    np.set_printoptions(suppress=False)
    np.savetxt(resultpath, result, fmt='%.16f')

