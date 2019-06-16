import datetime
import numpy as np
import tensorflow as tf

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

#import training data
X_train = np.loadtxt('./train_x.txt')
y_train = np.loadtxt('./train_y.txt')

# Model architecture parameters
n_dim = 15
n_neurons_1 = 128
n_neurons_2 = 512
n_neurons_3 = 512
n_neurons_4 = 256
n_rot = 6
batch_size = 32

# Make Session
net = tf.Session()
# Placeholder
X = tf.placeholder(dtype=tf.float32, shape=[None, n_dim])
Y = tf.placeholder(dtype=tf.float32, shape=[None, n_rot])

# Initializers
sigma = 1
weight_initializer = tf.variance_scaling_initializer(mode="fan_avg", distribution="uniform", scale=sigma)
bias_initializer = tf.zeros_initializer()

# Layer 1: Variables for hidden weights and biases
W_hidden_1 = tf.Variable(weight_initializer([n_dim, n_neurons_1]))
bias_hidden_1 = tf.Variable(bias_initializer([n_neurons_1]))
hidden_1 = tf.nn.relu(tf.add(tf.matmul(X, W_hidden_1), bias_hidden_1))
#CNN layer 2
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
x_conv1 = tf.reshape(hidden_1, [-1, batch_size, n_neurons_1, 1])
h_conv1 = tf.nn.relu(conv2d(x_conv1, W_conv1) + b_conv1)
#CNN layer 3
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)
# Layer 4: Variables for hidden weights and biases
W_fc1 = weight_variable([batch_size*n_neurons_1*64, n_neurons_2])
b_fc1 = bias_variable([n_neurons_2])
h_pool2_flat = tf.reshape(h_conv2, [-1,batch_size*n_neurons_1*64])
hidden_2 = tf.nn.relu(tf.add(tf.matmul(h_pool2_flat, W_fc1), b_fc1))
# Layer 5: Variables for hidden weights and biases
W_hidden_3 = tf.Variable(weight_initializer([n_neurons_2, n_neurons_3]))
bias_hidden_3 = tf.Variable(bias_initializer([n_neurons_3]))
hidden_3 = tf.nn.relu(tf.add(tf.matmul(hidden_2, W_hidden_3), bias_hidden_3))
# Output layer: Variables for output weights and biases
W_out = tf.Variable(weight_initializer([n_neurons_3, n_rot]))
bias_out = tf.Variable(bias_initializer([n_rot]))
out = tf.add(tf.matmul(hidden_3, W_out), bias_out)

# Cost function
mse = tf.reduce_mean(tf.squared_difference(out, Y))
#Learning rate decrease
global_step=tf.Variable(0,trainable=False)
start_learning_rate=0.001
learning_rate=tf.train.exponential_decay(start_learning_rate,global_step,decay_steps=30000,decay_rate=0.98,staircase=True)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
gvs = optimizer.compute_gradients(mse)
capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
opt = optimizer.apply_gradients(capped_gvs,global_step=global_step)

net.run(tf.global_variables_initializer())


# Number of epochs and batch size
epochs = 10000
batch_size = 32
saver = tf.train.Saver()
starttime = datetime.datetime.now()
saver = tf.train.Saver(max_to_keep=None)
min_mse=1000
for e in range(epochs):
    print("------" + str(e) + ":-------")
    # Shuffle training data
    shuffle_indices = np.random.permutation(np.arange(len(y_train)))
    X_train = X_train[shuffle_indices]
    y_train = y_train[shuffle_indices]
    total_mse = 0
    # Minibatch training
    for i in range(0, len(y_train) // batch_size):
        start = i * batch_size
        batch_x = X_train[start:start + batch_size]
        batch_y = y_train[start:start + batch_size]
        # Run optimizer with batch
        net.run(opt, feed_dict={X: batch_x, Y: batch_y})

        # Show progress
        if np.mod(i, 20) == 0:
            mse_final = net.run(mse, feed_dict={X: batch_x, Y: batch_y})
            #print(mse_final)
            total_mse += mse_final
    with open("loss_cnn.txt", 'a') as f:
        endtime = datetime.datetime.now()
        runtime = (endtime - starttime).seconds
        f.write(str(e) + " " + str(runtime) + ' ' + str(total_mse) + '\n')
    print("Step " + str(e) + " Loss = " + str(total_mse) + ' Learning_rate=' + str(net.run(
        learning_rate)) + ' time=' + str(runtime))
