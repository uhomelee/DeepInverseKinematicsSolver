import numpy as np
import tensorflow as tf

#import matplotlib.pyplot as plt
X_train = np.loadtxt('./train_x.txt')
y_train = np.loadtxt('./train_y.txt')

#Model architecture parameters
n_dim = 3
n_neurons_1 = 128
n_neurons_2 = 128
n_neurons_3 = 256
n_rot = 6

# Make Session
net = tf.Session()
# Placeholder
X = tf.placeholder(dtype=tf.float32, shape=[None, n_dim])
Y = tf.placeholder(dtype=tf.float32, shape=[None, n_rot])

# Initializers
sigma = 1
weight_initializer = tf.variance_scaling_initializer(mode="fan_avg", distribution="uniform", scale=sigma)
bias_initializer = tf.zeros_initializer()

W_hiddens=locals()
bias_hiddens=locals()
hiddens=locals()

# Layer 1: Variables for hidden weights and biases
W_hidden_1 = tf.Variable(weight_initializer([n_dim, n_neurons_1]))
bias_hidden_1 = tf.Variable(bias_initializer([n_neurons_1]))
hidden_1 = tf.nn.relu(tf.add(tf.matmul(X, W_hidden_1), bias_hidden_1))
# Layer 2: Variables for hidden weights and biases
W_hidden_2 = tf.Variable(weight_initializer([n_neurons_1, n_neurons_2]))
bias_hidden_2 = tf.Variable(bias_initializer([n_neurons_2]))
hidden_2 = tf.nn.relu(tf.add(tf.matmul(hidden_1, W_hidden_2), bias_hidden_2))
# Layer 3: Variables for hidden weights and biases
W_hidden_3 = tf.Variable(weight_initializer([n_neurons_2, n_neurons_3]))
bias_hidden_3 = tf.Variable(bias_initializer([n_neurons_3]))
hidden_3 = tf.nn.relu(tf.add(tf.matmul(hidden_2, W_hidden_3), bias_hidden_3))
W_out = tf.Variable(weight_initializer([256, n_rot]))
bias_out = tf.Variable(bias_initializer([n_rot]))
#out = tf.add(tf.matmul(hiddens['hidden_15'], W_out), bias_out)
out = tf.add(tf.matmul(hidden_3, W_out), bias_out)

mse = tf.reduce_mean(tf.squared_difference(out, Y))
opt = tf.train.AdamOptimizer().minimize(mse)

# Compute the gradients for a list of variables.


# Run initializer
net.run(tf.global_variables_initializer())



# Number of epochs and batch size
epochs = 10000
batch_size = 128
saver = tf.train.Saver(max_to_keep=None)
min_mse=100000
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
        if np.mod(i, 5) == 0:
            mse_final = net.run(mse, feed_dict={X: batch_x, Y: batch_y})
            print(mse_final)
            total_mse += mse_final
            #total_mse+=1
    if total_mse < min_mse:
        with open("epoch.txt", 'a') as fopen:
            path_temp = "./model/model/model_final"
            save_path = saver.save(net, path_temp)
            min_mse = total_mse
            string = str(e) + "|" + str(total_mse) + '\n'
            fopen.write(string)
    if e % 500 == 0:
        model_path = "./model/model/model_" + str(e / 500 + 1)
        save_path = saver.save(net, model_path)
    with open("loss.txt", 'a') as f:
        f.write(str(e) + " " + str(total_mse) + '\n')

# Print final MSE after Training
#mse_final = net.run(mse, feed_dict={X: X_test, Y: y_test})
#print(mse_final)