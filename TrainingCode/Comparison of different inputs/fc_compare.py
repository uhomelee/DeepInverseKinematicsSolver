import numpy as np
import tensorflow as tf
#import matplotlib.pyplot as plt

X_train = np.loadtxt('./TrainDataOutput/5f/train_x_5_10f.txt')
y_train = np.loadtxt('./TrainDataOutput/5f/train_y_5_10f.txt')
X_test = np.loadtxt('./TrainDataOutput/5f/test_x_5_10f.txt')
y_test = np.loadtxt('./TrainDataOutput/5f/test_y_5_10f.txt')

n_dim = 15
"""
n_neurons_1 = 256
n_neurons_2 = 512
n_neurons_3 = 512
n_neurons_4 = 1024
n_neurons_5 = 1024
n_neurons_6 = 1024
n_rot=6
"""
n_neurons_1 = 128
n_neurons_2 = 128
n_neurons_3 = 256
n_rot = 6

# Make Session
net = tf.Session()
# Placeholder
X = tf.placeholder(dtype=tf.float32, shape=[None, n_dim])
Y = tf.placeholder(dtype=tf.float32, shape=[None, n_rot])
keep_prob = tf.placeholder(tf.float32)

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
"""
## Layer 4: Variables for hidden weights and biases
W_hidden_4 = tf.Variable(weight_initializer([n_neurons_3, n_neurons_4]))
bias_hidden_4 = tf.Variable(bias_initializer([n_neurons_4]))
hidden_4 = tf.nn.relu(tf.add(tf.matmul(hidden_3, W_hidden_4), bias_hidden_4))

## Layer 5: Variables for hidden we
# ights and biases
W_hidden_5 = tf.Variable(weight_initializer([n_neurons_4, n_neurons_5]))
bias_hidden_5 = tf.Variable(bias_initializer([n_neurons_5]))
hidden_5 = tf.nn.relu(tf.add(tf.matmul(hidden_4, W_hidden_5), bias_hidden_5))
W_hidden_6 = tf.Variable(weight_initializer([n_neurons_5, n_neurons_6]))
bias_hidden_6 = tf.Variable(bias_initializer([n_neurons_6]))
hiddens['hidden_6']=tf.nn.relu(tf.add(tf.matmul(hidden_5, W_hidden_6), bias_hidden_6))

for i in range(7,16):
    #print(i)
    W_hiddens['W_hidden_%s'%i]=tf.Variable(weight_initializer([1024, 1024]))
    bias_hiddens['bias_hidden_%s'%i]=tf.Variable(bias_initializer([1024]))
    hiddens['hidden_%s'%i]=tf.nn.relu(tf.add(tf.matmul(hiddens['hidden_%s'%(i-1)], W_hiddens['W_hidden_%s'%i]), bias_hiddens['bias_hidden_%s'%i]))
    #print(hiddens['hidden_%s'%i].shape)
# Output layer: Variables for output weights and biases
"""
dropout=tf.nn.dropout(hidden_3, keep_prob=keep_prob)
W_out = tf.Variable(weight_initializer([256, n_rot]))
bias_out = tf.Variable(bias_initializer([n_rot]))
#out = tf.add(tf.matmul(hiddens['hidden_15'], W_out), bias_out)
out = tf.add(tf.matmul(dropout, W_out), bias_out)

mse = tf.reduce_mean(tf.squared_difference(out, Y))
opt = tf.train.AdamOptimizer().minimize(mse)
#global_step=tf.Variable(0,trainable=False)
#start_learning_rate=0.001
#learning_rate=tf.train.exponential_decay(start_learning_rate,global_step,decay_steps=850000,decay_rate=0.96,staircase=True)
#optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
#optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
#gvs = optimizer.compute_gradients(mse)
#capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
#train_op = optimizer.apply_gradients(capped_gvs,global_step=global_step)
#train_op = optimizer.apply_gradients(capped_gvs)

# Compute the gradients for a list of variables.


# Run initializer
net.run(tf.global_variables_initializer())



# Number of epochs and batch size
epochs = 100000
batch_size = 128
saver = tf.train.Saver(max_to_keep=None)
min_mse=100000
min_mse_test=10000
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
        net.run(opt, feed_dict={X: batch_x, Y: batch_y,keep_prob:0.01})

        # Show progress
        if np.mod(i, 5) == 0:
            mse_final = net.run(mse, feed_dict={X: batch_x, Y: batch_y,keep_prob:1})
            print(mse_final)
            total_mse += mse_final
            #total_mse+=1


































































































































































































































































































































































































































































































































































































































    test_mse=net.run(mse,feed_dict={X:X_test,Y:y_test,keep_prob:1})
    print('loss of test='+str(test_mse))
    if test_mse<min_mse_test:
        saver.save(net, './model/model_compare/model_fortest')
        min_mse_test=test_mse
        with open("epoch_test.txt", 'a') as fopen:
            string = str(e) + "|" + str(min_mse_test) + '\n'
            fopen.write(string)
    if total_mse < min_mse:
        with open("epoch.txt", 'a') as fopen:
            path_temp = "./model/model_compare/model_final"
            save_path = saver.save(net, path_temp)
            min_mse = total_mse
            string = str(e) + "|" + str(total_mse) + '\n'
            fopen.write(string)

    if e % 500 == 0:
        model_path = "./model/model/model_" + str(e / 500 + 1)
        save_path = saver.save(net, model_path)
    with open("loss.txt", 'a') as f:
        f.write(str(e) + " " + str(total_mse) + '\n')
    with open("loss_test.txt", 'a') as f:
        f.write(str(e) + " " + str(test_mse) + '\n')

# Print final MSE after Training
#mse_final = net.run(mse, feed_dict={X: X_test, Y: y_test})
#print(mse_final)