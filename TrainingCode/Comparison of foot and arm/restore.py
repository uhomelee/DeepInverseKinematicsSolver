import tensorflow as tf
import numpy as np
import os

#X_test = np.loadtxt('xyz_0101.txt')


n_dim = 15

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
"""
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

for i in range(7,11):
    #print(i)
    W_hiddens['W_hidden_%s'%i]=tf.Variable(weight_initializer([1024, 1024]))
    bias_hiddens['bias_hidden_%s'%i]=tf.Variable(bias_initializer([1024]))
    hiddens['hidden_%s'%i]=tf.nn.relu(tf.add(tf.matmul(hiddens['hidden_%s'%(i-1)], W_hiddens['W_hidden_%s'%i]), bias_hiddens['bias_hidden_%s'%i]))
    #print(hiddens['hidden_%s'%i].shape)
# Output layer: Variables for output weights and biases

W_out = tf.Variable(weight_initializer([1024, n_rot]))
bias_out = tf.Variable(bias_initializer([n_rot]))
out = tf.add(tf.matmul(hiddens['hidden_10'], W_out), bias_out)
# Run initializer
#net.run(tf.global_variables_initializer())


saver = tf.train.Saver()

#todo
saver.restore(net, "./model/model_10layer/model_final")



# Now, let's access and create placeholders variables and
# create feed-dict to feed new data
#batch_size=1
count=0
path='./xyzData'
files= os.listdir(path)
for file in files:
    path='./xyzData/'+file
    X_test=np.loadtxt(path)
    resultpath='./channelData/'+file
    # for i in range(0, len(X_test)):
    # start=i*batch_size
    # batch_x = X_test[start:start + batch_size]
    # batch_x=X_test[i]
    # feed_dict={X:batch_x}
    #print(X_test.shape)
    result = net.run(out, feed_dict={X: X_test})
    count+=1
    print(str(count)+':'+str(result.shape))
    # for i in range(0,len(result)):
    # f.write(str(result[i,:])+'\n')
    # print(i)
    # print(result[i,:])
    np.set_printoptions(suppress=False)
    np.savetxt(resultpath, result, fmt='%.16f')
    """
count=0
path='./xyz_data_5_10f/test'
files= os.listdir(path)
for file in files:
    path='./xyz_data_5_10f/test/'+file
    X_test=np.loadtxt(path)
    resultpath='./channel_5_10f/test/'+file
    # for i in range(0, len(X_test)):
    # start=i*batch_size
    # batch_x = X_test[start:start + batch_size]
    # batch_x=X_test[i]
    # feed_dict={X:batch_x}
    #print(X_test.shape)
    result = net.run(out, feed_dict={X: X_test})
    count+=1
    print(str(count)+':'+str(result.shape))
    # for i in range(0,len(result)):
    # f.write(str(result[i,:])+'\n')
    # print(i)
    # print(result[i,:])
    np.set_printoptions(suppress=False)
    np.savetxt(resultpath, result, fmt='%.16f')
    """

