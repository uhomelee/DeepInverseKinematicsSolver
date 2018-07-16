import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf

Z = np.loadtxt('train_x_5_10f.txt')
X = np.loadtxt('train_y_5_10f.txt')

batch_size=128
sigma=1

weight_initializer = tf.variance_scaling_initializer(mode="fan_avg", distribution="uniform", scale=sigma)
bias_initializer = tf.zeros_initializer()

def iterate_minibatch(x,batch_size,shuffle=True):
    indices=np.arange(x.shape[0])
    if shuffle:
        np.random.shuffle(indices)
    for i in range(0, x.shape[0], batch_size):
        yield x[indices[i:i+batch_size],:]

class GAN(object):
    def __init__(self):
        #input and output
        self.z=tf.placeholder(tf.float32,shape=[None,15],name='z')
        self.x=tf.placeholder(tf.float32,shape=[None,21],name='real_x')
        #define the network
        self.fake_x=tf.concat([self.z,self.netG(self.z)], axis=1)
        self.real_logits=self.netD(self.x,reuse=False)
        self.fake_logits=self.netD(self.fake_x,reuse=True)
        #define losses
        self.loss_D=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.real_logits,
                                                                           labels=tf.ones_like(self.real_logits)))+ \
                    tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.fake_logits,
                                                                   labels=tf.zeros_like(self.real_logits)))
        self.loss_G=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.fake_logits,
                                                                           labels=tf.ones_like(self.real_logits)))
        #collect variables
        t_vars=tf.trainable_variables()
        self.d_vars=[var for var in t_vars if 'd_' in var.name]
        self.g_vars=[var for var in t_vars if 'g_' in var.name]

    def netG(self,z):
        # 4 layer full-connected network
        with tf.variable_scope("generator") as scope:
            W1=tf.get_variable(name='g_W1',shape=[15,256],
                           initializer=tf.contrib.layers.xavier_initializer(),
                           trainable=True)
            b1 = tf.get_variable(name="g_b1", shape=[256],
                                initializer=tf.zeros_initializer(),
                                trainable=True)
            W2 = tf.get_variable(name='g_W2', shape=[256, 256],
                             initializer=tf.contrib.layers.xavier_initializer(),
                             trainable=True)
            b2 = tf.get_variable(name="g_b2", shape=[256],
                                 initializer=tf.zeros_initializer(),
                                 trainable=True)
            W3 = tf.get_variable(name='g_W3', shape=[256, 512],
                             initializer=tf.contrib.layers.xavier_initializer(),
                             trainable=True)
            b3 = tf.get_variable(name="g_b3", shape=[512],
                                 initializer=tf.zeros_initializer(),
                                 trainable=True)
            W4 = tf.get_variable(name='g_W4', shape=[512, 6],
                             initializer=tf.contrib.layers.xavier_initializer(),
                             trainable=True)
            b4 = tf.get_variable(name="g_b4", shape=[6],
                                 initializer=tf.zeros_initializer(),
                                 trainable=True)

            layer1=tf.nn.relu(tf.add(tf.matmul(z, W1),b1))
            layer2=tf.nn.relu(tf.add(tf.matmul(layer1,W2),b2))
            layer3=tf.nn.relu(tf.add(tf.matmul(layer2,W3),b3))
            return tf.add(tf.matmul(layer3,W4),b4)

    def netD(self,x,reuse=False):
        # 11 layer full-connected network
        with tf.variable_scope('discriminator') as scope:
            if reuse:
                scope.reuse_variables()
            W1 = tf.get_variable(name='d_W1', shape=[21, 64],
                                 initializer=tf.contrib.layers.xavier_initializer(),
                                 trainable=True)
            b1 = tf.get_variable(name="d_b1", shape=[64],
                                 initializer=tf.zeros_initializer(),
                                 trainable=True)
            W2 = tf.get_variable(name='d_W2', shape=[64, 128],
                                 initializer=tf.contrib.layers.xavier_initializer(),
                                 trainable=True)
            b2 = tf.get_variable(name="d_b2", shape=[128],
                                 initializer=tf.zeros_initializer(),
                                 trainable=True)
            W3 = tf.get_variable(name='d_W3', shape=[128, 256],
                                 initializer=tf.contrib.layers.xavier_initializer(),
                                 trainable=True)
            b3 = tf.get_variable(name="d_b3", shape=[256],
                                 initializer=tf.zeros_initializer(),
                                 trainable=True)
            W4 = tf.get_variable(name='d_W4', shape=[256, 512],
                                 initializer=tf.contrib.layers.xavier_initializer(),
                                 trainable=True)
            b4 = tf.get_variable(name="d_b4", shape=[512],
                                 initializer=tf.zeros_initializer(),
                                 trainable=True)
            W5 = tf.get_variable(name='d_W5', shape=[512, 512],
                                 initializer=tf.contrib.layers.xavier_initializer(),
                                 trainable=True)
            b5 = tf.get_variable(name="d_b5", shape=[512],
                                 initializer=tf.zeros_initializer(),
                                 trainable=True)
            W6 = tf.get_variable(name='d_W6', shape=[512, 512],
                                 initializer=tf.contrib.layers.xavier_initializer(),
                                 trainable=True)
            b6 = tf.get_variable(name="d_b6", shape=[512],
                                 initializer=tf.zeros_initializer(),
                                 trainable=True)
            W7 = tf.get_variable(name='d_W7', shape=[512,512],
                                 initializer=tf.contrib.layers.xavier_initializer(),
                                 trainable=True)
            b7 = tf.get_variable(name="d_b7", shape=[512],
                                 initializer=tf.zeros_initializer(),
                                 trainable=True)
            W8 = tf.get_variable(name='d_W8', shape=[512, 512],
                                 initializer=tf.contrib.layers.xavier_initializer(),
                                 trainable=True)
            b8 = tf.get_variable(name="d_b8", shape=[512],
                                 initializer=tf.zeros_initializer(),
                                 trainable=True)
            W9 = tf.get_variable(name='d_W9', shape=[512, 32],
                                 initializer=tf.contrib.layers.xavier_initializer(),
                                 trainable=True)
            b9 = tf.get_variable(name="d_b9", shape=[32],
                                 initializer=tf.zeros_initializer(),
                                 trainable=True)
            W10 = tf.get_variable(name='d_W10', shape=[32, 4],
                                 initializer=tf.contrib.layers.xavier_initializer(),
                                 trainable=True)
            b10 = tf.get_variable(name="d_b10", shape=[4],
                                 initializer=tf.zeros_initializer(),
                                 trainable=True)
            W11 = tf.get_variable(name='d_W11', shape=[4, 1],
                                 initializer=tf.contrib.layers.xavier_initializer(),
                                 trainable=True)
            b11 = tf.get_variable(name="d_b11", shape=[1],
                                 initializer=tf.zeros_initializer(),
                                 trainable=True)

            """
            layer1 = tf.nn.tanh(tf.matmul(x, W1) + b1)
            layer2 = tf.nn.tanh(tf.matmul(layer1, W2) + b2)
            layer3 = tf.nn.tanh(tf.matmul(layer2, W3) + b3)
            layer4 = tf.nn.tanh(tf.matmul(layer3, W4) + b4)
            layer5 = tf.nn.tanh(tf.matmul(layer4, W5) + b5)
            layer6 = tf.nn.tanh(tf.matmul(layer5, W6) + b6)
            layer7 = tf.nn.tanh(tf.matmul(layer6, W7) + b7)
            layer8 = tf.nn.tanh(tf.matmul(layer7, W8) + b8)
            layer9 = tf.nn.tanh(tf.matmul(layer8, W9) + b9)
            layer10 = tf.nn.tanh(tf.matmul(layer9, W10) + b10)
            """
            layer1 =tf.nn.relu(tf.add(tf.matmul(x, W1),b1))
            layer2 = tf.nn.relu(tf.add(tf.matmul(layer1, W2), b2))
            layer3 = tf.nn.relu(tf.add(tf.matmul(layer2, W3), b3))
            layer4 = tf.nn.relu(tf.add(tf.matmul(layer3, W4), b4))
            layer5 = tf.nn.relu(tf.add(tf.matmul(layer4, W5), b5))
            layer6 = tf.nn.relu(tf.add(tf.matmul(layer5, W6), b6))
            layer7 = tf.nn.relu(tf.add(tf.matmul(layer6, W7), b7))
            layer8 = tf.nn.relu(tf.add(tf.matmul(layer7, W8), b8))
            layer9 = tf.nn.relu(tf.add(tf.matmul(layer8, W9), b9))
            layer10 = tf.nn.tanh(tf.matmul(layer9, W10) + b10)
            return tf.matmul(layer10, W11) + b11
            #return tf.nn.tanh(tf.matmul(layer10,W11),b11)
#training loop
gan=GAN()
#todo
d_optim = tf.train.AdamOptimizer(learning_rate=0.00001).minimize(gan.loss_D, var_list=gan.d_vars)
g_optim = tf.train.AdamOptimizer(learning_rate=0.00001).minimize(gan.loss_G, var_list=gan.g_vars)

init=tf.global_variables_initializer()
#config = tf.ConfigProto(device_count = {'GPU': 0})

#with tf.Session(config=config) as sess:
with tf.Session() as sess:
    sess.run(init)

    for epoch in range(8000):
        avg_loss=0
        count=0
        shuffle_indices = np.random.permutation(np.arange(len(X)))
        z = Z[shuffle_indices]
        x = X[shuffle_indices]
        for i in range(0,len(x)//batch_size):
            start=i*batch_size
            x_batch=X[start:start+batch_size]
            z_batch=Z[start:start+batch_size]

            # update D network
            loss_D, _ = sess.run([gan.loss_D, d_optim],
                                 feed_dict={
                                     gan.z: z_batch,
                                     gan.x: x_batch,
                                 })
            #print('loss_D:'+str(loss_D))

            # update G network
            loss_G, _ = sess.run([gan.loss_G, g_optim],
                                 feed_dict={
                                     gan.z: z_batch,
                                     gan.x: x_batch,  # dummy input#todo
                                 })
            #print('loss_G:' + str(loss_G))

            avg_loss += loss_D
            count += 1

        avg_loss /= count
        #z = np.random.normal(size=(100, 15))
        excerpt = np.random.randint(1000, size=100)
        fake_x, real_logits, fake_logits = sess.run([gan.fake_x, gan.real_logits, gan.fake_logits],
                                                    feed_dict={gan.z: Z[excerpt,:], gan.x: X[excerpt, :]})
        accuracy = 0.5 * (np.sum(real_logits > 0) / 100. + np.sum(fake_logits < 0) / 100.)
        print('\ndiscriminator loss at epoch %d: %f' % (epoch, avg_loss))
        print('\ndiscriminator accuracy at epoch %d: %f' % (epoch, accuracy))

    path='/home/student/fulldata/nine_input/xyz_data_5_10f/train/01_01.txt'
    result_path='./01_01.txt'
    Z_test = np.loadtxt(path)
    print(Z_test.shape)
    result = sess.run(gan.fake_x, feed_dict={gan.z: Z_test})
    np.set_printoptions(suppress=False)
    np.savetxt(result_path, result, fmt='%.16f')
