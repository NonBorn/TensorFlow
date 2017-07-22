import os
import numpy as np
from PIL import Image

import tensorflow as tf
#path = '/Users/nonborn/[MSc] Business Analytics/3rd Semester/[SQ-PT-2017] Social Network Analysis and Social Media Analytics/Assignment (tensorflow)/sketches/airplane'
path = '/Users/nonborn/Desktop/test/'

files = [ path+f for f in  os.listdir(path)]

print(len(files))






def get_numpy(fpath):
    im = Image.open(fpath)
    im = im.convert('L') # to convert an image to grayscale
    im = im.resize((64, 64), Image.ANTIALIAS)
    im = np.asarray(im, dtype=np.float32)
    return im


b_size = 1
cnt = 0

if (cnt < len(files)):
    batch = np.asarray([ get_numpy(fpath) for fpath in files[cnt:cnt + b_size]  ])
    image = batch[0,:]
    print batch.shape
    #print batch



# CNN

import tensorflow as tf

# Parameters
learning_rate = 0.001
training_iters = 30000
batch_size = 1
display_step = 1

# Network Parameters
n_input = 64 #  data input (img shape: 64*64)
n_classes = 2 #  total classes (airplane or not)
dropout = 0.75 # Dropout, probability to keep units

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input,n_input])
y = tf.placeholder(tf.float32, [None,])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)


# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

# Create model
def conv_net(x, weights, biases, dropout):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, 64, 64, 1])
    #
    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)
    #
    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)
    print( conv2.get_shape() )
    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
    print( conv3.get_shape() )
    #
    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
#     fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.reshape(conv3, [-1, 16*16*20])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)
    #
    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out


# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    'wc3': tf.Variable(tf.random_normal([3, 3, 64, 20])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([16*16*20, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, n_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bc3': tf.Variable(tf.random_normal([20])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}


# Construct model
pred = conv_net(x, weights, biases, keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()


# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_x = batch
        batch_y = [1, 0]
            #mnist.train.next_batch(batch_size)
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})
        if step % display_step == 0:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y, keep_prob: 1.})
            # Calculate accuracy for 256 mnist test images
            t_acc = sess.run(accuracy, feed_dict={x: mnist.test.images[:256], y: mnist.test.labels[:256], keep_prob: 1.})
            print(
                "Iter " + str(step*batch_size) +
                ", Minibatch Loss= " + "{:.6f}".format(loss) +
                ", Training Accuracy= " + "{:.5f}".format(acc) +
                ", Testing Accuracy= " + "{:.5f}".format(t_acc)
            )
        step += 1
    print("Optimization Finished!")

