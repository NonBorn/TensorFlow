"""""""""""""""""""""""""""
Import required libraries

"""""""""""""""""""""""""""

import os
import numpy as np
from PIL import Image
import regex as re
import tensorflow as tf
from datetime import datetime


"""""""""""""""""""""
Global Parameters:

"""""""""""""""""""""

# Resized source input image dimensions:
ImageSize = 64
Height = ImageSize
Width = ImageSize
offset = 40

batch_size = 400


"""""""""""""""""""""
Pre-Processing

"""""""""""""""""""""

# define file path of the folder of images
path_init = '/Users/nonborn/Desktop/Train Dataset'
init_train_path = '/Users/nonborn/Desktop/whole_train'
init_test_path = '/Users/nonborn/Desktop/whole_test'


def exclude_os_files(files_path):
    # Regex expression for hidden mac os files
    reg = re.compile("^\.")
    filenames = [x for x in os.listdir(files_path) if not reg.match(x)]
    return filenames


def get_numpy(fpath):
    im = Image.open(fpath)
    #im.show(); # for debugging purposes
    im = im.resize((Height, Width), Image.ANTIALIAS) # resize the image to 64 x 64
    #im.show();
    im = im.convert('L')  # to convert an image to grayscale
    im = np.asarray(im, dtype=np.float32)
    return im


def one_hot_function(word):
    # Vocabulary & 1 hot vectors
    text_idx = range(0, num_of_classes)
    #print(text_idx)
    vocab_size = len(class_path1)
    text_length = len(text_idx)
    one_hot = np.zeros(([vocab_size, text_length]))
    one_hot[text_idx, np.arange(text_length)] = 1
    one_hot = one_hot.astype(int)
    return one_hot[class_path1.index(word)]

"""
# random_batch  -  Takes images sequentially
def random_batch(parent_path, train_files_dir, index):
    if index + batch_size <= len(train_files_dir):
        tmp_list = [x for x in train_files_dir[index:index + batch_size]]
        classes = [x.rsplit('_', 1)[0].rsplit('/', 1)[1] for x in tmp_list]

        batch_xx = np.asarray([get_numpy(x) for x in train_files_dir[index:index + batch_size]])
        batch_yy = np.asarray([one_hot_function(y) for y in classes])
        t_index = index + batch_size + 1
    else:
        tmp_list = [x for x in train_files_dir[index:index + len(train_files_dir)]]
        classes = [x.rsplit('_', 1)[0].rsplit('/', 1)[1] for x in tmp_list]

        batch_xx = np.asarray([get_numpy(x) for x in train_files_dir[index:index + len(train_files_dir)]])
        batch_yy = np.asarray([one_hot_function(y) for y in classes])
        t_index = 0
    return batch_xx, batch_yy, t_index
"""

# random_Batch  -  Takes images by an offset
def random_batch(train_files_dir, index):
    if index + offset <= len(train_files_dir):
        tmp_list = [x for x in train_files_dir[index:index + batch_size]]
        classes = [x.rsplit('_', 1)[0].rsplit('/', 1)[1] for x in tmp_list]

        batch_xx = np.asarray([get_numpy(x) for x in train_files_dir[index:index + batch_size]])
        batch_yy = np.asarray([one_hot_function(y) for y in classes])
        t_index = index + offset + 1
    else:
        tmp_list = [x for x in train_files_dir[index:index + len(train_files_dir)]]
        classes = [x.rsplit('_', 1)[0].rsplit('/', 1)[1] for x in tmp_list]

        batch_xx = np.asarray([get_numpy(x) for x in train_files_dir[index:index + len(train_files_dir)]])
        batch_yy = np.asarray([one_hot_function(y) for y in classes])
        t_index = 0
    return batch_xx, batch_yy, t_index


# Variables Initialization
train_set = exclude_os_files(init_train_path)
test_set = exclude_os_files(init_test_path)
class_path1 = exclude_os_files(path_init)
# print(len(train_path))
# print(len(test_path))
num_of_classes = 125
print ('\n')
print ('Number of Classes: ' + str(num_of_classes))
print ('Batch Size: ' + str(batch_size) + '\n')


train_filenames = exclude_os_files(init_train_path)
test_filenames = exclude_os_files(init_test_path)

train_image_paths = [init_train_path + '/' + f for f in train_filenames]
test_image_paths = [init_test_path + '/' + f for f in test_filenames]

print ('Number of train images: ' + str(len(train_image_paths)))
print ('Number of test images: ' + str(len(test_image_paths))+ '\n')


# (parent_path, train_files_dir, index)
# Test Dataset Transformation
test_list = [x for x in test_image_paths[0:len(test_image_paths)]]
classes2 = [x.rsplit('_', 1)[0].rsplit('/', 1)[1] for x in test_list]

test_xx = np.asarray([get_numpy(x) for x in test_image_paths[0:len(test_image_paths)]])
test_yy = np.asarray([one_hot_function(y) for y in classes2])



Epoch = 0   # Epochs in CNN training
ind = 0     # index


"""""""""
i = 0
for i in range(0, 200):
    if ind == 0:
        Epoch = Epoch + 1
        print('Epoch: ' + str(Epoch))
    print('Index: ' + str(ind))
    i = i + 1
    x, y, ind = random_batch(init_train_path, train_image_paths, ind)
    print(x.shape)
    print(y.shape)
"""""""""




######################################################
####################  C N N  #########################
######################################################

# Parameters
learning_rate = 0.001
training_iters = 200000
display_step = 10

# Network Parameters
n_classes = 125             # Total classes (airplane or not)
dropout = 0.5               # Dropout, probability to keep units
lvl1_out = 48
lvl2_out = 128
lvl3_out = 192
cl_out = 2048


# tf Graph input
x = tf.placeholder(tf.float32, [None, ImageSize, ImageSize])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)


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
    print(conv1.get_shape().as_list())
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)
    print(conv1.get_shape().as_list())
    #
    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    print(conv2.get_shape().as_list())
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)
    print(conv2.get_shape().as_list())
    # print( conv2.get_shape() )

    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
    print(conv3.get_shape().as_list())
    conv3 = maxpool2d(conv3, k=2)
    print(conv3.get_shape().as_list())
    print('----')
    # print( conv3.get_shape() )


    # Fully connected layer

    # Reshape conv3 output to fit fully connected layer input
    fc1 = tf.reshape(conv3, [-1, weights['wd1'].get_shape().as_list()[0]])
    print(fc1.get_shape().as_list())

    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    #print (fc1.get_shape().as_list())
    fc1 = tf.nn.relu(fc1)
    print (fc1.get_shape().as_list())
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    print(out.get_shape().as_list())
    return out


# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    'wc3': tf.Variable(tf.random_normal([3, 3, 64, 20])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([8 * 8 * 20, 1024])),
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
    while t_acc <= 0.9:
        #step * batch_size < training_iters :
        if ind == 0:
            tt = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            Epoch = Epoch + 1
            print('Epoch ' + str(Epoch) + ' (' + str(tt) + ')')
        batch_x, batch_y, ind = random_batch(train_image_paths, ind)
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})

        if step % display_step == 0:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y, keep_prob: 1.})
            # Calculate accuracy for 256 mnist test images
            t_acc = sess.run(accuracy, feed_dict={x: test_xx, y: test_yy, keep_prob: 1.})
            print(
                "Iter " + str(step * batch_size) +
                ", Minibatch Loss= " + "{:.2f}".format(loss) +
                ", Training Accuracy= " + "{:.5f}".format(acc) +
                ", Test Accuracy = " + "{:.5f}".format(t_acc))
            )
        step += 1
    print("Optimization Finished!")
