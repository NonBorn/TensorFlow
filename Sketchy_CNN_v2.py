"""""""""""""""""""""""""""
Import required libraries

"""""""""""""""""""""""""""
import os
import numpy as np
from PIL import Image
import regex as re
import tensorflow as tf
import random as rng



"""""""""""""""""""""
Global Parameters:

"""""""""""""""""""""

# Resized source input image dimensions:
Height = 256;
Width = Height;
# images_per_class represent the batch size of tensorflow as images_per_class * num_of_classes
images_per_class = 32



"""""""""""""""""""""
Pre-Processing

"""""""""""""""""""""

# define file path of the folder of images
path_init = '/Users/nonborn/Desktop/test/train'
target = '/Users/nonborn/Desktop/test/test'

def exclude_os_files(files_path):
    # Regex expression for hidden mac os files
    reg = re.compile("^\.")
    filenames = [x for x in os.listdir(files_path) if not reg.match(x)]
    return filenames


def get_numpy(fpath):
    im = Image.open(fpath)
    # im.show(); # for debugging purposes
    im = im.resize((Height, Width), Image.ANTIALIAS) # resize the image to 64 x 64
    # im.show();
    im = im.convert('L')  # to convert an image to grayscale
    im = np.asarray(im, dtype=np.float32)
    return im


def one_hot_function(word):
    # Vocuabulary & 1 hot vectors
    text_idx = range(0, num_of_classes)
    #print(text_idx)
    vocab_size = len(class_path1)
    text_length = len(text_idx)
    one_hot = np.zeros(([vocab_size, text_length]))
    one_hot[text_idx, np.arange(text_length)] = 1
    one_hot = one_hot.astype(int)
    return one_hot[class_path1.index(word)]


def random_batch(fpath):
    # exclude os files
    class_path = exclude_os_files(fpath)
    # print(class_path) # debugging - shows the list of folders within the parent folder

    tmp_img_list = []
    tmp_img_labels = []

    for f in range (0, num_of_classes):
        current_path = path_init + '/' + class_path[f]
        files = os.listdir(current_path)     # list of files in current path
        tmp = exclude_os_files(current_path)  # exclude system files
        index = rng.sample(range(0, len(tmp)), images_per_class)  # get a number of images per class - indexing
        # print (class_path[f])
        # create paths of random images per class
        tmp_img_list = tmp_img_list + [current_path + '/' + files[s] for s in index]
        tmp_img_labels = tmp_img_labels + [class_path[f] for s in index]
    #print(tmp_img_labels)
    #print(tmp_img_list)

    random_images_features = np.asarray([get_numpy(t) for t in tmp_img_list])
    #print(random_images_features.shape)
    random_images_labels = np.asarray([one_hot_function(f) for f in tmp_img_labels])
    #print(random_images_labels.shape)

    return random_images_features, random_images_labels

# Variables Initialization
class_path1 = exclude_os_files(path_init)
num_of_classes = len(class_path1);


#print (class_path);
print ('\n')
print ('Number of Classes: ' + str(num_of_classes))
print ('Batch Size: ' + str(images_per_class*num_of_classes)+ '\n')







######################################################
####################  C N N  #########################
######################################################

# Parameters
learning_rate = 0.001
training_iters = 200000
display_step = 1
lvl1_out = 48
lvl2_out = 128
lvl3_out = 192
lvl4_out = 192
lvl5_out = 128
cl_out = 2048

# Network Parameters

#  Data input (img shape: 64*64)
n_input = Height
# Dropout, probability to keep units
dropout = 0.5

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input, n_input])
y = tf.placeholder(tf.float32, [None, num_of_classes])
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
    x = tf.reshape(x, shape=[-1, n_input, n_input, 1])
    #
    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    #print(conv1.get_shape().as_list())
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)
    print('Conv. Layer 1: ' + str(conv1.get_shape().as_list()))
    #
    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    #print(conv2.get_shape().as_list())
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=1)
    print('Conv. Layer 2: ' + str(conv2.get_shape().as_list()))

    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
    #print(conv3.get_shape().as_list())
    #conv3 = maxpool2d(conv3, k=2)
    print('Conv. Layer 3: ' + str(conv3.get_shape().as_list()))

    conv4 = conv2d(conv3, weights['wc4'], biases['bc4'])
    #print(conv4.get_shape().as_list())
    #conv4 = maxpool2d(conv4, k=2)
    print('Conv. Layer 4: ' + str(conv4.get_shape().as_list()))

    conv5 = conv2d(conv4, weights['wc5'], biases['bc5'])
    # print(conv4.get_shape().as_list())
    conv5 = maxpool2d(conv5, k=2)
    print('Conv. Layer 5: ' + str(conv5.get_shape().as_list()))

    print('----')

    # Fully connected layer
    # Reshape conv3 output to fit fully connected layer input
    fc1 = tf.reshape(conv5, [-1, weights['wd1'].get_shape().as_list()[0]])
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
    'wc1': tf.Variable(tf.random_normal([10, 10, 1, lvl1_out])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, lvl1_out, lvl2_out])),
    'wc3': tf.Variable(tf.random_normal([3, 3, lvl2_out, lvl3_out])),
    'wc4': tf.Variable(tf.random_normal([3, 3, lvl3_out, lvl4_out])),
    'wc5': tf.Variable(tf.random_normal([3, 3, lvl4_out, lvl5_out])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([16 * 16 * 16 * lvl5_out, cl_out])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([cl_out, num_of_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([lvl1_out])),
    'bc2': tf.Variable(tf.random_normal([lvl2_out])),
    'bc3': tf.Variable(tf.random_normal([lvl3_out])),
    'bc4': tf.Variable(tf.random_normal([lvl4_out])),
    'bc5': tf.Variable(tf.random_normal([lvl5_out])),
    'bd1': tf.Variable(tf.random_normal([cl_out])),
    'out': tf.Variable(tf.random_normal([num_of_classes]))
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

i=1
# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * num_of_classes * images_per_class < training_iters:
        batch_x, batch_y, = random_batch(path_init)
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})

        if step % display_step == 0:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y, keep_prob: 1.})
            # Calculate accuracy for 256 mnist test images
            #t_acc = sess.run(accuracy, feed_dict={x: mnist.test.images[:256], y: mnist.test.labels[:256], keep_prob: 1.})
            print(
                "Iter " + str(step * num_of_classes * images_per_class) +
                ", Minibatch Loss = " + "{:.0f}".format(loss) +
                ", Training Accuracy = " + "{:.5f}".format(acc) +
                ", Test Accuracy = " + "{:.5f}".format(sess.run(accuracy, feed_dict={x: random_batch(target)[0], y: random_batch(target)[1], keep_prob: 1.}))
            )
            step += 1
    print("Optimization Finished!")