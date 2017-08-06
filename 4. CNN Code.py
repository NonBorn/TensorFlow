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
Height = 128;
Width = Height;


"""""""""""""""""""""
Pre-Processing

"""""""""""""""""""""

# define file path of the folder of images
train_path = '/Users/nonborn/Desktop/tx_000100000000/Numpy/Train'
train_path_labels = '/Users/nonborn/Desktop/tx_000100000000/Numpy/Train_Labels'
test_path = '/Users/nonborn/Desktop/tx_000100000000/Numpy/Test'
test_path_labels = '/Users/nonborn/Desktop/tx_000100000000/Numpy/Test_Labels'


def exclude_os_files(files_path):
    # Regex expression for hidden mac os files
    reg = re.compile("^\.")
    filenames = [x for x in os.listdir(files_path) if not reg.match(x)]
    return filenames


def random_batch(fpath_train, fpath_labels, b_size):
    # exclude os files
    class_path = exclude_os_files(fpath_train)
    # print(class_path) # debugging - shows the list of folders within the parent folder
    index = rng.sample(range(0, len(class_path)), b_size)  # get a number of images per class - indexing

    img_path = [fpath_train + '/' + os.listdir(train_path)[x] for x in index]
    label_path = [fpath_labels + '/' + os.listdir(train_path_labels)[x] for x in index]
    #print(img_path)
    #print(label_path)

    random_images_features = [np.load(img_path[x]) for x in range(0, len(img_path))]
    random_images_labels = [np.load(label_path[x]) for x in range(0, len(label_path))]
    return random_images_features, random_images_labels



# Variables Initialization
num_of_classes = 125
batch_size = 100


#print (class_path);
print ('\n')
print ('Number of Classes: ' + str(num_of_classes))
print ('Batch Size: ' + str(batch_size) + '\n')


x, y = random_batch(train_path, train_path_labels, batch_size)



# Load Test set
class_path = exclude_os_files(test_path)
index = range(0, len(class_path))

test_img_path = [test_path + '/' + os.listdir(test_path)[x] for x in index]
test_label_path = [test_path_labels + '/' + os.listdir(test_path_labels)[x] for x in index]

test_images_features = [np.load(test_img_path[x]) for x in range(0, len(test_img_path))]
test_images_labels = [np.load(test_label_path[x]) for x in range(0, len(test_label_path))]







######################################################
####################  C N N  #########################
######################################################

# Parameters
learning_rate = 0.001
training_iters = 200000
display_step = 1
lvl1_out = 32
lvl2_out = 64
lvl3_out = 20
cl_out = 1024

# Network Parameters

#  Data input (img shape: 64*64)
n_input = Height
# Dropout, probability to keep units
dropout = 0.75

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
    x = tf.reshape(x, shape=[-1, 128, 128, 1])
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
    'wd1': tf.Variable(tf.random_normal([4 * 8 * 8 * 20, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, num_of_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bc3': tf.Variable(tf.random_normal([20])),
    'bd1': tf.Variable(tf.random_normal([1024])),
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
    while step * num_of_classes < training_iters:
        batch_x, batch_y, = random_batch(train_path, train_path_labels, batch_size)
        print(np.shape(batch_x))
        print(np.shape(batch_y))
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})

        if step % display_step == 0:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y, keep_prob: 1.})
            # Calculate accuracy for 256 mnist test images
            t_acc = sess.run(accuracy, feed_dict={x: test_images_features[:250], y: test_images_labels[:250], keep_prob: 1.})
            print(
                "Iter " + str(step * num_of_classes) +
                ", Minibatch Loss = " + "{:.0f}".format(loss) +
                ", Training Accuracy = " + "{:.5f}".format(acc) +
                ", Test Accuracy = " + "{:.5f}".format(t_acc)
            )
            step += 1
    print("Optimization Finished!")