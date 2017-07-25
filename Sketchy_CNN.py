import os
import numpy as np
from PIL import Image
import regex as re
import tensorflow as tf



# build test dataset -> https://stackoverflow.com/questions/701402/best-way-to-choose-a-random-file-from-a-directory

# define file path of the folder of images
path_init = '/Users/nonborn/[MSc] Business Analytics/3rd Semester/[SQ-PT-2017] Social Network Analysis and Social Media Analytics/Assignment (tensorflow)/rendered_256x256/256x256/sketch/tx_000000000000'
# path_init = '/Users/nonborn/Desktop/test/'

# print(len(files))
# print('Folders in Directory of Parent Folder')
# print(os.listdir(path_init))

# Get all list of subfolders into filenames filtering those that start with . (hidden folders)
reg = re.compile("^\.")
filenames = [x for x in os.listdir(path_init) if not reg.match(x)]


# print('Folders in Directory of Parent Folder - After filtering')
#print (filenames)
dep = len(filenames)

# f_im = path_init + filenames[0]
# print(f_im)

# files = [f_im + '/' + f for f in os.listdir(f_im)]
# print(files)


# print('----')

# Function to transform an image to an numpy array. Resizing is performed by the function as well using antialiazing technique
def get_numpy(fpath):
    height = 64
    width = 64

    im = Image.open(fpath)
    # im.show(); # for debugging purposes
    im = im.resize((height, width), Image.ANTIALIAS) # resize the image to 64 x 64
    # im.show();
    im = im.convert('L')  # to convert an image to grayscale
    im = np.asarray(im, dtype=np.float32)
    return im


def one_hot_function(word):
    # Vocuabulary & 1 hot vectors
    text_idx = range(0,125)
    #print(text_idx)
    vocab_size = len(filenames)
    text_length = len(text_idx)
    one_hot = np.zeros(([vocab_size, text_length]))
    one_hot[text_idx, np.arange(text_length)] = 1
    one_hot = one_hot.astype(int)
    return one_hot[filenames.index(word)]



# batch Variables:
# ------------------
# files_list: list with the names of all the subfolders of the path
# child_index: pointer which has the starting folder of the files_list
# position: position index within the folder
def batch(files_list, child_index, b_size, pos_index):
    lpath = path_init + '/' + files_list[child_index]
    reg1 = re.compile("^\.")
    test = [x for x in os.listdir(lpath) if not reg1.match(x)]
    image_paths = [lpath + '/' + f for f in test]
    # print(test)

    if pos_index + b_size <= len(test):
        #print('true')
        batch_xx = np.asarray([get_numpy(fpath) for fpath in image_paths[pos_index:pos_index + b_size]])
        batch_yy = np.asarray([one_hot_function(files_list[child_index]) for x in image_paths[pos_index:pos_index + b_size]])
        pos_index = pos_index + b_size
    else:
        #print('false')
        batch_yy1 = np.asarray([one_hot_function(files_list[child_index]) for x in image_paths[pos_index:len(test)]])

        image_paths = [x for x in image_paths[pos_index:len(test)]]
        add_images = b_size - (len(test) - pos_index)
        # print(add_images)

        pos_index = 0
        child_index = child_index + 1
        lpath1 = path_init + '/' + files_list[child_index]
        reg1 = re.compile("^\.")
        test = [x for x in os.listdir(lpath1) if not reg1.match(x)]
        image_paths1 = [lpath1 + '/' + f for f in test]
        batch_yy2 = np.asarray([one_hot_function(files_list[child_index]) for x in image_paths1[pos_index:add_images]])
        #print(batch_yy2.shape)

        image_paths1 = [x for x in image_paths1[pos_index:add_images]]
        #print (image_paths1)

        image_paths2 = image_paths + image_paths1
        #print(image_paths2)

        batch_yy = np.append(batch_yy1, batch_yy2, axis = 0)

        batch_xx = np.asarray([get_numpy(fpath) for fpath in image_paths2])
        pos_index = pos_index + add_images
    return batch_xx,batch_yy, child_index, pos_index






child_index = 0
pos_index = 0
b_size = 50

#x = batch(filenames, child_index, b_size, pos_index)
# print(x[0],x[2])
# print(len(x))
# print(x[0])
# image = x[0, :]
# print x[0].shape
# print x[1].shape


# np.savetxt(path_init+'/test.txt', x[0], delimiter=';', newline='\n')




######################################################
####################  C N N  #########################
######################################################

# Parameters
learning_rate = 0.001
training_iters = 200000
display_step = 10

# Network Parameters
n_input = 64 #  data input (img shape: 64*64)
n_classes = 125 #  total classes (airplane or not)
dropout = 0.75 # Dropout, probability to keep units

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
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
    'wd1': tf.Variable(tf.random_normal([8*8*20, 1024])),
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

i=1
# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * b_size < training_iters:
        batch_x, batch_y, child_index, pos_index = batch(filenames, child_index, b_size, pos_index)
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})

        if step % display_step == 0:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y, keep_prob: 1.})
            # Calculate accuracy for 256 mnist test images
            #t_acc = sess.run(accuracy, feed_dict={x: mnist.test.images[:256], y: mnist.test.labels[:256], keep_prob: 1.})
            print(
                "Iter " + str(step*b_size) +
                ", Minibatch Loss= " + "{:.6f}".format(loss) +
                ", Training Accuracy= " + "{:.5f}".format(acc)
            )
        step += 1
    print("Optimization Finished!")
