import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

learning_rate = 0.01
training_epochs = 10
batch_size = 100
display_step = 1

def categorical_cross_entropy_loss(y,pred):
    # # Minimize error using cross entropy
    # # tf.reduce_mean(input_tensor, axis=None, keep_dims=False, name=None, reduction_indices=None)       # Computes the mean of elements across dimensions of a tensor.
    # # tf.reduce_sum(input_tensor, axis=None, keep_dims=False, name=None, reduction_indices=None)        # Computes the sum of elements across dimensions of a tensor.
    # # tf.log(x, name=None)                                                                              # Computes natural logarithm of x element-wise.
    cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))
    # # For brevity, let x = logits, z = labels. The logistic loss is z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
    # cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(targets=y, logits=tf.log(pred)))
    return cost

# tf Graph Input
x = tf.placeholder(tf.float32, [None, 784]) # mnist data image of shape 28*28=784
y = tf.placeholder(tf.float32, [None, 10]) # 0-9 digits recognition => 10 classes

# Set model weights
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Construct model
pred = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax
# print(pred.get_shape())

cost = categorical_cross_entropy_loss(y,pred)

# Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Test model
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
# Calculate accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# Initializing the variables
#init = tf.initialize_all_variables()    # tf version < 1.0.0
init = tf.global_variables_initializer() # tf version == 1.0.0

with tf.Session() as sess:
    sess.run(init)
    epochs = []
    tr_acc = []
    test_acc = []
    costs = []
    for epoch in range(training_epochs):
        avg_cost = 0.
        avg_acc = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c, acc = sess.run(
                [
                    optimizer,
                    cost,
                    accuracy
                ],
                feed_dict={
                    x: batch_xs,
                    y: batch_ys
                }
            )
            avg_cost += c / total_batch
            avg_acc  += acc / total_batch
            # print(avg_cost)
        if (epoch+1) % display_step == 0:
            # cmpute test acc
            tes_acc = sess.run( accuracy, feed_dict={
                x: mnist.test.images,
                y: mnist.test.labels
            } )
            print(
                "Epoch: "+str('%04d' % (epoch+1))+
                " cost="+"{:.9f}".format(avg_cost)+
                " train accuracy="+"{:.9f}".format(avg_acc)+
                " test accuracy="+"{:.9f}".format(tes_acc)
            )
            # add results to matrices
            epochs.append(epoch+1)
            tr_acc.append(avg_acc)
            test_acc.append(tes_acc)
            costs.append(avg_cost)
            #
    print("Optimization Finished!")