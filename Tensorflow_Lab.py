import tensorflow as tf

x = tf.constant([[1., 2.]])

neg_op = tf.negative(x)

with tf.Session() as sess:
    result = sess.run(neg_op)

print(result)

####################################################################

import tensorflow as tf

sess = tf.InteractiveSession()
x = tf.constant([[1., 2.]])
neg_x = tf.negative(x)
result = neg_x.eval()

print(result)

####################################################################

import tensorflow as tf

x = tf.constant([[1., 2.]])
neg_x = tf.negative(x)

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    result = sess.run(neg_x)
    sess.close()

print(result)

####################################################################

import tensorflow as tf

sess = tf.InteractiveSession()
raw_data = [1., 2., 8., -1., 0., 5.5, 6., 13]
spike = tf.Variable(False)
spike.initializer.run()
for i in range(1, len(raw_data)):
    if raw_data[i] - raw_data[i-1] > 5:
        updater = tf.assign(spike, True)
        updater.eval()
    else:
        tf.assign(spike, False).eval()
    print("Spike", spike.eval())

sess.close()