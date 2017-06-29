from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# Define placeholders
x = tf.placeholder(tf.float32, shape=[None, 784])
y = tf.placeholder(tf.float32, shape=[None, 10])
learn_rate = tf.placeholder(tf.float32)

reshaped_x = tf.reshape(x, [-1, 28, 28, 1])

# Use initialization recommended by He et al.
# Have to double-check whether this is the right approach.
def w_init(n):
    return np.sqrt(2.0/n)

# Define variables
cnn_layer_depth1 = 16
cnn_layer_depth2 = 32

W1 = tf.Variable(tf.truncated_normal(shape=[3, 3, 1, cnn_layer_depth1], stddev=w_init(3*3*1)))
b1 = tf.Variable(tf.constant(0.1, shape=[cnn_layer_depth1]))

W2 = tf.Variable(tf.truncated_normal(shape=[3, 3, cnn_layer_depth1, cnn_layer_depth2], stddev=w_init(3*3*cnn_layer_depth1)))
b2 = tf.Variable(tf.constant(0.1, shape=[cnn_layer_depth2]))

fcW1 = tf.Variable(tf.truncated_normal(shape=[7 * 7 * cnn_layer_depth2, 1024], stddev=w_init(7*7*cnn_layer_depth2)))
fcb1 = tf.Variable(tf.constant(0.1, shape=[1024]))

fcW2 = tf.Variable(tf.truncated_normal(shape=[1024, 512], stddev=w_init(1024)))
fcb2 = tf.Variable(tf.constant(0.1, shape=[512]))

outputW = tf.Variable(tf.truncated_normal(shape=[512, 10], stddev=w_init(512)))
outputb = tf.Variable(tf.constant(0.1, shape=[10]))


# Build Network Graph
conv1 = tf.nn.conv2d(reshaped_x, W1, strides=[1,1,1,1], padding='SAME')
conv1act = tf.nn.relu(tf.nn.bias_add(conv1, b1))

maxpool1 = tf.nn.max_pool(conv1act, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

conv2 = tf.nn.conv2d(maxpool1, W2, strides=[1,1,1,1], padding='SAME')
conv2act = tf.nn.relu(tf.nn.bias_add(conv2, b2))

maxpool2 = tf.nn.max_pool(conv2act, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

flattened_layer = tf.reshape(maxpool2, [-1, 7 * 7 * cnn_layer_depth2])

fc1 = tf.add(tf.matmul(flattened_layer, fcW1), fcb1)
fc1act = tf.nn.relu(fc1)

fc2 = tf.add(tf.matmul(fc1act, fcW2), fcb2)
fc2act = tf.nn.relu(fc2)

output = tf.add(tf.matmul(fc2act, outputW), outputb)


# Define Hyperparameters
batch_size = 128
epochs = 10
learn_rate_value = 0.0001


cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y))
optimizer = tf.train.AdamOptimizer(learn_rate).minimize(cost)
correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# Run session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(epochs):
        for batch in range(mnist.train.num_examples//batch_size):
            x_train, y_train = mnist.train.next_batch(batch_size)

            sess.run(optimizer, feed_dict = {
                x: x_train,
                y: y_train,
                learn_rate: learn_rate_value})

            if batch % 100 == 0:
                loss = sess.run(cost, feed_dict = {
                    x: x_train,
                    y: y_train})

                train_acc = sess.run(accuracy, feed_dict={
                    x: x_train, 
                    y: y_train})

                valid_acc = sess.run(accuracy, feed_dict={
                    x: mnist.validation.images,
                    y: mnist.validation.labels})

                print(
                    'Epoch: {:>3d}\t'
                    'Batch: {:>4d}\t'
                    'Loss: {:4.4f}\t'
                    'Train Accuracy: {:>10.4f}\t'
                    'Validation Accuracy: {:10.4f}'.format(
                        epoch, batch, loss, train_acc, valid_acc))


    print('Final Test Accuracy: {:10.4f}'.format(sess.run(accuracy, feed_dict = {
        x: mnist.test.images, y: mnist.test.labels})))

