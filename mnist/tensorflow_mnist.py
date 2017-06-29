from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

layer1_weight = tf.Variable(tf.truncated_normal([784, 20]))
layer1_bias = tf.Variable(tf.zeros(20))

layer2_weight = tf.Variable(tf.truncated_normal([20, 22]))
layer2_bias = tf.Variable(tf.zeros(22))

output_weight = tf.Variable(tf.truncated_normal([22, 10]))
output_bias = tf.Variable(tf.zeros(10))

layer1_activation = tf.nn.relu(tf.add(tf.matmul(x, layer1_weight), layer1_bias))
layer2_activation = tf.nn.relu(tf.add(tf.matmul(layer1_activation, layer2_weight), layer2_bias))

output = tf.add(tf.matmul(layer2_activation, output_weight), output_bias)

learn_rate = tf.placeholder(tf.float32)


cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y))
optimizer = tf.train.AdamOptimizer(learn_rate).minimize(cost)
correct_preds = tf.equal(tf.argmax(y, 1), tf.argmax(output, 1))
accuracy = tf.reduce_mean(tf.cast(correct_preds, tf.float32))

epochs = 80
learn_rate_value = 0.001
batch_size = 256

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(epochs):
        for batch in range(mnist.train.num_examples//batch_size):
            x_train, y_train = mnist.train.next_batch(batch_size)
            
            feed_dict = {
                    x: x_train,
                    y: y_train,
                    learn_rate: learn_rate_value}

            sess.run(optimizer, feed_dict = feed_dict)
        
            loss = sess.run(cost, feed_dict= {
                                            x: x_train,
                                            y: y_train})
        valid_acc = sess.run(accuracy, feed_dict= {
                                        x: mnist.validation.images,
                                        y: mnist.validation.labels})

        print('Epoch {:>2},'
              'Loss: {:>10.4f} Validation Accuracy: {:.6f}'.format(
            epoch + 1,
            loss,
            valid_acc))

    test_acc = sess.run(accuracy, feed_dict= {
                                    x: mnist.test.images,
                                    y: mnist.test.labels})
    print('Testing Accuracy: {}'.format(test_acc))
