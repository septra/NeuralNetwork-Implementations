from functools import reduce
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import argparse

def network_builder(num_inputs, node_counts, num_outputs):
    layer_weights_biases = []
    x = tf.placeholder(tf.float32, [None, num_inputs])
    y = tf.placeholder(tf.float32, [None, num_outputs])

    for ix, node_count in enumerate(node_counts):

        if ix == 0:
            prev_output_num = num_inputs
        else:
            prev_output_num = node_counts[ix - 1]

        layer_weight = tf.Variable(tf.truncated_normal([prev_output_num, node_count]))
        layer_bias = tf.Variable(tf.zeros(node_count))

        layer_weights_biases.append((layer_weight, layer_bias))

    pre_output_activation = reduce(
            lambda acc, layer_bw: tf.nn.relu(tf.add(tf.matmul(acc, layer_bw[0]), layer_bw[1])), 
            layer_weights_biases, 
            x)

    output_weight = tf.Variable(tf.truncated_normal([node_counts[-1], num_outputs]))
    output_bias = tf.Variable(tf.zeros(num_outputs))

    output = tf.add(tf.matmul(pre_output_activation, output_weight), output_bias)

    return x, y, output


# Original by-hand construction of network architecture:
#
#x = tf.placeholder(tf.float32, [None, 784])
#y = tf.placeholder(tf.float32, [None, 10])
#
#layer1_weight = tf.Variable(tf.truncated_normal([784, 20]))
#layer1_bias = tf.Variable(tf.zeros(20))
#
#layer2_weight = tf.Variable(tf.truncated_normal([20, 22]))
#layer2_bias = tf.Variable(tf.zeros(22))
#
#output_weight = tf.Variable(tf.truncated_normal([22, 10]))
#output_bias = tf.Variable(tf.zeros(10))
#
#layer1_activation = tf.nn.relu(tf.add(tf.matmul(x, layer1_weight), layer1_bias))
#layer2_activation = tf.nn.relu(tf.add(tf.matmul(layer1_activation, layer2_weight), layer2_bias))
#
#output = tf.add(tf.matmul(layer2_activation, output_weight), output_bias)


if __name__ == '__main__':
    description = 'Deep network for classifying MNIST images.'

    parser = argparse.ArgumentParser(description = description)

    parser.add_argument(
            '-e', '--epochs', type = int, help = 'Number of Epochs for training the network.')

    parser.add_argument(
            '-r', '--learnrate', type = float, help = 'Learning Rate used in training.')

    parser.add_argument(
            '-l', 
            '--layers', 
            type = lambda x: [int(count) for count in x.split(',')], 
            help = 'Comma-delimited list of the number of nodes in the hidden layers.')

    parser.add_argument(
            '-b', '--batchsize', type = int, help = 'Total batch size used in training.')

    args = parser.parse_args()

    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    x, y, output = network_builder(784, args.layers, 10)

    learn_rate = tf.placeholder(tf.float32)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y))
    optimizer = tf.train.AdamOptimizer(learn_rate).minimize(cost)
    correct_preds = tf.equal(tf.argmax(y, 1), tf.argmax(output, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_preds, tf.float32))

    epochs = args.epochs
    learn_rate_value = args.learnrate
    batch_size = args.batchsize

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        print('''
            Running session with 
            \tEpochs: {:d}
            \tLearning Rate: {:.5f}
            \tBatch Size: {:d}
            \tNum of hidden layers (nodes): {:d} {}\n'''.format(
                epochs, learn_rate_value, batch_size, len(args.layers), args.layers))

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

            print('Epoch: {:>5} '
                  'Loss: {:>10.4f} Validation Accuracy: {:.4f}'.format(
                epoch + 1,
                loss,
                valid_acc))

        test_acc = sess.run(accuracy, feed_dict= {
                                        x: mnist.test.images,
                                        y: mnist.test.labels})
        print('Testing Accuracy: {:.3f}'.format(test_acc))



