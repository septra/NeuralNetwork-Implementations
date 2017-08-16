import tensorflow as tf
import tqdm
import numpy as np
import seaborn as sns # for nice looking graphs
from matplotlib import pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

np.random.seed(42) 

def create_layer(
        input_tensor,
        weight,
        name,
        activation,
        training=None):
    w = tf.Variable(weight)
    b = tf.Variable(tf.zeros([weight.shape[-1]]))
    z = tf.add(tf.matmul(input_tensor, w), b, name='layer_input_%s' % name)
    if name == 'output':
        return z, activation(z, name='activation_%s' % name)
    else:
        return activation(z, name='activation_%s' % name)

def create_batch_norm_layer(
        input_tensor,
        weight,
        name,
        activation,
        training):
    w = tf.Variable(weight)
    linear_output = tf.matmul(input_tensor, w)
    batch_norm_z = tf.layers.batch_normalization(
            linear_output, training=training, name='bn_layer_input_%s' % name)
    if name == 'output':
        return batch_norm_z, activation(batch_norm_z, name='bn_activation_%s' % name)
    else:
        return activation(batch_norm_z, name='bn_activation_%s' % name)

def get_tensors(
        layer_creation_fn,
        inputs,
        labels,
        weights,
        activation,
        learning_rate,
        is_training):
    l1 = layer_creation_fn(inputs, weights[0], '1', activation, training=is_training)
    l2 = layer_creation_fn(l1, weights[1], '2', activation, training=is_training)
    l3 = layer_creation_fn(l2, weights[2], '3', activation, training=is_training)
    l4 = layer_creation_fn(l3, weights[3], '4', activation, training=is_training)
    l5 = layer_creation_fn(l4, weights[4], '5', activation, training=is_training)
    l6 = layer_creation_fn(l5, weights[5], '6', activation, training=is_training)
    l7 = layer_creation_fn(l6, weights[6], '7', activation, training=is_training)
    l8 = layer_creation_fn(l7, weights[7], '8', activation, training=is_training)
    l9 = layer_creation_fn(l8, weights[8], '9', activation, training=is_training)
    logits, output = layer_creation_fn(
            l9, weights[9], 'output', tf.nn.sigmoid, training=is_training)

    cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))

    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    if layer_creation_fn.__name__ == 'create_batch_norm_layer':
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            optimizer = tf.train.GradientDescentOptimizer(
                            learning_rate).minimize(cross_entropy)
    else:
        optimizer = tf.train.GradientDescentOptimizer(
                        learning_rate).minimize(cross_entropy)

    return accuracy, optimizer


def train_network(
        learning_rate_val,
        num_batches,
        batch_size,
        activation,
        bad_init=False,
        plot_accuracy=True):

    inputs = tf.placeholder(tf.float32, shape=[None, 784], name='inputs')
    labels = tf.placeholder(tf.float32, shape=[None, 10], name='labels')
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')
    is_training = tf.placeholder(tf.bool, name='is_training')

    np.random.seed(42)

    scale = 1 if bad_init else 0.1

    weights = [
            np.random.normal(size=(784, 100), scale=scale).astype(np.float32),
            np.random.normal(size=(100, 100), scale=scale).astype(np.float32),
            np.random.normal(size=(100, 100), scale=scale).astype(np.float32),
            np.random.normal(size=(100, 100), scale=scale).astype(np.float32),
            np.random.normal(size=(100, 100), scale=scale).astype(np.float32),
            np.random.normal(size=(100, 100), scale=scale).astype(np.float32),
            np.random.normal(size=(100, 100), scale=scale).astype(np.float32),
            np.random.normal(size=(100, 100), scale=scale).astype(np.float32),
            np.random.normal(size=(100, 100), scale=scale).astype(np.float32),
            np.random.normal(size=(100, 10), scale=scale).astype(np.float32)]

    vanilla_accuracy, vanilla_optimizer = get_tensors(
        create_layer, 
        inputs, 
        labels, 
        weights, 
        activation, 
        learning_rate, 
        is_training)

    bn_accuracy, bn_optimizer = get_tensors(
        create_batch_norm_layer,
        inputs,
        labels,
        weights,
        activation,
        learning_rate,
        is_training)

    vanilla_accuracy_vals = []
    bn_accuracy_vals = []

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for i in tqdm.tqdm(list(range(num_batches))):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)

            sess.run([vanilla_optimizer], feed_dict={
                inputs: batch_xs,
                labels: batch_ys,
                learning_rate: learning_rate_val,
                is_training: True})

            sess.run([bn_optimizer], feed_dict={
                inputs: batch_xs,
                labels: batch_ys,
                learning_rate: learning_rate_val,
                is_training: True})

            if i % batch_size == 0:
                vanilla_acc = sess.run(vanilla_accuracy, feed_dict={
                    inputs: mnist.validation.images,
                    labels: mnist.validation.labels,
                    is_training: False})

                bn_acc = sess.run(bn_accuracy, feed_dict={
                    inputs: mnist.validation.images,
                    labels: mnist.validation.labels,
                    is_training: False})

                vanilla_accuracy_vals.append(vanilla_acc)
                bn_accuracy_vals.append(bn_acc)

                print(
                    'Iteration: %s; ' % i,
                    'Vanilla Accuracy: %2.4f; ' % vanilla_acc,
                    'BN Accuracy: %2.4f' % bn_acc)
    
    if plot_accuracy:
        plt.title('Training Accuracy')
        plt.plot(range(0, len(vanilla_accuracy_vals) * batch_size, batch_size),
                vanilla_accuracy_vals, label='Vanilla network')
        plt.plot(range(0, len(bn_accuracy_vals) * batch_size, batch_size),
                bn_accuracy_vals, label='Batch Normalized network')
        plt.tight_layout()
        plt.legend()
        plt.grid(True)
        plt.show()


