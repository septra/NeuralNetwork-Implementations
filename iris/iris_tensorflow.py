from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd

def print_epoch_stats(epoch_i, sess, last_features, last_labels):
    """
    Print cost and validation accuracy of an epoch
    """
    current_cost = sess.run(
        cost,
        feed_dict={features: last_features, labels: last_labels})
    print('Epoch: {:<4} - Cost: {:<8.3} '.format(
        epoch_i,
        current_cost))


data = load_iris()

X = data.data
y = data.target

y_one_hot = pd.get_dummies(y).values


X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)


# no of input features = 4
# no of hidden nodes = 5
# no of output nodes = 1


n_features = 4
n_labels = 3
n_hidden = 9

features = tf.placeholder(tf.float32)
labels = tf.placeholder(tf.float32)

hidden_w = tf.Variable(tf.truncated_normal((n_features, n_hidden)))
output_w = tf.Variable(tf.truncated_normal((n_hidden, n_labels)))

hidden_b = tf.Variable(tf.zeros(n_hidden))
output_b = tf.Variable(tf.zeros(n_labels))

hidden_layer = tf.nn.relu(tf.add(tf.matmul(features, hidden_w), hidden_b))
logits = tf.add(tf.matmul(hidden_layer, output_w), output_b)

learning_rate = tf.placeholder(tf.float32)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# Calculate accuracy
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()

epochs = 10000
learn_rate = 0.01

with tf.Session() as sess:

    sess.run(init)

    for epoch_i in range(epochs):
        train_feed_dict = {
                features: X_train,
                labels: y_train,
                learning_rate: learn_rate}
        sess.run(optimizer, feed_dict=train_feed_dict)

    print_epoch_stats(epoch_i, sess, X_train, y_train)

    test_accuracy = sess.run(
            accuracy,
            feed_dict={features: X_test, labels: y_test})

    output = sess.run(tf.argmax(logits, 1), feed_dict={features: X_test})

print('Test Accuracy: {}'.format(test_accuracy))
