import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/")

x = tf.placeholder(tf.float32, shape=[ None, 784], name='inputs')
y = tf.placeholder(tf.int32, shape=[ None ], name='labels')

# Hyperparameters
epochs = 20
learning_rate = 0.01
batch_size = 128

##### Build Data Pipeline #####
def filter_condition(x): 
    return tf.less(x['labels'], 3)

training_data = tf.data.Dataset.from_tensor_slices({
    'features': mnist.train.images, 
    'labels': mnist.train.labels}) \
            .filter(filter_condition) \
            .shuffle(buffer_size=5000) \
            .batch(batch_size)

filtered_validation_ixs = np.where(mnist.validation.labels < 3)[0]
validation_data = tf.data.Dataset.from_tensor_slices({
    'features': mnist.validation.images[filtered_validation_ixs],
    'labels': mnist.validation.labels[filtered_validation_ixs]}) \
            .batch(len(filtered_validation_ixs)) \
            .repeat()

filtered_test_ixs = np.where(mnist.test.labels < 3)[0]
test_data = tf.data.Dataset.from_tensor_slices({
    'features': mnist.test.images[filtered_test_ixs],
    'labels': mnist.test.labels[filtered_test_ixs]}) \
            .batch(len(filtered_test_ixs))

training_iterator = training_data.make_initializable_iterator()
next_training_batch = training_iterator.get_next()

validation_iterator = validation_data.make_one_shot_iterator()
next_validation_batch = validation_iterator.get_next()

test_iterator = test_data.make_one_shot_iterator()
next_test_batch = test_iterator.get_next()
#####

##### Begin Model Definition
with tf.name_scope("restorable"):
    # Layer 1
    with tf.name_scope("layer_1"):
        w1 = tf.Variable(tf.truncated_normal(shape=(784, 10), mean=0, stddev=0.01), name='weights', trainable=False)
        b1 = tf.Variable(tf.zeros([10]), name='biases', trainable=False)
        linear_1 = tf.nn.xw_plus_b(x, w1, b1, name='combination')
        act_1 = tf.nn.relu(linear_1, name='activation')

    # Layer 2
    with tf.name_scope("layer_2"):
        w2 = tf.Variable(tf.truncated_normal(shape=(10, 10), mean=0, stddev=0.01), name='weights', trainable=False)
        b2 = tf.Variable(tf.zeros([10]), name='biases', trainable=False)
        linear_2 = tf.nn.xw_plus_b(act_1, w2, b2, name='combination')
        act_2 = tf.nn.relu(linear_2, name='activation')

# Layer 3
with tf.name_scope("layer_3_new"):
    w3 = tf.Variable(tf.truncated_normal(shape=(10, 3), mean=0, stddev=1), name='weights')
    b3 = tf.Variable(tf.zeros([3]), name='biases')
    linear_3 = tf.nn.xw_plus_b(act_2, w3, b3, name='combination')
    #act_2 = tf.nn.softmax(linear_3, name='activation')
    #act_3 = tf.nn.relu(linear_3)

with tf.name_scope("results"):
    logits = tf.identity(linear_3, name='logits')
    predictions = tf.nn.softmax(logits, name='predictions')

with tf.name_scope('hyperparameters'):
    lr = tf.placeholder(tf.float32)

with tf.name_scope('training'):
    with tf.name_scope('loss'):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=y)
        total_loss = tf.reduce_mean(cross_entropy)

    optimizer = tf.train.AdamOptimizer(lr)
    train_opt = optimizer.minimize(cross_entropy)

with tf.name_scope('accuracy'):
    correct_predictions = tf.equal(
            tf.argmax(predictions, 1, output_type=tf.int32), y)
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))


with tf.Session() as sess:
    init_vars = [var 
            for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            if not var.name.startswith("restorable")]

    sess.run(tf.variables_initializer(var_list = init_vars))

    saver = tf.train.Saver(var_list={
        "layer_1/weights":w1,
        "layer_1/biases" :b1,
        "layer_2/weights":w2,
        "layer_2/biases" :b2})
    saver.restore(sess, "checkpoint/model.ckpt")

    print(f'''
        Running session with:
        Epochs: {epochs:>3d}
        Learning Rate: {learning_rate:>6.3f}
        Batch Size: {batch_size:>3d}''')

    for epoch in range(epochs):
        sess.run(training_iterator.initializer)

        while True:
            try:
                training_data = sess.run(next_training_batch)

                feed_dict = {
                        x : training_data['features'],
                        y : training_data['labels'],
                        lr : learning_rate }

                _, loss = sess.run([train_opt, total_loss], feed_dict = feed_dict)

            # Calculate validation accuracy every epoch.
            except tf.errors.OutOfRangeError:
                validation_data = sess.run(next_validation_batch)
                valid_acc = sess.run(accuracy, feed_dict = {
                    x : validation_data['features'],
                    y : validation_data['labels']})

                print(f'Epoch: {epoch:>5d}; Loss: {loss: >10.3f}; Validation Accuracy: {valid_acc:>1.4f}')
                break

    test_data = sess.run(next_test_batch)
    test_accuracy = sess.run(accuracy, feed_dict = {
        x : test_data['features'],
        y : test_data['labels']})

    print(f'Final test accuracy: {test_accuracy:>2.2f}')

    saver.save(sess, 'checkpoint/transfer_model.ckpt')
    #file_writer = tf.summary.FileWriter('./logs/1', sess.graph)
