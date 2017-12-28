import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

# Get MNIST data without using the `one_hot = True` flag as
# we will be using the `tf.nn.sparse_softmax_cross_entropy_with_logits` function.
mnist = input_data.read_data_sets("MNIST_data/")

x = tf.placeholder(tf.float32, shape=[ None, 784], name='inputs')
# Use dtype as `tf.int32` specifying output type from `tf.argmax` to be `tf.int32`
# (It defaults to `tf.int64`)
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

# We use the initializable iterator as we'll re-run through all the batches
# in the training dataset, initializing the spent iterator every epoch.
training_iterator = training_data.make_initializable_iterator()
next_training_batch = training_iterator.get_next()

# Since the validation set as a whole will be used over and over 
# every epoch, we specified the batch_size to be the entire dataset size
# and made it `repeat()` again and again.
validation_iterator = validation_data.make_one_shot_iterator()
next_validation_batch = validation_iterator.get_next()

# We'll use the test dataset only once so make a one-shot iterator
# with the batch_size as the entire dataset.
test_iterator = test_data.make_one_shot_iterator()
next_test_batch = test_iterator.get_next()
#####

##### Begin Model Definition
# Use the _restorable_ name scope as these variables should not be initialized
# and we will use this name scope to filter them out.
# Also, we set `trainable = False` as we are using these layers for feature extraction
# and not to fine-tune the network. If you'd like to fine-tune the network, then set 
# `trainable = True`.
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
# We use 3 output nodes instead of 10.
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
    # Filter out the graph for variables which will be restored from previously
    # trained models and thus should not be initialized.
    init_vars = [var 
            # for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            for var in tf.global_variables()
            if not var.name.startswith("restorable")]

    # Using `tf.variable_initializer` and passing in the list of variables
    # we want initialized instead of using `tf.global_variables_initializer`.
    sess.run(tf.variables_initializer(var_list = init_vars))

    # Restore model weights from the previously trained network.
    # The keys in the dictionary passed to `var_list` represent the variable
    # name in the previously trained networks and the values represent the 
    # variables these weights should be assigned to. 
    # We could have used a list of names if the variables names in the current network
    # were the same as those in the previous network but as we used the _restorable_
    # namespace, the correspondence was broken.
    # See: https://www.tensorflow.org/programmers_guide/saved_model

    # Another alternative is to run the following snippet instead of the code above
    # All the variables are initialized and then the _restorable_ variables are 
    # overwritten with the previously trained model weights when we restore the model
    # weights using `tf.train.Saver`
    # See: https://stackoverflow.com/questions/41621071/restore-subset-of-variables-in-tensorflow
    """
    sess.run(tf.global_variables_initializer())
    """

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

            # Calculate validation accuracy every epoch once the iterator has exhausted all batches.
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
