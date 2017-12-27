import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Load MNIST dataset
mnist = input_data.read_data_sets("MNIST_data/")

# Define placeholders for dataset inputs (for feeding in training, validation and testing sets)
# This provides a convenient way to apply the same filter/map operations on 
# the train/validation/test sets. We are making use of the one_shot iterator here.
# See: https://www.tensorflow.org/programmers_guide/datasets

# Define placeholders for the inputs and labels
x = tf.placeholder(tf.float32, shape=[ None, 784], name='inputs')
y = tf.placeholder(tf.int64, shape=[ None], name='labels')

# Hyperparameters
epochs = 20
learning_rate = 0.01
batch_size = 128

##### Build Data Pipeline #####
training_dataset = tf.data.Dataset.from_tensor_slices({
    'features': mnist.train.images, 
    'labels'  : mnist.train.labels}) \
            .shuffle(buffer_size=10000) \
            .batch(batch_size) \

# Repeat entire validation dataset indefinitely
validation_dataset = tf.data.Dataset.from_tensor_slices({
    'features': mnist.validation.images, 
    'labels'  : mnist.validation.labels}) \
            .batch(mnist.validation.images.shape[0]) \
            .repeat()

test_dataset = tf.data.Dataset.from_tensor_slices({
    'features': mnist.test.images, 
    'labels'  : mnist.test.labels}) \
            .batch(mnist.test.images.shape[0]) \

training_iterator = training_dataset.make_initializable_iterator()
next_training_batch = training_iterator.get_next()

validation_iterator = validation_dataset.make_one_shot_iterator()
next_validation_batch = validation_iterator.get_next()

test_iterator = test_dataset.make_one_shot_iterator()
next_test_batch = test_iterator.get_next()
#####

##### Begin Model Definition
# Layer 1
with tf.name_scope("layer_1"):
    w1 = tf.Variable(tf.truncated_normal(shape=(784, 10), mean=0, stddev=0.1), name='weights')
    b1 = tf.Variable(tf.zeros([10]), name='biases')
    linear_1 = tf.nn.xw_plus_b(x, w1, b1, name='combination')
    act_1 = tf.nn.relu(linear_1, name='activation')

# Layer 2
with tf.name_scope("layer_2"):
    w2 = tf.Variable(tf.truncated_normal(shape=(10, 10), mean=0, stddev=0.1), name='weights')
    b2 = tf.Variable(tf.zeros([10]), name='biases')
    linear_2 = tf.nn.xw_plus_b(act_1, w2, b2, name='combination')
    act_2 = tf.nn.relu(linear_2, name='activation')

# Layer 3
with tf.name_scope("layer_3"):
    w3 = tf.Variable(tf.truncated_normal(shape=(10, 10), mean=0, stddev=0.1), name='weights')
    b3 = tf.Variable(tf.zeros([10]), name='biases')
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
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
        total_loss = tf.reduce_mean(cross_entropy)

    optimizer = tf.train.AdamOptimizer(lr)
    train_opt = optimizer.minimize(cross_entropy)

with tf.name_scope('accuracy'):
    correct_predictions = tf.equal(tf.argmax(predictions, 1), y)
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

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

    saver = tf.train.Saver()
    saver.save(sess, 'checkpoint/model.ckpt')
    file_writer = tf.summary.FileWriter('./logs/1', sess.graph)
