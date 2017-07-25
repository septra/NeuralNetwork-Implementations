import numpy as np
import tensorflow as tf
import os
import sys
import time
from urllib.request import urlretrieve
from collections import Counter
import string

FILENAME = '2600-0.txt'
URL = 'http://www.gutenberg.org/files/2600/'

if not os.path.exists(FILENAME):
    "File {} not found, downloading...".format(FILENAME)
    FILENAME, _ = urlretrieve(URL + FILENAME, FILENAME)

with open(FILENAME, 'rb') as f:
    data = str(f.read())

vocab = Counter(data)

vocab = [n for n in vocab if 
              n in string.ascii_letters or
              n in string.digits or
              n in string.punctuation] + [' ']

char_to_int = { char: ix for ix, char in enumerate(vocab) }
int_to_char = { ix: char for char, ix in char_to_int.items() }

encoded = [char_to_int[char] for char in data]


## Batching code
def get_batch(raw, n_seqs, n_steps):
    raw = np.array(raw)

    chars_in_minibatch = n_seqs * n_steps
    num_minibatch = len(raw) // chars_in_minibatch

    raw = raw[: num_minibatch * chars_in_minibatch]

    raw = raw.reshape((n_seqs, -1))

    for n in range(0, raw.shape[1], n_steps):
        x = raw[:, n:n+n_steps]

        y = np.zeros_like(x)
        y[:, :-1], y[:, -1] = x[:, 1:], x[:, 0]

        yield x, y


###################### Sampling code
def pick_top_n(preds, vocab_size, top_n=5):
    p = np.squeeze(preds)
    p[np.argsort(p)[:-top_n]] = 0
    p = p / np.sum(p)
    c = np.random.choice(vocab_size, 1, p=p)[0]
    return c

def sample(checkpoint, n_samples, lstm_size, vocab_size, prime="The "):
    samples = [c for c in prime]
         
    # Custom code start
    batch_size, chars_in_batch = 1, 1
    tf.reset_default_graph()
    inputs = tf.placeholder(
        tf.int32,
        [batch_size, chars_in_batch],
        name='inputs')
    targets = tf.placeholder(
        tf.int32,
        [batch_size, chars_in_batch],
        name='targets')
    keep_prob = tf.placeholder(tf.float32, name='dropout')

    cell = tf.contrib.rnn.MultiRNNCell(
            [build_cell(lstm_size, keep_prob) for _ in range(num_layers)])
    initial_state = cell.zero_state(batch_size, tf.float32)

    x_one_hot = tf.one_hot(inputs, num_classes)


    outputs, state = tf.nn.dynamic_rnn(
                            cell, 
                            x_one_hot, 
                            initial_state=initial_state)
    final_state = state

    ## Output
    rnn_output = tf.concat(outputs, axis=1)
    x = tf.reshape(rnn_output, [-1, lstm_size])

    with tf.variable_scope('softmax'):
        softmax_w = tf.Variable(
                tf.truncated_normal([lstm_size, num_classes], stddev=0.1))
        softmax_b = tf.Variable(tf.zeros(num_classes))

    logits = tf.add(tf.matmul(x, softmax_w), softmax_b)
    prediction = tf.nn.softmax(logits, name='predictions')

    # Custom code end

    #model = CharRNN(len(vocab), lstm_size=lstm_size, sampling=True)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, checkpoint)
        new_state = sess.run(initial_state)
        for c in prime:
            x = np.zeros((1, 1))
            x[0,0] = char_to_int[c]
            feed = {inputs: x,
                    keep_prob: 1.,
                    initial_state: new_state}
            preds, new_state = sess.run([prediction, final_state], 
                                         feed_dict=feed)

        c = pick_top_n(preds, len(vocab))
        samples.append(int_to_char[c])

        for i in range(n_samples):
            x[0,0] = c
            feed = {inputs: x,
                    keep_prob: 1.,
                    initial_state: new_state}
            preds, new_state = sess.run([prediction, final_state], 
                                         feed_dict=feed)

            c = pick_top_n(preds, len(vocab))
            samples.append(int_to_char[c])
        
    return ''.join(samples)
############################



# Hyper-parameters
batch_size = 10
chars_in_batch = 100
lstm_size = 256
epochs = 10
keep_prob_val = 0.5
learning_rate = 0.01
num_layers = 2
num_classes = len(vocab)


tf.reset_default_graph()

####### Build the network

inputs = tf.placeholder(tf.int32, [batch_size, chars_in_batch], name='inputs')

targets = tf.placeholder(tf.int32, [batch_size, chars_in_batch], name='targets')

keep_prob = tf.placeholder(tf.float32, name='dropout')

## Build lstm cells
def build_cell(units, keep_prob):
    lstm = tf.contrib.rnn.BasicLSTMCell(units)
    dropout_applied_lstm = tf.contrib.rnn.DropoutWrapper(
                                lstm, 
                                output_keep_prob=keep_prob)

    return dropout_applied_lstm

cell = tf.contrib.rnn.MultiRNNCell(
        [build_cell(lstm_size, keep_prob) for _ in range(num_layers)])
initial_state = cell.zero_state(batch_size, tf.float32)

x_one_hot = tf.one_hot(inputs, num_classes)


outputs, state = tf.nn.dynamic_rnn(
                        cell, 
                        x_one_hot, 
                        initial_state=initial_state)
final_state = state

## Output
rnn_output = tf.concat(outputs, axis=1)
x = tf.reshape(rnn_output, [-1, lstm_size])

with tf.variable_scope('softmax'):
    softmax_w = tf.Variable(
            tf.truncated_normal([lstm_size, num_classes], stddev=0.1))
    softmax_b = tf.Variable(tf.zeros(num_classes))

logits = tf.add(tf.matmul(x, softmax_w), softmax_b)
prediction = tf.nn.softmax(logits, name='predictions')


# Loss
y_one_hot = tf.one_hot(targets, num_classes)
y_reshaped = tf.reshape(y_one_hot, logits.get_shape())

loss = tf.nn.softmax_cross_entropy_with_logits(
        logits=logits,
        labels=y_reshaped)

loss = tf.reduce_mean(loss)


# Optimizer
grad_clip = 5
tvars = tf.trainable_variables()
grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), grad_clip)
train_op = tf.train.AdamOptimizer(learning_rate)
optimizer = train_op.apply_gradients(zip(grads, tvars))

######## Network Built



checkpoint_every = 200 # Checkpoint at every 200th iteration of training.

saver = tf.train.Saver(max_to_keep=100)

# Train the model
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # load previous checkpoints and resume training.
    # saver.restore(sess, 'checkpoints/___.ckpt')

    counter = 0

    for e in range(epochs):
        # Training
        new_state = sess.run(initial_state)
        for x, y in get_batch(encoded, batch_size, chars_in_batch):
            counter += 1
            start = time.time()

            feed_dict = {
                    inputs: x,
                    targets: y, 
                    keep_prob: keep_prob_val,
                    initial_state: new_state}

            batch_loss, new_state, _ = sess.run([
                                        loss,
                                        final_state,
                                        optimizer],
                                        feed_dict = feed_dict)

            end = time.time()
            print('Epoch: {}/{}... '.format(e+1, epochs),
                  'Training Step: {}... '.format(counter),
                  'Training loss: {:.4f}... '.format(batch_loss),
                  '{:.4f} sec/batch'.format((end-start)))

        
            if (counter % checkpoint_every == 0):
                saver.save(
                        sess, 
                        "checkpoints/i{}_l{}.ckpt".format(counter, lstm_size))

    saver.save(sess, "checkpoints/i{}_l{}.ckpt".format(counter, lstm_size))

#### Training End


tf.train.get_checkpoint_state('checkpoints')


## Test Checkpoints

#checkpoint = tf.train.latest_checkpoint('checkpoints')
#samp = sample(checkpoint, 2000, lstm_size, len(vocab), prime="Man")
#print(samp)
#
#
#checkpoint = 'checkpoints/i200_l128.ckpt'
#samp = sample(checkpoint, 1000, lstm_size, len(vocab), prime="Far")
#print(samp)
