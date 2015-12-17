from os import listdir, mkdir
import input_data
from os.path import exists
import logging
import tensorflow as tf

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)
logging.info('Starting logger for image grabber.')

TRAINING_SET_DIR = '../grabber/out'
TEMP_DATASET_DIR = 'out'

RECORDED_SESSION = False


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


# Script start

dataset = input_data.DataSets()

if not RECORDED_SESSION:
    if not (exists(TRAINING_SET_DIR)) or listdir(TRAINING_SET_DIR) == []:
        print "Training set folder does not exist or is empty."
        exit()

    dataset = input_data.read_data_sets(TRAINING_SET_DIR, 20, 20)

    # Save datasets

    if not (exists(TEMP_DATASET_DIR)):
        mkdir(TEMP_DATASET_DIR)
        print "Created output folder"

    input_data.save_dataset_array_to_file(dataset.train, TEMP_DATASET_DIR + "/train_set")
    input_data.save_dataset_array_to_file(dataset.validation, TEMP_DATASET_DIR + "/validation")
    input_data.save_dataset_array_to_file(dataset.test, TEMP_DATASET_DIR + "/test")

else:

    dataset.train = input_data.load_dataset_from_file(TEMP_DATASET_DIR + "/train_set")
    dataset.validation = input_data.load_dataset_from_file(TEMP_DATASET_DIR + "/validation")
    dataset.test = input_data.load_dataset_from_file(TEMP_DATASET_DIR + "/test")

sess = tf.InteractiveSession()

x = tf.placeholder("float", shape=[None, 10000])
y_ = tf.placeholder("float", shape=[None, 5])
W = tf.Variable(tf.zeros([10000, 5]))
b = tf.Variable(tf.zeros([5]))
sess.run(tf.initialize_all_variables())

y = tf.nn.softmax(tf.matmul(x, W) + b)

cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

for i in range(1000):
    batch = dataset.train.next_batch(50)
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print accuracy.eval(feed_dict={x: dataset.test.images, y_: dataset.test.labels})

# first layer
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
x_image = tf.reshape(x, [-1, 100, 100, 1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# second layer
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# dense layer
W_fc1 = weight_variable([25 * 25 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 25 * 25 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder("float")  # dropout
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# readout layer
W_fc2 = weight_variable([1024, 5])
b_fc2 = bias_variable([5])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# train
cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess.run(tf.initialize_all_variables())
for i in range(20000):
    batch = dataset.train.next_batch(50)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x: batch[0], y_: batch[1], keep_prob: 1.0})
        print "step %d, training accuracy %g" % (i, train_accuracy)
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print "test accuracy %g" % accuracy.eval(feed_dict={
    x: dataset.test.images, y_: dataset.test.labels, keep_prob: 1.0})





