from os import listdir, mkdir
import input_data
from os.path import exists
import logging
import tensorflow as tf

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)
logging.info('Starting logger for image grabber.')

TRAINING_SET_DIR = '/media/alessio/DATA/ML_workspace/img_out'
TEMP_DATASET_DIR = '/media/alessio/DATA/ML_workspace/data_out'
TF_LOG_DIR = '/media/alessio/DATA/ML_workspace/log'
# Set negative to disable, number of images to browse, actual number of images in
# datasets will be lower depending on FORBIDDEN_ORIENTATIONS defined in input_data.py
MAXIMUM_NUM_IMGS = 1000
MAX_ITERATIONS = 1000  # 20000
RECORDED_SESSION = True


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

if not RECORDED_SESSION:
    LOGGER.info('New session started. Processing directory: "%s"', TRAINING_SET_DIR)
    if not (exists(TRAINING_SET_DIR)) or listdir(TRAINING_SET_DIR) == []:
        print "Training set folder does not exist or is empty."
        exit()

    if MAXIMUM_NUM_IMGS <= 0:
        LOGGER.info("Selected maximum images sample size %i", MAXIMUM_NUM_IMGS)
    dataset = input_data.get_data_sets(TRAINING_SET_DIR, 20, 20, MAXIMUM_NUM_IMGS)

    # Save datasets

    if not (exists(TEMP_DATASET_DIR)):
        mkdir(TEMP_DATASET_DIR)
        print "Created output folder"

    input_data.save_datasets_to_file(dataset, TEMP_DATASET_DIR + "/datasets.plk")

else:
    LOGGER.info('Recorded session option selected. Loading arrays in folder %s', TEMP_DATASET_DIR)
    dataset = input_data.load_datasets_from_file(TEMP_DATASET_DIR + "/datasets.plk")
    LOGGER.info('Dataset loading complete')

# Set up tf logger

if not (exists(TF_LOG_DIR)):
    mkdir(TF_LOG_DIR)
    print "Created log folder"

sess = tf.InteractiveSession()

x = tf.placeholder("float", shape=[None, 100, 100])
y_ = tf.placeholder("float", shape=[None, 5])

#  first layer
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
x_image = tf.reshape(x, [-1, 100, 100, 1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# Add summary ops to collect data
w_1st_layer_hist = tf.histogram_summary("weights_1st_layer", W_conv1)
b_1st_layer_hist = tf.histogram_summary("biases_1st_layer", b_conv1)
h_1st_layer_hist = tf.histogram_summary("h_1st_layer", h_conv1)
h_pool_1st_layer_hist = tf.histogram_summary("h_pool_1st_layer", h_pool1)

# second layer
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# Add summary ops to collect data
w_2nd_layer_hist = tf.histogram_summary("weights_2nd_layer", W_conv2)
b_2nd_layer_hist = tf.histogram_summary("biases_2nd_layer", b_conv2)
h_2nd_layer_hist = tf.histogram_summary("h_2nd_layer", h_conv2)
h_pool_2nd_layer_hist = tf.histogram_summary("h_pool_2nd_layer", h_pool2)

# dense layer
W_fc1 = weight_variable([25 * 25 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 25 * 25 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder("float")  # dropout
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Add summary ops to collect data
w_dense_layer_hist = tf.histogram_summary("weights_dense_layer", W_fc1)
b_dense_layer_hist = tf.histogram_summary("biases_dense_layer", b_fc1)
h_dense_layer_hist = tf.histogram_summary("h_dense_layer", h_fc1)

# readout layer
W_fc2 = weight_variable([1024, 5])
b_fc2 = bias_variable([5])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# Add summary ops to collect data
w_final_layer_hist = tf.histogram_summary("weights_final_layer", W_fc2)
b_final_layer_hist = tf.histogram_summary("biases_final_layer", b_fc2)
y_final_layer_hist = tf.histogram_summary("y_final_layer", y_conv)

# train
with tf.name_scope("cross_entropy") as scope:
    cross_entropy = -tf.reduce_sum(y_*tf.log(tf.clip_by_value(y_conv, 1e-10, 1.0)))
    ce_summ = tf.scalar_summary("cross entropy", cross_entropy)

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
accuracy_summary = tf.scalar_summary("accuracy", accuracy)

sess.run(tf.initialize_all_variables())

# Merge all the summaries and write them out to /tmp/mnist_logs
merged = tf.merge_all_summaries()
writer = tf.train.SummaryWriter(TF_LOG_DIR, sess.graph_def)
tf.initialize_all_variables().run()

for i in range(MAX_ITERATIONS):
    batch = dataset.train.next_batch(50)
    if i % 100 == 0:
        result = sess.run([merged, accuracy], feed_dict={
            x: batch[0], y_: batch[1], keep_prob: 1.0})
        summary_str = result[0]
        train_accuracy = result[1]
        writer.add_summary(summary_str, i)
        print "step %d, training accuracy %g" % (i, train_accuracy)
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print "test accuracy %g" % accuracy.eval(feed_dict={
    x: dataset.test.images, y_: dataset.test.labels, keep_prob: 1.0})





