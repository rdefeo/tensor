# A nearest neighbor learning algorithm example using TensorFlow library.
# This example is using the MNIST database of handwritten digits
# (http://yann.lecun.com/exdb/mnist/)

# Author: Aymeric Damien
# Project: https://github.com/aymericdamien/TensorFlow-Examples/


import numpy as np
import input_data
import tensorflow as tf

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# In this example, we limit mnist data
x_training_data, y_training_data = mnist.train.next_batch(5000)
x_test_data, y_test_data = mnist.test.next_batch(200)


# Reshape images to 1D
x_flat_training_data = np.reshape(x_training_data, newshape=(-1, 28 * 28))
x_flat_test_data = np.reshape(x_test_data, newshape=(-1, 28 * 28))

x_train_placeholder = tf.placeholder("float", [None, 784])
x_test_placeholder = tf.placeholder("float", [784])

distance = tf.reduce_sum(tf.abs(tf.add(x_train_placeholder, tf.neg(x_test_placeholder))), reduction_indices=1)

prediction = tf.arg_min(distance, 0)

accuracy = 0.

init = tf.initialize_all_variables()

with tf.Session() as session:
    session.run(init)

    for index in range(len(x_flat_test_data)):
        # get nearest negihbour
        nearest_neighbor_index = session.run(
            prediction,
            feed_dict={
                x_train_placeholder: x_flat_training_data,
                x_test_placeholder: x_flat_test_data[index,:]
            }
        )
        print "Test", index, "Prediction:", np.argmax(y_training_data[nearest_neighbor_index]), \
              "True Class:", np.argmax(y_test_data[index])

        if np.argmax(y_training_data[nearest_neighbor_index]) == np.argmax(y_test_data[index]):
            accuracy += 1./len(x_flat_test_data)

    print "Done!"
    print "Accuracy:", accuracy