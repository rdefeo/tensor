__author__ = 'robdefeo'
import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

import tensorflow as tf

learning_rate = 0.01
training_epochs = 25
batch_size = 100
display_step = 1

# tf graph input
x = tf.placeholder("float", [None, 784])
y = tf.placeholder("float", [None, 10])

# set model weights
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# construct the model
activation = tf.nn.softmax(tf.matmul(x, W) + b)

# minimise error using cross entropy
cost = -tf.reduce_sum(y * tf.log(activation))
# gradient descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init = tf.initialize_all_variables()

with tf.Session() as session:
    session.run(init)

    # training cylcle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples / batch_size)

        # loop over batch
        for index in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # fit the training using the batch data
            session.run(
                optimizer,
                feed_dict={
                    x: batch_xs,
                    y: batch_ys
                }
            )
            # compute average loss
            avg_cost += session.run(
                cost,
                feed_dict={
                    x: batch_xs,
                    y: batch_ys
                }
            ) / total_batch

            # display logs per epoch step
            if epoch % display_step == 0:
                print "Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost)

    print("optimise finished")

    # test model
    correct_prediction = tf.equal(
        tf.argmax(activation, 1),
        tf.argmax(y, 1)
    )

    # calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print "Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels})