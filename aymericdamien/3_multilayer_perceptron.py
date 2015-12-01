__author__ = 'robdefeo'
import input_data
import tensorflow as tf

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

learning_rate = 0.001
training_epochs = 25
batch_size = 100
display_step = 1

# network parameters
n_hidden_1 = 256  # 1st layer num features
n_hidden_2 = 256  # 2nd layer num features
number_of_inputs = 784  # Mnist data input (img shape: 28 * 28)
number_of_classes = 10  # Mnist total calsses 0-9

# graph input
x = tf.placeholder("float", [None, number_of_inputs])
y = tf.placeholder("float", [None, number_of_classes])


# create the model
def multilayer_perceptro(_X, _weights, _biases):
    # hidden layer with RELU activation
    layer_1 = tf.nn.relu(
        tf.add(
            tf.matmul(
                _X,
                _weights['h1']
            ),
            _biases['b1']
        )
    )
    layer_2 = tf.nn.relu(
        tf.add(
            tf.matmul(
                layer_1,
                _weights['h2']
            ),
            _biases['b2']
        )
    )
    return tf.matmul(
        layer_2,
        _weights['out']
    ) + biases['out']


weights = {
    'h1': tf.Variable(tf.random_normal([number_of_inputs, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, number_of_classes]))
}

biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([number_of_classes]))
}

prediction_model = multilayer_perceptro(x, weights, biases)

# define loss and optimizer
# softmax loss
cost = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(prediction_model, y)
)
# adam optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

initilized_variables = tf.initialize_all_variables()

with tf.Session() as session:
    session.run(initilized_variables)

    # training cycle
    for epoch in range(training_epochs):
        average_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)

        for index in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # fit training using the batch data
            session.run(
                optimizer,
                feed_dict={
                    x: batch_xs,
                    y: batch_ys
                }
            )
            # compute average loss
            average_cost += session.run(
                cost,
                feed_dict={
                    x: batch_xs,
                    y: batch_ys
                }
            ) / total_batch

        if epoch % display_step ==0 :
            print "Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(average_cost)

    print("optimizeation finished")

    # test model
    correct_prediction = tf.equal(
        tf.argmax(prediction_model, 1),
        tf.argmax(y, 1)
    )
    # calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print "Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels})