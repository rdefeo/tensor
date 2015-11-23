""" A fully connected MNIST model.

    1. inference() - builds the model required to run the system forward
    2. loss()      - adds layer to generate loss
    3. training()  - add to loss the operations required to generate and apply gradients
"""

from __future__ import absolute_import, division, print_function
import math
import tensorflow.python.platform
import tensorflow as tf

NUM_CLASSES = 10
IMAGE_EDGE_SIZE = 28
IMAGE_PIXEL_SIZE = IMAGE_EDGE_SIZE * IMAGE_EDGE_SIZE


def relu_neural_layer(scope_name, features, tensor_height, tensor_width):

    with tf.name_scope(scope_name) as scope:
        weights = tf.Variable(
            tf.truncated_normal([tensor_height, tensor_width],
                                stddev=1.0 / math.sqrt(float(tensor_height))),
            name='weights')
        biases = tf.Variable(
            tf.zeros([tensor_width]), name='biases')
        hidden = tf.nn.relu(tf.matmul(features, weights) + biases)
        return hidden


def linear_neural_layer(scope_name, features, tensor_height, tensor_width):
    with tf.name_scope(scope_name) as scope:
        weights = tf.Variable(
            tf.truncated_normal([tensor_height, tensor_width],
                                stddev=1.0 / math.sqrt(float(tensor_height))),
            name='weights')
        biases = tf.Variable(
            tf.zeros([tensor_width]), name='biases')
        logits = tf.matmul(features, weights) + biases
        return logits


def inference(images, layer1_size, layer2_size):
    hidden_layer1 = relu_neural_layer('hidden1', images, IMAGE_PIXEL_SIZE, layer1_size)
    hidden_layer2 = relu_neural_layer('hidden2', hidden_layer1, layer1_size, layer2_size)
    softmax_layer = linear_neural_layer('softmax_linear', hidden_layer2, layer2_size, NUM_CLASSES)
    return softmax_layer


def calculate_loss(logits, labels):
    """Calculates the loss from the logits and the labels.
    Args:
        logits: Logits tensor, float - [batch_size, NUM_CLASSES].
        labels: Labels tensor, int32 - [batch_size].
    Returns:
        loss: Loss tensor of type float. 1-hot, dense.
    """
    batch_size = tf.size(labels)
    labels = tf.expand_dims(labels, 1)  # make a batch
    indices = tf.expand_dims(tf.range(0, batch_size), 1)
    concated = tf.concat(1, [indices, labels])  # label/index correspondence
    onehot_labels_dense = tf.sparse_to_dense(
        concated, tf.pack([batch_size, NUM_CLASSES]), 1.0, 0.0)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits,
                                                            onehot_labels_dense,
                                                            name='xentropy')
    loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    return loss


def training(loss, learning_rate):
    """Sets up the training Ops.
    Creates a summarizer to track the loss over time in TensorBoard.
    Creates an optimizer and applies the gradients to all trainable variables.
    The Op returned by this function is what must be passed to the
    `sess.run()` call to cause the model to train.
    Args:
      loss: Loss tensor, from loss().
      learning_rate: The learning rate to use for gradient descent.
    Returns:
      train_op: The Op for training.
    """
    tf.scalar_summary(loss.op.name, loss)
    grad_desc_optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = grad_desc_optimizer.minimize(loss, global_step=global_step)
    return train_op


def evaluation(logits, labels):
    """Evaluate the quality of the logits at predicting the label.
    Args:
      logits: Logits tensor, float - [batch_size, NUM_CLASSES].
      labels: Labels tensor, int32 - [batch_size], with values in the
        range [0, NUM_CLASSES).
    Returns:
      A scalar int32 tensor with the number of examples (out of batch_size)
      that were predicted correctly.
    """
    # For a classifier model, we can use the in_top_k Op.
    # It returns a bool tensor with shape [batch_size] that is true for
    # the examples where the label's is was in the top k (here k=1)
    # of all logits for that example.
    correct = tf.nn.in_top_k(logits, labels, 1)
    # Return the number of true entries.
    return tf.reduce_sum(tf.cast(correct, tf.int32))
