# from tensorflow.g3doc.tutorials.mnist import input_data
from time import time

import mnist
import tensorflow as tf

# mnist = aymericdamien.input_data.read_data_sets('MNIST_data', one_hot=True)

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate')
flags.DEFINE_integer('max_steps', 2000, 'Number of steps to run trainer.')
flags.DEFINE_integer('hidden1', 128, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 32, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('batch_size', 100, 'Batch size.  '
                                        'Must divide evenly into the dataset sizes.')
flags.DEFINE_string('train_dir', 'MNIST_data', 'Directory to put the training data.')
flags.DEFINE_boolean('fake_data', False, 'If true, uses fake data '
                                         'for unit testing.')


def placeholder_inputs(batch_size):
    """Generate placeholder variables to represent the the input tensors.
    These placeholders are used as inputs by the rest of the model building
    code and will be fed from the downloaded data in the .run() loop, below.
    Args:
      batch_size: The batch size will be baked into both placeholders.
    Returns:
      images_placeholder: Images placeholder.
      labels_placeholder: Labels placeholder.
    """
    images_placeholder = tf.placeholder(
        tf.float32,
        shape=(
            batch_size,
            mnist.IMAGE_PIXELS
        )
    )
    labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))
    return images_placeholder, labels_placeholder


def fill_feed_dict(data_set, images_placeholder, labels_placeholder):
    """Fills the feed_dict for training the given step.
    A feed_dict takes the form of:
    feed_dict = {
        <placeholder>: <tensor of values to be passed for placeholder>,
        ....
    }
    Args:
      data_set: The set of images and labels, from input_data.read_data_sets()
      images_placeholder: The images placeholder, from placeholder_inputs().
      labels_placeholder: The labels placeholder, from placeholder_inputs().
    Returns:
      feed_dict: The feed dictionary mapping from placeholders to values.
    """
    images_feed, labels_feed = data_set.next_batch(
        FLAGS.batch_size,
        FLAGS.fake_data
    )

    return {
        images_placeholder: images_feed,
        labels_placeholder: labels_feed
    }


def do_evaluation(session, eval_correct, images_placeholder, labels_placeholder, data_set):
    """Runs one evaluation against the full epoch of data.
    :type session: Session
    Args:
      session: The session in which the model has been trained.
      eval_correct: The Tensor that returns the number of correct predictions.
      images_placeholder: The images placeholder.
      labels_placeholder: The labels placeholder.
      data_set: The set of images and labels to evaluate, from input_data.read_data_sets().
    """
    # And run one epoch of eval.
    true_count = 0  # Counts the number of correct predictions
    steps_per_epoch = data_set.num_examples // FLAGS.batch_size
    number_of_examples = steps_per_epoch * FLAGS.batch_size
    for step in xrange(steps_per_epoch):
        feed_dict = fill_feed_dict(
            data_set,
            images_placeholder,
            labels_placeholder
        )
        true_count += session.run(
            eval_correct,
            feed_dict=feed_dict
        )
        precision = true_count / number_of_examples
        print(
            '  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
            (number_of_examples, true_count, precision)
        )


def run_training():
    """Train MNIST for a number of steps."""
    # Get the sets of images and labels for training, validation, and test on MNIST.
    data_sets = aymericdamien.input_data.read_data_sets(FLAGS.train_dir, FLAGS.fake_data)

    # Tell TensorFlow that the model will be built into the default Graph.
    with tf.Graph().as_default():
        # Generate placeholders for the images and labels
        images_placeholder, labels_placeholder = placeholder_inputs(FLAGS.batch_size)

        # Build a graph the Ops for loss calculation
        logits = mnist.inference(
            images_placeholder,
            FLAGS.hidden1,
            FLAGS.hidden2
        )

        # Add to the graph ops for loss calculation
        loss = mnist.loss(logits, labels_placeholder)

        # Add to the Graph the Ops that calculate and apply gradients.
        train_op = mnist.training(loss, FLAGS.learning_rate)

        # Add the Op to compare the logits to the labels during evaluation
        eval_correct = mnist.evaluation(logits, labels_placeholder)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.merge_all_summaries()

        # Create a saver for writing training checkpoints
        saver = tf.train.Saver()

        # create a session for running ops on the graph
        session = tf.Session()

        # Run the Op to initialize the variables
        init = tf.initialize_all_variables()
        session.run(init)

        # Instantiate a SummaryWriter to output summaries and the Graph
        summary_writer = tf.train.SummaryWriter(
            FLAGS.train_dir,
            graph_def=session.graph_def
        )

        # After everything is built start the training loop
        for step in xrange(FLAGS.max_steps):
            start_time = time()

            # Fill the feed dictionary with the actual set of images and labels
            # for this training step
            feed_dict = fill_feed_dict(
                data_sets.train,
                images_placeholder,
                labels_placeholder
            )

            # run one step of the model, return values are the activations from
            # the 'train_op' (which is discarded) and the 'loss' Op.  To inspect
            # the values of your Ops or Variables, you may include them in the list
            # passed to the session.Run() and the value tensors will be returned in
            # the tuple from the call
            _, loss_value = session.run(
                [train_op, loss],
                feed_dict=feed_dict
            )

            duration = time() - start_time

            if step % 100 == 0:
                # print status update
                print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
                # Update the events file.
                summary_str = session.run(summary_op, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)

            # Save a checkpoint and evaluate the model periodically
            if (step + 1) % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                saver.save(session, FLAGS.train_dir, global_step=step)
                # Evaluate against the training set.
                print('Training Data Eval:')
                do_evaluation(
                    session,
                    eval_correct,
                    images_placeholder,
                    labels_placeholder,
                    data_sets.train)

                # Evaluate against the validation set.
                print('Validation Data Eval:')
                do_evaluation(
                    session,
                    eval_correct,
                    images_placeholder,
                    labels_placeholder,
                    data_sets.validation)
                # Evaluate against the test set.
                print('Test Data Eval:')
                do_evaluation(
                    session,
                    eval_correct,
                    images_placeholder,
                    labels_placeholder,
                    data_sets.test)


def main(_):
    run_training()


if __name__ == '__main__':
    tf.app.run()
