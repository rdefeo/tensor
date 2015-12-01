__author__ = 'robdefeo'
import tensorflow as tf

hello = tf.constant("Hellwo")

session = tf.Session()


print session.run(hello)