__author__ = 'robdefeo'
import tensorflow as tf

a = tf.constant(2)
b = tf.constant(3)

with tf.Session() as session:
    print "a=2, b=3"
    print(session.run(a + b))
    print(session.run(a * b))

a = tf.placeholder(tf.types.int16)
b = tf.placeholder(tf.types.int16)

add = tf.add(a, b)
multiply = tf.add(a, b)

with tf.Session() as session:
    print("addition %s" % session.run(add, feed_dict={a: 3, b: 3}))
    print("multiplication %s" % session.run(multiply, feed_dict={a: 3, b: 3}))

matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2.], [2.]])
product = tf.matmul(matrix1, matrix2)

with tf.Session() as session:
    result = session.run(product)
    print result