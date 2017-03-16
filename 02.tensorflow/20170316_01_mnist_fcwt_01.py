"""
First Contact with Tensorflow 실습
단일 레이어 뉴럴 네트워크 구성
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets("./mnist_data", one_hot=True)

# 데이터 체크
print(mnist.train.images)
print(mnist.train.images.shape)
print(mnist.train.labels)
print(mnist.train.labels.shape)

print(mnist.test.images.shape)

x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

#h = tf.matmul(x, W) + b
#y = tf.nn.softmax(h)
#cost_cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
y = tf.matmul(x, W) + b
cost_cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

optimizer = tf.train.GradientDescentOptimizer(0.01)
train_step = optimizer.minimize(cost_cross_entropy)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

costs = []
for i in range(10000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    if i % 10 == 0:
        costs.append(sess.run(cost_cross_entropy, feed_dict={x: batch_xs, y_: batch_ys}))
        print(i
              , costs[-1]
              , sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys}))

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
plt.plot(range(len(costs)), costs)
plt.show()
