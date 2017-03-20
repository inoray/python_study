import pandas as pd
import numpy as np
import tensorflow as tf

# csv 읽어오기
df_train = pd.read_csv('data/train.csv', header=0)
df_test = pd.read_csv('data/test.csv', header=0)

"""
print(df_train)

print(df_train["Survived"].value_counts(normalize=True))

df_trainSex = df_train["Sex"] == 'male'
print(df_trainSex)

df_trainSS = df_train['Survived'][df_train['Sex'] == 'male'].value_counts(normalize=True)
print(dfSS)
"""

print(df_train['Pclass'].values)

"""
x = tf.placeholder(tf.float32, [None, 1])
y_ = tf.placeholder(tf.float32, [None, 1])

W = tf.Variable(tf.zeros([1, 1]))
b = tf.Variable(tf.zeros([1]))

y = tf.matmul(W, x) + b
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

optimizer = tf.train.GradientDescentOptimizer(0.0001)
train_step = optimizer.minimize(cost)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for i in range(3000):
    sess.run(train_step, feed_dict={x: df_train['Pclass'], y: df_train['Survived']})
    _cost = sess.run(cost, feed_dict={x: df_train['Pclass'], y: df_train['Survived']})
    _acc = sess.run(accuracy, feed_dict={x: df_train['Pclass'], y: df_train['Survived']})
    print(_cost, _acc)

print("accuracy test = ", sess.run(accuracy, feed_dict={x: df_test['Pclass'], y: df_test['Survived']}))

"""

