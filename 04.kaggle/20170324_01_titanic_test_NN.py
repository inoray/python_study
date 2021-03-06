import pandas as pd
import numpy as np
import tensorflow as tf


def MinMaxScaler(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # noise term prevents the zero division
    return numerator / (denominator + 1e-7)


def title_mapping(x):
    if x.Title in set(['Don', 'Rev', 'Sir', 'Jonkheer']):
        return 'Mr'
    elif x.Title in set(['Lady', 'the Countess']):
        return 'Mrs'
    elif x.Title in set(['Mlle', 'Mme', 'Dona', 'Ms']):
        return "Miss"
    elif x.Title in set(['Major', 'Col', 'Capt']):
        return "Officer"
    elif x.Title == 'Dr' and x.Sex == 'female':
        return 'Mrs'
    elif x.Title == 'Dr' and x.Sex == 'male':
        return 'Mr'
    else:
        return x.Title


df_train = pd.read_csv('data/train.csv', header=0)

df_train['Sex_c'] = df_train.Sex.astype('category').cat.codes
df_train['Embarked_c'] = df_train.Embarked.astype('category').cat.codes
df_train['Title'] = df_train['Name'].apply(lambda x: (x.split(',')[1]).split('.')[0][1:])
df_train['Title'] = df_train.apply(title_mapping, axis=1)
df_train['Title'] = df_train.Title.astype('category').cat.codes
#df_train['Fare'] = df_train['Fare'] / 10

df_train = df_train.drop(['Ticket', 'Cabin', 'PassengerId', 'Name', 'Sex', 'Embarked'], axis=1)
df_train = df_train.dropna()

df_train_x = df_train.drop(['Survived'], axis=1)
df_train_y = df_train.Survived
ty = df_train_y.values.reshape(-1, 1)

x = tf.placeholder(tf.float32, [None, df_train_x.shape[1]])
y_ = tf.placeholder(tf.float32, [None, 1])

W1 = tf.get_variable("W1", shape=[df_train_x.shape[1], 21], initializer=tf.contrib.layers.xavier_initializer())
W2 = tf.get_variable("W2", shape=[21, 21], initializer=tf.contrib.layers.xavier_initializer())
W3 = tf.get_variable("W3", shape=[21, 1], initializer=tf.contrib.layers.xavier_initializer())

b1 = tf.Variable(tf.random_normal([21]))
b2 = tf.Variable(tf.random_normal([21]))
b3 = tf.Variable(tf.random_normal([1]))

fc1 = tf.nn.relu(tf.matmul(x, W1) + b1)
fc2 = tf.nn.relu(tf.matmul(fc1, W2) + b2)

y = tf.nn.sigmoid(tf.matmul(fc2, W3) + b3)

beta = 0.001
#l2reg = 0.001*tf.reduce_sum(tf.square())
cost = (tf.reduce_mean(tf.square(y_ - y))
        + beta*tf.nn.l2_loss(W1)
        + beta*tf.nn.l2_loss(W2)
        + beta*tf.nn.l2_loss(W3)
        + beta*tf.nn.l2_loss(b1)
        + beta*tf.nn.l2_loss(b2)
        + beta*tf.nn.l2_loss(b3))

optimizer = tf.train.AdamOptimizer(0.001)
train_step = optimizer.minimize(cost)

correct_prediction = tf.equal(y_, tf.round(y))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(50000):
    sess.run(train_step, feed_dict={x: df_train_x.values, y_: ty})
    if i % 100 == 0:
        cost_ = sess.run(cost, feed_dict={x: df_train_x.values, y_: ty})
        acc_ = sess.run(accuracy, feed_dict={x: df_train_x.values, y_: ty})
        print(i, cost_, acc_)


#result
df_test = pd.read_csv('data/test.csv', header=0)
df_test['Sex_c'] = df_test.Sex.astype('category').cat.codes
df_test['Embarked_c'] = df_test.Embarked.astype('category').cat.codes
df_test['Title'] = df_test['Name'].apply(lambda x: (x.split(',')[1]).split('.')[0][1:])
df_test['Title'] = df_test.apply(title_mapping, axis=1)
df_test['Title'] = df_test.Title.astype('category').cat.codes
#df_test['Fare'] = df_test['Fare'] / 10

df_test_x = df_test.drop(['PassengerId', 'Name', 'Sex', 'Embarked', 'Ticket', 'Cabin'], axis=1)

df_result = pd.DataFrame(columns=('PassengerId', 'Survived'))
df_result.PassengerId = df_result.PassengerId.astype(np.int64)
df_result.Survived = df_result.Survived.astype(np.int64)
for i in range(df_test_x.shape[0]):
    predict = sess.run(y, feed_dict={x: df_test_x.values[i].reshape(-1, df_test_x.shape[1])})
    df_result.loc[i] = [df_test.PassengerId[i].astype(int), sess.run(tf.round(predict))[0][0].astype(int)]
df_result.Survived[df_result.Survived < 0] = 0
df_result.to_csv("data/result_NN_02.csv", index=False)

sess.close()

