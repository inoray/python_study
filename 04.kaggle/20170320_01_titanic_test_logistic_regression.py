import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.learn as learn

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

print('----- 데이터 형태 파악 ----- ')
print(df_train.describe())
print(df_train.shape)

# ticket과 cabin 정보는 NaN가 많으므로 버린다.
df_train = df_train.drop(['Ticket', 'Cabin'], axis=1)
# Remove NaN values
df_train = df_train.dropna()
print(df_train.describe())
print(df_train.shape)

print('----- 타이타닉 data 분석 -----')
print('- 생존자 분석')
print(df_train['Survived'].value_counts())
print(df_train['Survived'].value_counts(normalize=True))

print('- 남자 생존자 분포')
df_t_male = df_train['Sex'] == 'male'
print(df_train['Survived'][df_t_male].value_counts())
print(df_train['Survived'][df_t_male].value_counts(normalize=True))

print('- 여자 생존자 분포')
df_t_female = df_train['Sex'] != 'male'
print(df_train['Survived'][df_t_female].value_counts())
print(df_train['Survived'][df_t_female].value_counts(normalize=True))


if False:
    # specifies the parameters of our graphs
    fig1 = plt.figure(figsize=(18, 8))
    alpha = alpha_scatterplot = 0.2
    alpha_bar_chart = 0.55

    # lets us plot many different shaped graphs together
    ax1 = plt.subplot2grid((2, 3), (0, 0))
    # plots a bar graph of those who survived vs those who did not.
    df_train.Survived.value_counts().plot(kind='bar', alpha=alpha_bar_chart)
    # this nicely sets the margins in matplotlib to deal with a recent bug 1.3.1
    ax1.set_xlim(-1, 2)
    # puts a title on our graph
    plt.title("Distribution of Survival, (1 = Survived)")


    plt.subplot2grid((2, 3), (0, 1))
    plt.scatter(df_train.Survived, df_train.Age, alpha=alpha_scatterplot)
    # sets the y axis label
    plt.ylabel("Age")
    # formats the grid line style of our graphs
    plt.grid(b=True, which='major', axis='y')
    plt.title("Survival by Age,  (1 = Survived)")

    ax3 = plt.subplot2grid((2, 3), (0, 2))
    df_train.Pclass.value_counts().plot(kind="barh", alpha=alpha_bar_chart)
    ax3.set_ylim(-1, len(df_train.Pclass.value_counts()))
    plt.title("Class Distribution")

    plt.subplot2grid((2, 3), (1, 0), colspan=2)
    # plots a kernel density estimate of the subset of the 1st class passangers's age
    df_train.Age[df_train.Pclass == 1].plot(kind='kde')
    df_train.Age[df_train.Pclass == 2].plot(kind='kde')
    df_train.Age[df_train.Pclass == 3].plot(kind='kde')

    # plots an axis label
    plt.xlabel("Age")
    plt.title("Age Distribution within classes")
    # sets our legend for our graph.
    plt.legend(('1st Class', '2nd Class', '3rd Class'), loc='best')

    ax5 = plt.subplot2grid((2, 3), (1, 2))
    df_train.Embarked.value_counts().plot(kind='bar', alpha=alpha_bar_chart)
    ax5.set_xlim(-1, len(df_train.Embarked.value_counts()))
    # specifies the parameters of our graphs
    plt.title("Passengers per boarding location")

    plt.show()

if False:
    fig2 = plt.figure(figsize=(18, 6))

    # create a plot of two subsets, male and female, of the survived variable.
    # After we do that we call value_counts() so it can be easily plotted as a bar graph.
    # 'barh' is just a horizontal bar graph
    df_male = df_train.Survived[df_train.Sex == 'male'].value_counts().sort_index()
    df_female = df_train.Survived[df_train.Sex == 'female'].value_counts().sort_index()
    print(df_male)
    print(df_female)

    ax1 = fig2.add_subplot(121)
    df_male.plot(kind='barh', label='Male', alpha=0.55)
    df_female.plot(kind='barh', color='#FA2379', label='Female', alpha=0.55)
    plt.title("Who Survived? with respect to Gender, (raw value counts) ")
    plt.legend(loc='best')
    ax1.set_ylim(-1, 2)

    # adjust graph to display the proportions of survival by gender
    ax2 = fig2.add_subplot(122)
    (df_male/float(df_male.sum())).plot(kind='barh', label='Male', alpha=0.55)
    (df_female/float(df_female.sum())).plot(kind='barh', color='#FA2379', label='Female', alpha=0.55)
    plt.title("Who Survived proportionally? with respect to Gender")
    plt.legend(loc='best')

    ax2.set_ylim(-1, 2)

    plt.show()

if False:
    fig = plt.figure(figsize=(18, 4))
    alpha_level = 0.65

    # building on the previous code, here we create an additional subset with in the gender subset 
    # we created for the survived variable. I know, thats a lot of subsets. After we do that we call 
    # value_counts() so it it can be easily plotted as a bar graph. this is repeated for each gender 
    # class pair.
    ax1 = fig.add_subplot(141)
    female_highclass = df_train.Survived[df_train.Sex == 'female'][df_train.Pclass != 3].value_counts()
    female_highclass.plot(kind='bar', label='female, highclass', color='#FA2479', alpha=alpha_level)
    ax1.set_xticklabels(["Survived", "Died"], rotation=0)
    ax1.set_xlim(-1, len(female_highclass))
    plt.title("Who Survived? with respect to Gender and Class")
    plt.legend(loc='best')

    ax2 = fig.add_subplot(142, sharey=ax1)
    female_lowclass = df_train.Survived[df_train.Sex == 'female'][df_train.Pclass == 3].value_counts()
    female_lowclass.plot(kind='bar', label='female, low class', color='pink', alpha=alpha_level)
    ax2.set_xticklabels(["Died", "Survived"], rotation=0)
    ax2.set_xlim(-1, len(female_lowclass))
    plt.legend(loc='best')

    ax3 = fig.add_subplot(143, sharey=ax1)
    male_lowclass = df_train.Survived[df_train.Sex == 'male'][df_train.Pclass == 3].value_counts()
    male_lowclass.plot(kind='bar', label='male, low class', color='lightblue', alpha=alpha_level)
    ax3.set_xticklabels(["Died", "Survived"], rotation=0)
    ax3.set_xlim(-1, len(male_lowclass))
    plt.legend(loc='best')

    ax4 = fig.add_subplot(144, sharey=ax1)
    male_highclass = df_train.Survived[df_train.Sex == 'male'][df_train.Pclass != 3].value_counts()
    male_highclass.plot(kind='bar', label='male, highclass', alpha=alpha_level, color='steelblue')
    ax4.set_xticklabels(["Died", "Survived"], rotation=0)
    ax4.set_xlim(-1, len(male_highclass))
    plt.legend(loc='best')

    plt.show()

if False:
    fig = plt.figure(figsize=(15, 8))

    ax1 = fig.add_subplot(221)
    df_train.SibSp.value_counts().plot(kind='bar', alpha=0.5)
    plt.title("SibSp count")

    ax2 = fig.add_subplot(222)

    #for name, group in df_train['SibSp'].groupby(df_train['Survived']):
    #    print(name)
    #    print(group)
    groupSibsp = df_train['SibSp'].groupby(df_train['Survived'])
    df_t_SibSp = pd.DataFrame({'Non-Survivors': groupSibsp.get_group(0),
                               'Survivors': groupSibsp.get_group(1)})
    df_t_SibSp.plot(kind='hist', ax=ax2, alpha=0.5, stacked=True)
    plt.title("SibSp by Survived")

    for p in ax2.patches:
        bl = p.get_xy()
        height = p.get_height()
        ax2.text(p.get_x()+p.get_width()/2., height/2+bl[1]+3, "%d" % height, ha="center")

    ax3 = fig.add_subplot(223)

    groupParch = df_train['Parch'].groupby(df_train['Survived'])
    df_t_Parch = pd.DataFrame({'Non-Survivors': groupParch.get_group(0),
                               'Survivors': groupParch.get_group(1)})
    df_t_Parch.plot(kind='hist', ax=ax3, alpha=0.5, stacked=True)
    plt.title('Parch by Survived')
    for p in ax3.patches:
        bl = p.get_xy()
        height = p.get_height()
        ax3.text(p.get_x()+p.get_width()/2., height/2+bl[1]+3, "%d" % height, ha="center")
    print(df_t_Parch)

    ax4 = fig.add_subplot(224)
    ax4.hist((groupSibsp.get_group(0), groupSibsp.get_group(1))
             , alpha=0.5
             , stacked=True)
    ax4.set_xticklabels([str(i) for i in range(0, 9)])
    plt.show()

if True:
    df_train['Sex_c'] = df_train.Sex.astype('category').cat.codes
    df_train['Embarked_c'] = df_train.Embarked.astype('category').cat.codes
    #print(df_train.Sex_c)
    #print(df_train.Embarked_c)
    df_train_x = df_train.drop(['PassengerId', 'Survived', 'Name', 'Sex', 'Embarked'], axis=1)
    df_train_y = df_train.Survived
    ty = df_train_y.values.reshape(-1, 1)
    #print(df_train_x)
    #df_train_real = pd.DataFrame()
    #lst = df_train_x.values
    #print(df_train_x.values)
    #print(df_train_y.values.reshape(-1, 1))

    print(df_train_x.shape[1])

    x = tf.placeholder(tf.float32, [None, df_train_x.shape[1]])
    y_ = tf.placeholder(tf.float32, [None, 1])

    W = tf.Variable(tf.zeros([df_train_x.shape[1], 1]))
    b = tf.Variable(tf.zeros([1]))

    y = tf.nn.sigmoid(tf.matmul(x, W) + b)
    cost = tf.reduce_mean(tf.square(y-y_))

    correct_prediction = tf.equal(y_, tf.round(y))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train_step = optimizer.minimize(cost)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    for i in range(30000):
        sess.run(train_step, feed_dict={x: df_train_x.values, y_: ty})
        if i % 100 == 0:
            _cost = sess.run(cost, feed_dict={x: df_train_x.values, y_: ty})
            _acc = sess.run(accuracy, feed_dict={x: df_train_x.values, y_: ty})
            print(i, _cost, _acc)

    print("accuracy test = ", sess.run(accuracy, feed_dict={x: df_train_x.values, y_: ty}))
    for i in range(5):
        print("data test = ", i, ty[i].reshape(-1, 1), sess.run(y, feed_dict={x: df_train_x.values[i].reshape(-1, df_train_x.shape[1]), y_: ty[i].reshape(-1, 1)}))

    # test
    df_test['Sex_c'] = df_test.Sex.astype('category').cat.codes
    df_test['Embarked_c'] = df_test.Embarked.astype('category').cat.codes

    df_test_x = df_test.drop(['PassengerId', 'Name', 'Sex', 'Embarked', 'Ticket', 'Cabin'], axis=1)

    df_result = pd.DataFrame(columns=('PassengerId', 'Survived'))
    df_result.PassengerId = df_result.PassengerId.astype(np.int64)
    df_result.Survived = df_result.Survived.astype(np.int64)
    for i in range(df_test_x.shape[0]):
        predict = sess.run(y, feed_dict={x: df_test_x.values[i].reshape(-1, df_test_x.shape[1])})
        df_result.loc[i] = [df_test.PassengerId[i].astype(int), sess.run(tf.round(predict))[0][0].astype(int)]
    df_result.Survived[df_result.Survived < 0] = 0
    df_result.to_csv("data/result.csv", index=False)


