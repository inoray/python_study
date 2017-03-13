"""
해커에게 전해들은 머신러닝 study

출처: https://tensorflow.blog/%ED%95%B4%EC%BB%A4%EC%97%90%EA%B2%8C-%EC%A0%84%ED%95%B4%EB%93%A4%EC%9D%80-%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-1/
"""
from sklearn import datasets, linear_model, metrics
from sklearn.cross_validation import train_test_split, KFold
import matplotlib.pyplot as plt
import numpy as np


class SingleNeuron(object):
    def __init__(self):
        self._x = 0
        self._w = 0
        self._b = 0

    def setParam(self, w, b):
        self._w = w
        self._b = b

    def forPass(self, x):
        self._x = x
        _y_hat = self._x * self._w + self._b
        return _y_hat

    def backprob(self, err, lr=0.1):
        m = len(self._x)
        self._w_grad = lr * np.sum(err * self._x) / m
        self._b_grad = lr * np.sum(err * 1) / m

    def update_grad(self):
        self.setParam(self._w + self._w_grad, self._b + self._b_grad)

    def fit(self, x, y, iter=10, lr=0.1, cost_check=False):
        cost = []
        for i in range(iter):
            y_hat = self.forPass(x)
            err = y - y_hat
            self.backprob(err, lr)
            self.update_grad()
            if cost_check:
                cost.append(np.sum(np.square(err)) / len(y))
        return cost


# 사이킷 런에 포함된 샘플데이터 가져오기. 당뇨병 데이터 셋
diabetes = datasets.load_diabetes()
print(diabetes.data.shape, diabetes.target.shape)

# 샘플데이터를 학습데이터와 테스트데이터로 분리
x_train, x_test, y_train, y_test = train_test_split(diabetes.data[:, 2],
                                                    diabetes.target, test_size=0.1, random_state=10)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

# SingleNeuron 클래스를 사용한 학습
n1 = SingleNeuron()
n1.setParam(5, 1)
n1.fit(x_train, y_train, 30000)

# learning rate 별 cost 비교
costs = []
learning_rate = [1.999, 1.0, 0.1]
for lr in learning_rate:
    n2 = SingleNeuron()
    n2.setParam(5, 1)
    costs.append([])
    costs[-1] = n2.fit(x_train, y_train, 2000, lr, True)

for i, color in enumerate(['red', 'blue', 'black']):
    plt.plot(list(range(2000)), costs[i], color)
plt.ylim(3500, 7000)
plt.xlabel('iter')
plt.ylabel('cost')
plt.show()

# 크로스 밸리데이션
kf = KFold(n_splits=5, shuffle=True, random_state=10)
costs = []
learning_rate = [1.2, 1.0, 0.8]
for lr in learning_rate:
    validation_errors = 0
    n2 = SingleNeuron()
    n2.setParam(5, 1)
    for train, validation in kf.split(x_train):
        n2.fit(x_train[train], y_train[train], 2000, lr)
        y_hat = n2.forPass(x_train[validation])
        validation_errors += metrics.mean_squared_error(y_train[validation], y_hat)
    print(validation_errors / 5)


print("- 직접구현한 SingleNeuron 클래스 사용")
print("final w, b: %f, %f" %(n1._w, n1._b))


# 사이킷런의 선형회기 알고리즘 사용한 방법
sgd_regr = linear_model.SGDRegressor(n_iter=30000, penalty='none')
sgd_regr.fit(diabetes.data[:, 2].reshape(-1, 1), diabetes.target)

print("- 사이킷런의 선형회기 알고리즘 사용한 방법")
print('Coefficients: ', sgd_regr.coef_, sgd_regr.intercept_)

# 도표
plt.scatter(diabetes.data[:, 2], diabetes.target)
plt.xlabel('x')
plt.ylabel('y')
plt.plot(diabetes.data[:, 2], diabetes.data[:, 2]*n1._w + n1._b, "blue")
plt.plot(diabetes.data[:, 2], diabetes.data[:, 2]*sgd_regr.coef_+ sgd_regr.intercept_, "red")
plt.show()
