from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import pandas as pd
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

COLUMNS = ["crim", "zn", "indus", "nox", "rm", "age", "dis", "tax", "ptratio", "medv"]
FEATURES = ["crim", "zn", "indus", "nox", "rm", "age", "dis", "tax", "ptratio"]
LABEL = "medv"


def input_fn(data_set):
    features = {k: tf.constant(data_set[k].values) for k in FEATURES}
    labels = tf.constant(data_set[LABEL].values)
    return features, labels


training_set = pd.read_csv("./boston_data/boston_train.csv", skipinitialspace=True, skiprows=1, names=COLUMNS)
test_set = pd.read_csv("./boston_data/boston_test.csv", skipinitialspace=True, skiprows=1, names=COLUMNS)
prediction_set = pd.read_csv("./boston_data/boston_predict.csv", skipinitialspace=True, skiprows=1, names=COLUMNS)

print(training_set)

feature_cols = [tf.contrib.layers.real_valued_column(k) for k in FEATURES]
print(feature_cols)

regressor = tf.contrib.learn.DNNRegressor(feature_columns=feature_cols, hidden_units=[10, 10])

regressor.fit(input_fn=lambda: input_fn(training_set), steps=5000)

ev = regressor.evaluate(input_fn=lambda: input_fn(test_set), steps=1)
loss_score = ev["loss"]
print("Loss: %f" % loss_score)

y = regressor.predict(input_fn=lambda: input_fn(prediction_set))
predictions = list(itertools.islice(y, 6))
print("Predictions: {}".format(str(predictions)))

