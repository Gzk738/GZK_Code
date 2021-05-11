# coding: utf-8
# Team : Quality Management Center
# Author：Guo Zikun
# Email: gzk798412226@gmail.com
# Date ：2021/5/7 16:46
# Tool ：PyCharm
import time
import numpy as np
import xgboost as xgb
from xgboost import plot_importance, plot_tree
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_boston
import matplotlib
import matplotlib.pylab as plt
import os

iris = load_iris()
X, y= iris.data, iris.target
X_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234565)

params ={
    'booster': 'gbtree',

    'objective': 'multi:softmax',
    'num_class':3,

    'gamma':0.1
}
plst = list(params.items())
dtrain = xgb.DMatrix(X_train, y_train)
dtest = xgb. DMatrix(x_test)

num_rounds = 50
model = xgb.train(plst, dtrain, num_rounds)

y_pred = model.predict(dtest)

acc = accuracy_score(y_test, y_pred)
print(acc)

plot_importance(model)
plt.show()

plot_tree(model, num_trees=5)
