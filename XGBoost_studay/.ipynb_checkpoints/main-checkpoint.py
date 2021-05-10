# coding: utf-8
# Team : Quality Management Center
# Author：Guo Zikun
# Email: gzk798412226@gmail.com
# Date ：2021/5/7 16:46
# Tool ：PyCharm
from xgboost import XGBClassifier
import pandas as pd
import numpy as np
import xgboost
from sklearn.metrics import accuracy_score
def load_data():
    data = pd.read_csv('data/cancer/usingData/train/train.csv')
    #前4/5作为训练集，后1/5作为测试集
    data_training = data[0:int(len(data)*3/5)]
    data_test = data[int(len(data)*3/5):len(data)]
    #分割
    train_x = np.array(data_training.iloc[:, [i for i in range(data_training.shape[1]-1)]])
    train_y = np.array(data_training['Recurrence'])
    test_x = np.array(data_test.iloc[:, [i for i in range(data_test.shape[1]-1)]])
    test_y = np.array(data_test['Recurrence'])

    return train_x, train_y, test_x, test_y


def XGBoost():
    train_x, train_y, test_x, test_y = load_data()
    #训练
    clf=XGBClassifier(base_score=0.5, booster='gbtree', learning_rate=0.05, max_depth=8, n_estimators=50)
    clf.fit(train_x, train_y)
    #测试
    result = clf.score(test_x, test_y)
    print("prediction rate :", result)


if __name__ == '__main__':
    XGBoost()

