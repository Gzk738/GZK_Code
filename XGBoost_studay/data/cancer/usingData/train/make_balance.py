# -*- coding:utf-8 -*-

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

"""
Create by doyoung on 18. 6. 19
Title : 
Explanation

"""
import copy

import pandas as pd

group0 = pd.read_csv("./Recurrence0.csv", encoding="utf-8")
group1 = pd.read_csv("./Recurrence1.csv", encoding="utf-8")

for i in range(100):
    sub_group0 = group0.sample(frac=(len(group1)/len(group0)))
    sub_train = pd.concat([sub_group0, group1]).reset_index(drop=True)
    sub_train = sub_train.sample(frac=1)
    sub_train.to_csv("./subtrain/{}.csv".format(i + 1), index=False, encoding="utf-8")

norm_train = pd.read_csv("./norm_train.csv", encoding="utf-8")
norm_group0 = norm_train[norm_train["Recurrence"]==0].reset_index(drop=True)
norm_group1 = norm_train[norm_train["Recurrence"]==1].reset_index(drop=True)

for i in range(100):
    sub_norm_group0 = norm_group0.sample(frac=(len(norm_group1)/len(norm_group0)))
    sub_norm_train = pd.concat([sub_norm_group0, norm_group1]).reset_index(drop=True)
    sub_norm_train = sub_norm_train.sample(frac=1)
    sub_norm_train.to_csv("./norm_subtrain/{}.csv".format(i + 1), index=False, encoding="utf-8")
