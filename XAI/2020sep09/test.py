# coding: utf-8
# Team : Quality Management Center
# Author：Guo Zikun
# Email: gzk798412226@gmail.com
# Date ：2021/9/17 20:30
# Tool ：PyCharm
from datasets import load_dataset
datasets = load_dataset('squad')
for i in range(5):
    text = datasets['train'][i]['context']
    question = datasets['train'][i]['question']
    answers = datasets['train'][i]['answers']
    ids = datasets['train'][i]['id']
    print(ids)
print( answers = datasets['train'][i]['answers'])