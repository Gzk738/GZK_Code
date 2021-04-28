# coding: utf-8
# Team : Quality Management Center
# Author：Carson
# Date ：2021/3/15 14:27
# Tool ：PyCharm
from gensim.models import word2vec
import logging
import jieba
import os
import pathlib
import codecs,sys

"""p = pathlib.Path("D:/software/github/Deep_learn/HELP_GUNGUN/data/data")
dirs = p.glob("**/*.txt")

out_dirs = open("output_dirs", "w")

for item in dirs:
	tmp = str(item)
	print(tmp)
	f = open(item, "r", encoding='utf8')
	buf = f.read()
	output_f = open("corpus.txt", "a", encoding='utf8')
	output_f.write(buf)
	f.close()
	output_f.close()

"""


"""import re

f = open ("data.txt", "r", encoding="utf-8")
words = f.read()
f.close()
print("读取原始文件完成\n开始正则表达式")
words = re.sub("[A-Za-z0-9\!\%\[\]\,\。]", "", words)
words = re.sub(r'[^\u4e00-\u9fa5]',"", words)
print("开始正则表达式OK")
f = open("data_output.txt", "w", encoding="utf-8")
f.write(words)
f.close()"""



#数据分词
import jieba.posseg as psg
import re

f = open("data_output.txt", "r", encoding="utf-8")
s = f.read()

words = psg.cut(s)
print("cut successed")

for word, flag in words:
    out_f = open("jieba_data.txt", "a", encoding="utf-8")
    out_f.write(word + " ")
    out_f.close()

print("finished!!!")

"""#数据清洗
import re
f = open("jieba_data.txt", "r", encoding="utf-8")
str = f.read()
f.close()
str = re.sub("[A-Za-z0-9\!\%\[\]\,\。]", "", str)

str = re.sub(r'[^\u4e00-\u9fa5]',"", str)



print(str)"""
