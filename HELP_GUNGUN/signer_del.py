# coding: utf-8
# Team : Quality Management Center
# Author：Carson
# Date ：2021/3/19 15:35
# Tool ：PyCharm
f = open("jieba_data.txt", "r", encoding = "utf-8")
str = f.read()
str = str.split()
for line in str:
	print(line)
	if len(line) == 1:
		continue
	else:
		outf = open("nosingle.txt", "a", encoding="utf-8")
		outf.write(line + " ")
		outf.close()

