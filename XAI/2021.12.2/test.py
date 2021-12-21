with open("./两种方法的测试记录_逐级删除.txt", "r")as f:
    data = f.read()
print(type(data))
b=eval(data)
print(type(b))
# data = dict(data)
