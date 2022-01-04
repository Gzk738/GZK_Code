def cascading_min(data, min_id):
    print("threshold : ", min_id)
    for i in range(len(data)):
        if data[i] == 0:
            data[i] = 999
    d = {}
    li_min = []
    if min_id == 0:  # 设定一个空字典
        for i, v in enumerate(data):  # 利用函数enumerate列出lt的每个元素下标i和元素v
            d[v] = i  # 把v作为字典的键，v对应的值是i
        data.sort()  # 运用sort函数对lt元素排
        y = data[min_id]  # 此时lt中第二小的下标是1，求出对应的元素就是字典对应的键
        return [d[y]]
    if min_id != 0:
        for i, v in enumerate(data):  # 利用函数enumerate列出lt的每个元素下标i和元素v
            d[v] = i  # 把v作为字典的键，v对应的值是i
        data.sort()  # 运用sort函数对lt元素排
        for i in range(min_id):
            li_min.append(d[data[i]])  # 此时lt中第二小的下标是1，求出对应的元素就是字典对应的键
        return li_min
data = [0.1654018293427946, 1.6875491658403383, 2.5871719439388476, 999, 999, 999, 999]

cascading_min(data = "0x11111")