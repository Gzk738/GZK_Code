# coding: utf-8
# Team : Quality Management Center
# Author：Guo Zikun
# Email: gzk798412226@gmail.com
# Date ：2021/4/28 4:56
# Tool ：PyCharm
import csv
class handle():
    lable = ""
    max_value = 0
    max_index = 0
    list_value = []
    country = []
def init_handle():
    """
    get the country list
    :return:
    """
    with open('world-happiness-report-2021.csv', 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        """' Column: 'Sweden', 'Luxembourg', 'New Zealand', 'Austria', 'Australia'... """
        colunm = [row[0] for row in reader]
    colunm.pop(0)
    for i in range(len(colunm)):
        handle.country.append(colunm[i])

    return
def read_data(row_index):
    with open('world-happiness-report-2021.csv','r',encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        """' Column: 'Sweden', 'Luxembourg', 'New Zealand', 'Austria', 'Australia'... """
        colunm = [row[row_index] for row in reader]
        handle.lable = colunm[0]
        colunm.pop(0)
        handle.list_value = colunm
    return colunm
def find_max_index(original_list):
    temp_list = []
    for i in range(len(original_list)):
        temp_list.append(float(original_list[i]))
    max_value = max(temp_list)
    for j in range(len(temp_list)):
        if temp_list[j] == max_value:
            handle.max_index = j
            return

if __name__ == '__main__':
    init_handle()
    for i in range(20):
        if i != 0 and i !=1:
            data = read_data(i)
            max_data = find_max_index(data)
            print( handle.lable,":", handle.country[i],":", data[handle.max_value])
            data.clear()
            max_data = 0

