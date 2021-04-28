# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time : 2021/2/23 15:59
# @Author : Guo Zikun
# @Email : gzk798412226@gmail.com | gzk798412226@163.com
# @File : openexl.py
# @Software: PyCharm
import xml.sax
from xml.dom import minidom
import os
import xlwt

class ele:
    filename = []
    width = []
    heigh = []
    class_ = []

    xmlname = []

def get_ele(Failname):
    """
    Get the node name of path and related attribute values
    :param Failname:
    :return:
    """
    file_dir = "D:/software/github/Deep_learn/HELP_Lisijia/annotations/"

    for i in range(len(Failname)):
        dom = xml.dom.minidom.parse(file_dir + Failname[i])
        root = dom.documentElement

        xml_filnam = root.getElementsByTagName('filename')
        ele.filename.append(xml_filnam[0].firstChild.data)

        xml_width = root.getElementsByTagName('width')
        ele.width.append(xml_width[0].firstChild.data)

        xml_heigh = root.getElementsByTagName('height')
        ele.heigh.append(xml_heigh[0].firstChild.data)

        xml_class = root.getElementsByTagName('name')
        ele.class_.append(xml_class[0].firstChild.data)

    print("get all ele")


def get_filename():
    """
    get all the file name to process
    :return:
    """
    file_dir = "D:/software/github/Deep_learn/HELP_Lisijia/annotations/"
    for root, dirs, files in os.walk(file_dir, topdown=False):
        ele.xmlname.append(files)
    ele.xmlname = ele.xmlname[0]

    print("Number of files:", len(ele.xmlname))

    # for i in range(len(ele.xmlname)):
    #     #
    #     get_ele(file_dir + ele.xmlname[i])
def wr_excel():
    font = xlwt.Font()
    font.bold = True
    font.underline = True
    font.italic = True

    workbook = xlwt.Workbook(encoding = 'ascii')
    worksheet = workbook.add_sheet('My Worksheet')
    font = xlwt.Font()
    font.bold = True
    font.underline = True
    font.italic = True
    worksheet.write(0, 0, 'index')
    worksheet.write(0, 1, 'filename')
    worksheet.write(0, 2, 'width')
    worksheet.write(0, 3, 'hight')
    worksheet.write(0, 4, 'class')
    worksheet.write(0, 5, 'label')

    for i in range(len(ele.xmlname)):
        worksheet.write(i + 1, 0, str(i + 1))
        worksheet.write(i + 1, 1, ele.filename[i])
        worksheet.write(i + 1, 2, ele.width[i])
        worksheet.write(i + 1, 3, ele.heigh[i])
        worksheet.write(i + 1, 4, ele.class_[i])
        if ele.class_[i] == "crazing":
            worksheet.write(i + 1, 5, "0")
        elif ele.class_[i] == "inclusion":
            worksheet.write(i + 1, 5, "1")
        elif ele.class_[i] == "patches":
            worksheet.write(i + 1, 5, "2")
        elif ele.class_[i] == "pitted_surface":
            worksheet.write(i + 1, 5, "3")
        elif ele.class_[i] == "rolled-in_scale":
            worksheet.write(i + 1, 5, "4")
        elif ele.class_[i] == "scratches":
            worksheet.write(i + 1, 5, "5")

    workbook.save('formatting.xls')

if (__name__ == "__main__"):
    get_filename()

    get_ele(ele.xmlname)

    wr_excel()

