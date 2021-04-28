# -*- coding: utf-8 -*-
# @Time : 4/14/2021 1:52 PM
# @Author : XXX
# @Site : 
# @File : class02.py
# @Software: PyCharm
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math
import json

def read_this(image_file, gray_scale=False):
    image_src = cv2.imread(image_file)
    if gray_scale:
        image_src = cv2.cvtColor(image_src, cv2.COLOR_BGR2GRAY)
    else:
        image_src = cv2.cvtColor(image_src, cv2.COLOR_BGR2RGB)
    return image_src

def binarize_lib(image_file, thresh_val=127, with_plot=False, gray_scale=False):
    image_src = read_this(image_file=image_file, gray_scale=gray_scale)
    th, image_b = cv2.threshold(src=image_src, thresh=thresh_val, maxval=255, type=cv2.THRESH_BINARY)
    if with_plot:
        cmap_val = None if not gray_scale else 'gray'
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 20))

        ax1.axis("off")
        ax1.title.set_text('Original')

        ax2.axis("off")
        ax2.title.set_text("Binarized")

        ax1.imshow(image_src, cmap=cmap_val)
        ax2.imshow(image_b, cmap=cmap_val)
        return True
    return image_b


def convert_binary(image_matrix, thresh_val):
    white = 255
    black = 0

    initial_conv = np.where((image_matrix <= thresh_val), image_matrix, white)
    final_conv = np.where((initial_conv > thresh_val), initial_conv, black)

    return final_conv

def img_sliding():
    """img = cv2.imread("LENA256.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    arr = np.asarray(img)
    temp_arr = arr.copy()
    for i in temp_arr:
        for j in range(len(i)):
            i[j] = i[j] * 0.5
    plt.imsave('img_sliding.png', temp_arr)
    img_sliding = cv2.imread("img_sliding.png")
    cv2.imshow("original_img", img)
    cv2.imshow("img_sliding", img_sliding)
    cv2.waitKey()"""

    """
    image = cv2.imread("LENA256.jpg")
    img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    rows, cols = img_gray.shape
    flat_gray = img_gray.reshape((cols * rows,)).tolist()
    A = min(flat_gray)
    B = max(flat_gray)
    print("最小灰度值：%d，最大灰度值：%d" % (A, B))
    img_sliding = np.uint8(255 / (B - A) * (img_gray - A) + 0.1)
    cv2.imshow("original_img", image)
    cv2.imshow("img_sliding", img_sliding)
    cv2.waitKey()"""
    c = 1.2
    b = 100
    image = cv2.imread('LENA256.jpg')
    h, w, ch = image.shape
    blank = np.zeros([h, w, ch], image.dtype)
    dst = cv2.addWeighted(image, c, blank, 1 - c, b)  # 改变像素的API
    cv2.imshow('original_img', image)
    cv2.imshow("sliding", dst)
    cv2.waitKey()


def img_Stretching():
    original_img = cv2.imread('LENA256.jpg')

    original_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)

    rows, cols = original_img.shape
    flat_gray = original_img.reshape((cols * rows,)).tolist()
    A = min(flat_gray)
    B = max(flat_gray)
    print('A = %d,B = %d' % (A, B))
    output = np.uint8(255 / (B - A) * (original_img - A) + 10)
    cv2.imshow('original_img', original_img)
    cv2.imshow('Stretching2image', output)

    cv2.waitKey()
    return
def img_shink():
    c = 1
    b = -100
    image = cv2.imread('LENA256.jpg')
    h, w, ch = image.shape
    blank = np.zeros([h, w, ch], image.dtype)
    dst = cv2.addWeighted(image, c, blank, 1 - c, b)
    cv2.imshow('original_img', image)
    cv2.imshow("sliding", dst)
    cv2.waitKey()
def good_picture():
    image = cv2.imread('BEPE.jpg')


    c = 1.2
    b = 100
    h, w, ch = image.shape
    blank = np.zeros([h, w, ch], image.dtype)
    dst = cv2.addWeighted(image, c, blank, 1 - c, b)  # 改变像素的API
    cv2.imshow('original_img', image)
    cv2.imshow("sliding", dst)
    cv2.waitKey()

    print(image.shape)
    height = image.shape[0]
    width = image.shape[1]
    chanmels = image.shape[2]
    print('width:%s,height:%s,chanmels:%s' % (width, height, chanmels))
    for row in range(height):
        for col in range(width):
            for c in range(chanmels):
                pv = image[row, col, c]
                image[row, col, c] = pv * 1.5
    cv2.imshow('original', image)
    cv2.imshow('good_image', image)
    cv2.waitKey()


def binarization():
    image = cv2.imread('girl.jpg')
    image2 = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # 二值化函数
    cv2.threshold(image2, 140, 255, 0, image2)  # 二值化函数

    cv2.imshow("original", image)
    cv2.imshow("sliding", image2)
    cv2.waitKey()

def Dynamic_binarization(num):
    image = cv2.imread('girl.jpg')
    image2 = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # 二值化函数
    cv2.threshold(image2, num, 255, 0, image2)  # 二值化函数

    cv2.imshow("original", image)
    cv2.imshow("sliding", image2)
    cv2.waitKey()
if __name__ == '__main__':
    img_sliding()
    img_Stretching()
    img_shink()
    good_picture()
    binarization()
    Dynamic_binarization(50)