# coding: utf-8
# Team : Quality Management Center
# Author：Guo Zikun
# Email: gzk798412226@gmail.com
# Date ：4/16/2021 8:19 PM
# Tool ：PyCharm
import cv2
import math
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
def segmentation():
	img = cv2.imread("flower.jpg")
	"""img 2 gray"""
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	"""blurred"""
	blurred = cv2.GaussianBlur(gray, (9, 9), 0)

	"""Extract gradient"""
	gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0)
	gradY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1)

	gradient = cv2.subtract(gradX, gradY)
	gradient = cv2.convertScaleAbs(gradient)

	"""show imgs"""
	cv2.imshow("flower", img)
	cv2.imshow("gray_flower", gray)
	cv2.imshow("blurred", blurred)
	cv2.imshow("gradient", gradient)
	cv2.waitKey()

if __name__ == '__main__':
    segmentation()