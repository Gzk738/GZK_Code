# coding: utf-8
# Team : Quality Management Center
# Author：Guo Zikun
# Email: gzk798412226@gmail.com
# Date ：2021/4/7 15:13
# Tool ：PyCharm
import cv2
import math
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
#load img
img = cv2.imread("eye.jpg")
scaling_factor = 0.85

#Scaled size and resolution
img = cv2.resize(img, None,
				 fx=scaling_factor,
				 fy=scaling_factor,
				 interpolation=cv2.INTER_AREA)
cv2.circle(img, (61, 60), 10, (255, 255, 255))
cv2.line(img, (50, 60), (70, 60), (255, 255, 255))
cv2.line(img, (60, 50), (60, 70), (255, 255, 255))
#show
cv2.imshow("sdf", img)
cv2.waitKey()

#load img
lena = cv2.imread("lena.jpg")

#BGR2GRAY
GRAY = cv2.cvtColor(lena, cv2.COLOR_BGR2GRAY)
cv2.imshow("original", lena)
cv2.imshow("GRAY", GRAY)
cv2.waitKey()
"""#2CMYK
CMYK = cv2.cvtColor(lena, cv2.COLOR_BGR2CMYK)
cv2.imshow("lena", CMYK)
cv2.waitKey()
"""
#2HSV
HSV = cv2.cvtColor(lena, cv2.COLOR_BGR2HSV)
cv2.imshow("original", lena)
cv2.imshow("HSV", HSV)
cv2.waitKey()

numpy_lena = Image.open("lena.jpg")
arr = np.asarray(numpy_lena)
print(arr.shape)
for i in arr:
	for j in i:
		print(j)
"""
RGB TO GRAY
formula = Grey = 0.299*R + 0.587*G + 0.114*B
"""
temp_arr = arr.copy()
for i in temp_arr:
	for j in i:
		j[0] = j[0]*0.3
		j[1] = j[1]*0.59
		j[2] = j[2]*0.11

"""
RGB TO HSV
"""
"""temp_arr = arr.copy()
for i in temp_arr:
	for j in i:
		j[0] = j[0]*0.3
		j[1] = j[1]*0.59
		j[2] = j[2]*0.11"""

"""
RGB TO CMYK
"""
"""temp_arr = arr.copy()
for i in temp_arr:
	for j in i:
		j[0] = j[0]*0.3
		j[1] = j[1]*0.59
		j[2] = j[2]*0.11
"""
plt.imsave('myfunc_numpy.png',temp_arr)
my_fun = cv2.imread("myfunc_numpy.png")
cv2.imshow("original", lena)
cv2.imshow("my_fun", my_fun)
cv2.waitKey()


