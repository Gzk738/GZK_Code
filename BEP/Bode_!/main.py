# coding: utf-8
# Team : Quality Management Center
# Author：Guo Zikun
# Email: gzk798412226@gmail.com
# Date ：2021/4/8 16:01
# Tool ：PyCharm
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("class2.jpg")
cv2.imshow(r"img", img)
cv2.waitKey()
kernels = [(u"test01",np.array([[1, 0, 0],[0, 1, 0],[0, 0, 1]])),
		   (u"Qualcomm filter",np.array([[0.0,-1, 0],[-1,5,-1],[0,-1,0]])),
		   (u"edge detection",np.array([[-1.0, -1, -1],[-1, 8, -1],[-1, -1, -1]])),]
index = 0
fig, axes = plt.subplots(1, 3, figsize=(12, 4.3))
for ax, (name, kernel) in zip(axes, kernels):#zip 将数组和元胞放在一起
    print("ax:",ax,"\n","name",name,"\n","knl\n",kernel)
    dst=cv2.filter2D(img, -1, kernel)
    # 由于matplotlib的颜色顺序和OpenCV的顺序相反
    cv2.imshow(r"@#$)(*&^%$#%@#%", dst)
    cv2.waitKey()
    ax.set_title(name)
    ax.axis("off")
fig.subplots_adjust(0.02, 0, 0.98, 1, 0.02, 0)#调节窗口大小
# Average Filter
img_mean = cv2.blur(img, (10,10))
cv2.imshow(r"Average Filter", img_mean)
cv2.waitKey()

# 高斯滤波
img_Guassian = cv2.GaussianBlur(img,(5,5),0)
cv2.imshow(r"Guass", img_Guassian)
cv2.waitKey()

# 中值滤波
img_median = cv2.medianBlur(img, 5)
cv2.imshow(r"Median", img_median)
cv2.waitKey()

# 双边滤波
img_bilater = cv2.bilateralFilter(img,9,75,75)
cv2.imshow(r"@#$)(*&^%$#%@#%", img_bilater)
cv2.waitKey()

