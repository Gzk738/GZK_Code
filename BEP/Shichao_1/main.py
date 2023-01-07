# coding: utf-8
# Team : Quality Management Center
# Author：Guo Zikun
# Email: gzk798412226@gmail.com
# Date ：4/23/2021 3:11 PM
# Tool ：PyCharm
# Gabor 滤波器实现
# K_size：Gabor核大小 K_size x K_size
# Sigma : σ
# Gamma： γ
# Lambda：λ
# Psi  ： ψ
# angle： θ
import numpy as np
import time
import cv2


def diag_sym_matrix(k=256):
    base_matrix = np.zeros((k,k))
    base_line = np.array(range(k))
    base_matrix[0] = base_line
    for i in range(1,k):
        base_matrix[i] = np.roll(base_line,i)
    base_matrix_triu = np.triu(base_matrix)
    return base_matrix_triu + base_matrix_triu.T

def cal_dist(hist):
    Diag_sym = diag_sym_matrix(k=256)
    hist_reshape = hist.reshape(1,-1)
    hist_reshape = np.tile(hist_reshape, (256, 1))
    return np.sum(Diag_sym*hist_reshape,axis=1)

def LC(image_gray):
    image_height,image_width = image_gray.shape[:2]
    hist_array = cv2.calcHist([image_gray], [0], None, [256], [0.0, 256.0])
    gray_dist = cal_dist(hist_array)

    image_gray_value = image_gray.reshape(1,-1)[0]
    image_gray_copy = [(lambda x: gray_dist[x]) (x)  for x in image_gray_value]
    image_gray_copy = np.array(image_gray_copy).reshape(image_height,image_width)
    image_gray_copy = (image_gray_copy-np.min(image_gray_copy))/(np.max(image_gray_copy)-np.min(image_gray_copy))
    return image_gray_copy


if __name__ == '__main__':
    file = r"human.png"
    img = cv2.imread("human.png")
    start = time.time()
    image_gray = cv2.imread(file, 0)
    saliency_image = LC(image_gray)
    end = time.time()

    print("Duration: %.2f seconds." % (end - start))
    cv2.imshow("original", img)
    cv2.imshow("gray saliency image", saliency_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# import torch
# import cv2
# import numpy as np
# import torchvision
# import torchvision.transforms as transforms
# from PIL import Image
# import torch.nn as nn
# import matplotlib.pyplot as plt
# import numpy as np
# import kornia
#
#
# def showTorchImage(image, simage):
#     mode = transforms.ToPILImage()(image).save(simage)
#
#
# def readImage(path='human.jpg'):  # 这里可以替换成自己的图片
#     mode = Image.open(path)
#     transform1 = transforms.Compose([
#         transforms.Resize((64, 64)),
#         transforms.ToTensor()
#     ])
#     mode = transform1(mode)
#     return mode
#
#
# img2 = readImage('human.jpg').unsqueeze(0)
#
# kernel = torch.ones(1, 3, 3)
# img2 = kornia.filter2D(img2, kernel).squeeze(0).view(64, 64, 3)
# print(img2.shape)
#
# img = readImage('human.jpg').view(64, 64, 3)
#
# # img = 0.2126 * img[0,:,:] + 0.7152 * img[1,:,:] + 0.0722 * img[2,:,:]
#
# img = img2 - img
#
# fft2 = np.fft.fft2(img.numpy())
# fshift = np.fft.fftshift(fft2)  # 傅里叶变换
# ishift = np.fft.ifftshift(fshift)
# io = np.fft.ifft2(ishift)
# io = np.abs(io)
#
# plt.subplot(233)
# plt.imshow(io, 'gray')
# plt.title('fft2')
# plt.show()





