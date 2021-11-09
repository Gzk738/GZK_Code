# 自适应中值滤波
# count 为最大窗口数，original 为原图
import matplotlib.pyplot as plt

import cv2
import numpy as np
def adaptiveMedianDeNoise(count, original):
    # 初始窗口大小
    startWindow = 3
    # 卷积范围
    c = int(count/2)
    rows, cols = original.shape
    newI = np.zeros(original.shape)
    for i in range(c, rows - c):
        for j in range(c, cols - c):
            k = int(startWindow / 2)
            median = np.median(original[i - k:i + k + 1, j - k:j + k + 1])
            mi = np.min(original[i - k:i + k + 1, j - k:j + k + 1])
            ma = np.max(original[i - k:i + k + 1, j - k:j + k + 1])
            if mi < median < ma:
                if mi < original[i, j] < ma:
                    newI[i, j] = original[i, j]
                else:
                    newI[i, j] = median

            else:
                while True:
                    startWindow = startWindow + 2
                    k = int(startWindow / 2)
                    median = np.median(original[i - k:i + k + 1, j - k:j + k + 1])
                    mi = np.min(original[i - k:i + k + 1, j - k:j + k + 1])
                    ma = np.max(original[i - k:i + k + 1, j - k:j + k + 1])

                    if mi < median < ma or startWindow > count:
                        break

                if mi < median < ma or startWindow > count:
                    if mi < original[i, j] < ma:
                        newI[i, j] = original[i, j]
                    else:
                        newI[i, j] = median

    return newI


def medianDeNoise(original):
    rows, cols = original.shape
    ImageDenoise = np.zeros(original.shape)
    for i in range(3, rows - 3):
        for j in range(3, cols - 3):
            ImageDenoise[i, j] = np.median(original[i - 3:i + 4, j - 3:j + 4])
    return ImageDenoise

def show(f, s, a, b, c):
    plt.subplot(a, b, c)
    plt.imshow(f, "gray")
    plt.axis('on')
    plt.title(s)


original = plt.imread("lena.jpg", 0)



adapMedianDeNoise = adaptiveMedianDeNoise(7, original)
mediDeNoise = medianDeNoise(original)
plt.figure()
show(original, "original", 2, 2, 1)

show(adapMedianDeNoise, "adaptiveMedianDeNoise", 2, 2, 3)
show(mediDeNoise, "medianDeNoise", 2, 2, 4)
plt.show()