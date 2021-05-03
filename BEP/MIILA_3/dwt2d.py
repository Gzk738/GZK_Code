from math import *
import numpy as np
import cv2

# Filters
# approximation
g_analysis = np.array([1, 1])/2
g_synthesis = np.array([1, 1])
# Detail
h_analysis = np.array([-1, 1])/2
h_synthesis =  np.array([1, -1])

# DWT funcs
def analysis(x, f):
    y = np.zeros(len(x)//2)
    for n in range(len(y)):
        y[n] = x[n*2] * f[0] + x[n*2+1] * f[1]
    return y

def synthesis(x, f):
    y = np.zeros(x.shape[0]*2)
    for n in range(x.shape[0]):
        y[2*n] = x[n] * f[0] 
        y[2*n + 1] = x[n] * f[1]
    return y

def dwt(x):
    a = analysis(x, g_analysis)
    d = analysis(x, h_analysis)
    return a, d

def idwt(a, d):
    a_ = synthesis(a, g_synthesis)
    d_ = synthesis(d, h_synthesis)
    x_ = a_ - d_
    return x_

def dwt2d(x):

    rows_d = np.zeros((x.shape[0], x.shape[1]//2))
    rows_a = np.zeros((x.shape[0], x.shape[1]//2))

    # rows
    for row_idx in range(rows_a.shape[0]):
        row = x[row_idx]
        rows_a[row_idx], rows_d[row_idx] = dwt(row)

    # assuming square image
    cA = np.zeros((x.shape[0]//2, x.shape[1]//2))
    cH = np.zeros((x.shape[0]//2, x.shape[1]//2))
    cV = np.zeros((x.shape[0]//2, x.shape[1]//2))
    cD = np.zeros((x.shape[0]//2, x.shape[1]//2))
        
    for col_idx in range(cA.shape[1]):
        col_d = rows_d[:,col_idx].T
        cA[:,col_idx], cH[:,col_idx] = dwt(col_d)
        col_a = rows_a[:,col_idx].T
        cV[:,col_idx], cD[:,col_idx] = dwt(col_a)

    return cA, cH, cV, cD

def idwt2d(cA, cH, cV, cD):

    rows_d = np.zeros((cA.shape[0]*2, cA.shape[0]))
    rows_a = np.zeros((cA.shape[0]*2, cA.shape[0]))

    for col_idx in range(rows_a.shape[1]):
        rows_d[:,col_idx] = idwt(cA[:,col_idx], cH[:,col_idx])
        rows_a[:,col_idx] = idwt(cV[:,col_idx], cD[:,col_idx])

    x_ = np.zeros((cA.shape[0]*2, cA.shape[0]*2))

    for row_idx in range(x_.shape[0]):
        x_[row_idx] = idwt(rows_a[row_idx], rows_d[row_idx])
    return x_

img = cv2.imread('noisy_img.png', 0)
img = cv2.resize(img, (256, 256))

cA, cH, cV, cD = dwt2d(img)

print(cA.max(), cA.mean())

thresh = 30

for dim in range(cD.shape[0]):
    for c in range(cD.shape[1]):
        if abs(cH[dim, c]) < thresh:
            cH[dim, c] = 0
        if abs(cV[dim, c]) < thresh:
            cV[dim, c] = 0
        if abs(cD[dim, c]) < thresh:
            cD[dim, c] = 0

x_ = idwt2d(cA, cH, cV, cD)

cv2.imshow('noisy lena', img.astype('uint8'))
cv2.imshow('lena', x_.astype('uint8'))
cv2.waitKey()

