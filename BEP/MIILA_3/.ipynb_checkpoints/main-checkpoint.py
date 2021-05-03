#%%
import cv2
import numpy as np
import pywt
import matplotlib.pyplot as plt

img = cv2.imread('eagle.png', 0) # flag 0 reads as grayscale
print(img.shape)
noise = np.random.rand(*img.shape) * 0.5 * 255
noisy_img = img + noise
cv2.normalize(noisy_img, noisy_img, 255, 0)

cv2.imshow('img', img)
cv2.imshow('noisy img', noisy_img)
#%%

thresh = [0, 0, 0.5]

coeffs = pywt.dwt2(noisy_img, 'haar')
cA, (cH, cV, cD) = coeffs

coeffs2 = pywt.dwt2(cA, 'haar')
cA2, (cH2, cV2, cD2) = coeffs2

coeffs3 = pywt.dwt2(cA2, 'haar')
cA3, (cH3, cV3, cD3) = coeffs3

const = 3

for dim in range(cD.shape[0]):
    for c in range(cD.shape[1]):
        thresh = cH[dim, c].mean() + cH[dim, c].std() * const
        if abs(cH[dim, c]) < thresh:
            cH[dim, c] = 0
        thresh = cV[dim, c].mean() + cV[dim, c].std() * const
        if abs(cV[dim, c]) < thresh:
            cV[dim, c] = 0
        thresh = cD[dim, c].mean() + cD[dim, c].std() * const
        if abs(cD[dim, c]) < thresh:
            cD[dim, c] = 0

for dim in range(cD2.shape[0]):
    for c in range(cD2.shape[1]):
        thresh = cH2[dim, c].mean() + cH2[dim, c].std() * const
        if abs(cH2[dim, c]) < thresh:
            cH2[dim, c] = 0
        thresh = cV2[dim, c].mean() + cV2[dim, c].std() * const
        if abs(cV2[dim, c]) < thresh:
            cV2[dim, c] = 0
        thresh = cD2[dim, c].mean() + cD2[dim, c].std() * const
        if abs(cD2[dim, c]) < thresh:
            cD2[dim, c] = 0

for dim in range(cD3.shape[0]):
    for c in range(cD3.shape[1]):
        thresh = cH3[dim, c].mean() + cH3[dim, c].std() * const
        if abs(cH3[dim, c]) < thresh:
            cH3[dim, c] = 0
        thresh = cV3[dim, c].mean() + cV3[dim, c].std() * const
        if abs(cV3[dim, c]) < thresh:
            cV3[dim, c] = 0
        thresh = cD3[dim, c].mean() + cD3[dim, c].std() * const
        if abs(cD3[dim, c]) < thresh:
            cD3[dim, c] = 0

coeffs3 = cA3, (cH3, cV3, cD3)
cA2 = pywt.idwt2(coeffs3, 'haar')
coeffs2 = cA2, (cH2, cV2, cD2)
cA = pywt.idwt2(coeffs2, 'haar')
coeffs = cA, (cH, cV, cD)
noisy_img_rec = pywt.idwt2(coeffs, 'haar')

# %%
cv2.imwrite('noisy_img_rec.png', noisy_img_rec*255)
cv2.imwrite('noisy_img.png', noisy_img*255)
cv2.waitKey(0)

# %%

