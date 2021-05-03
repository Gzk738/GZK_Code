import numpy as np

# Filters
# approximation
g_analysis = np.array([1, 1])/2
g_synthesis = np.array([1, 1])
# Detail
h_analysis = np.array([-1, 1])/2
h_synthesis =  np.array([1, -1])

# decompose signal
def analysis(x, f):
    y = np.zeros(len(x)//2)
    for n in range(len(y)):
        y[n] = x[n*2] * f[0] + x[n*2+1] * f[1]
    return y

# reconstruct signal
def synthesis(x, f):
    y = np.zeros(x.shape[0]*2)
    for n in range(x.shape[0]):
        y[2*n] = x[n] * f[0] 
        y[2*n + 1] = x[n] * f[1]
    return y

# discrete waveform transform
def dwt(x):
    # approximation coefficients
    a = analysis(x, g_analysis)
    # detailed coefficients
    d = analysis(x, h_analysis)
    return a, d

# inverse discrete waveform transform
def idwt(a, d):
    # approximated signal
    a_ = synthesis(a, g_synthesis)
    # detailed signal
    d_ = synthesis(d, h_synthesis)
    # reconstructed signal
    x_ = a_ - d_
    return x_.astype('uint8') # ensure compatibility with cv2

x = np.array([1, 3, 3, 7])
cA, dA = dwt(x)
x_reconstructed = idwt(cA, dA)

print('signal', x)
print('reconstruction', x_reconstructed)
