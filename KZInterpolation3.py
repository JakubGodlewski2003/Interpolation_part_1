import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
sns.set()
x = np.linspace(-2 * np.pi, 2 * np.pi, 25)
y = np.sin(x)
plt.scatter(x, y)
plt.show()

def linear_kernel(x, x0, w):
    x = x - x0
    x = x / w
    return ((1 - np.abs(x)) * (np.abs(x) < 1)).astype(float)
kernels = []
w = np.diff(x)[0]
for xsample, ysample in zip(x, y):
    xk = np.linspace(-2 * np.pi, 2 * np.pi, 50)
    k = ysample * linear_kernel(xk, x0=xsample, w=w)
    kernels.append(k)
kernels = np.asarray(kernels)
yinterp = kernels.sum(axis=0)
plt.plot(xk, yinterp)
plt.scatter(x, y)
plt.show()
from sklearn.metrics import mean_squared_error
f"{(mean_squared_error(yinterp, np.sin(xk)) / np.std(np.sin(xk))):.4%}"
