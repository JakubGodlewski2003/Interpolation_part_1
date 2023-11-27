import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
sns.set()
x = np.linspace(-2 * np.pi, 2 * np.pi, 25)
y = np.sign(np.sin(8 * x))
plt.scatter(x, y)
plt.show()
def kernel(x, x0, w):
    x = x - x0
    return ((x < w) * (x > 0)).astype(float)
xk = np.linspace(-2 * np.pi, 2 * np.pi, 100)
plt.plot(xk, kernel(xk, x0=0, w=1, ))
plt.show()
w = np.diff(x)[0]
kernels = []

for xsample, ysample in zip(x, y):
    xk = np.linspace(-2 * np.pi, 2 * np.pi, 150)
    k = ysample * kernel(xk, x0=xsample, w=w)
    kernels.append(k)
kernels = np.asarray(kernels)

yinterp = kernels.sum(axis=0)
plt.plot(xk, yinterp)
plt.scatter(x, y)
plt.show()
from sklearn.metrics import mean_squared_error
f"{(mean_squared_error(yinterp, np.sign(np.sin(8 * xk))) / np.std(np.sign(np.sin(8 * xk)))):.4%}"