import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error

sns.set()
x = np.linspace(-2 * np.pi, 2 * np.pi, 25)
x = [k for k in x if k != 0]
x = np.array(x)
y = np.sin(np.power(x, -1))
plt.scatter(x, y)
plt.show()
w = np.diff(x)[0]

def sinc_kernel(x, x0: float, w: float, alpha: float = np.inf):
    x = x - x0
    if np.any(x == 0):
        return np.ones_like(x)
    x = x / w
    return (x >= -alpha) * (x < alpha) * np.sinc(x)

xk = np.linspace(-2 * np.pi, 2 * np.pi, 25)
xk = [k for k in xk if k != 0]
xk = np.array(xk)
plt.plot(xk, sinc_kernel(xk, x0=0, w=1, alpha=2))
plt.show()
print(w)

kernels = []
for xsample, ysample in zip(x, y):
    xk = np.linspace(-2 * np.pi, 2 * np.pi, 50)
    xk = [k for k in xk if k != 0]
    xk = np.array(xk)
    k = ysample * sinc_kernel(xk, x0=xsample, w=w, alpha=np.pi)
    kernels.append(k)

kernels = np.asarray(kernels)
print(kernels.shape)
yinterp = kernels.sum(axis=0)

plt.plot(xk, yinterp)
plt.scatter(x, y)
plt.show()

print(f"{(mean_squared_error(yinterp, np.sin(np.power(xk,-1))) / np.std(np.sin(np.power(xk,-1)))):.4%}")
