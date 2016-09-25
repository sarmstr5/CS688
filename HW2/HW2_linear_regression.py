# Problem 1 part b
# Normal equation coefs = (Xtrans * X)inverted * Xtrans*y
from pylab import *
import pandas as pd
import numpy as np

df = pd.read_csv('ex1data1.txt', names=['city_population', 'profit'])
df.insert(1, 'ones', 1)
x = df[['ones', 'city_population']]
y = df['profit']
xTx = (x.T.dot(x))
xTx_inv = np.linalg.inv(xTx)
xTy = x.T.dot(y)
reg_coefs = xTx_inv.dot(xTy)
y_hat = x.dot(reg_coefs)
y_hat.head()
figure()
plt.scatter(x=x['city_population'], y=y_hat)


def gradient_descent(x, y, w, alpha):
    N = len(x)
    w0_gradient = 0
    w1_gradient = 0
    for i in range(0, N):
        w0_gradient += (1 / N) * (w[0] + w[1] * x[i] - y[i])
        w1_gradient += (1 / N) * (w[0] + w[1] * x[i] - y[i]) * x[i]
    w[0] += -alpha * w0_gradient
    w[1] += -alpha * w1_gradient
    return w


def run_descent(x, y, w, alpha, iterations):
    for i in range(iterations):
        w = gradient_descent(x, y, w, alpha)
    return w


w = np.ones(2)
x_pop = x['city_population']
alpha = .015
iterations = 1000
w = run_descent(x_pop, y, w, alpha, iterations)
print(w)
print(reg_coefs)