import numpy as np
import scipy.optimize as optimize
import matplotlib.pyplot as plt


def real_func(x):
    return np.sin(2 * np.pi * x)


def fit_func(p, x):
    f = np.poly1d(p)  # 给定p的数组，指定的就是在一维坐标上的多项式函数，所以这个函数返回的就是一个函数
    return f(x)


def residual_func(p, x, y):
    loss = fit_func(p, x) - y
    return loss


# 先拟定10个点
x = np.linspace(0, 1, 10)
x_points = np.linspace(0, 1, 1000)

y_ = real_func(x)
y = [np.random.normal(0, 0.1) + y1 for y1 in y_]


def fitting(M=0):
    # 多项式拟合，使用最小二乘法
    p_init = np.random.rand(M + 1)
    p_lsq = optimize.leastsq(residual_func, p_init, args=(x, y))
    print("拟合函数的参数为", p_lsq[0])
    plt.plot(x_points, real_func(x_points), label='real')
    plt.plot(x_points, fit_func(p_lsq[0], x_points), label='fitted curve')
    plt.plot(x, y, 'bo', label='noise')
    plt.show()
    return p_lsq


# 为了防止过拟合，引入正则化
regularization = 0.0001


def residuals_func_reularization(p, x, y):
    loss = fit_func(p, x) - y
    loss = np.append(loss, np.sqrt(0.5 * regularization * np.square(p)))  # 使用2范数z作为正则化项
    return loss


def fitting_regularization(M=0, regularization=0.001):
    # 多项式拟合，使用最小二乘法
    p_init = np.random.rand(M + 1)
    p_lsq = optimize.leastsq(residuals_func_reularization, p_init, args=(x, y))
    print("拟合函数的参数为", p_lsq[0])
    plt.plot(x_points, real_func(x_points), label='real')
    plt.plot(x_points, fit_func(p_lsq[0], x_points), label='fitted curve' + M + 1)
    plt.plot(x, y, 'bo', label='noise')
    plt.show()
    return p_lsq


p_lsq_0 = fitting(0)
p_lsq_1 = fitting(1)
p_lsq_2 = fitting(2)
p_lsq_3 = fitting(3)
p_lsq_4 = fitting(4)
p_lsq_5 = fitting(5)
p_lsq_9 = fitting_regularization(9)
