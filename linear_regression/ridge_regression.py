# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt


def plot_data(x, y):
    """绘制给定数据x与y的图像"""
    plt.figure()
    # ====================== 你的代码 ==========================
    # 绘制x与y的图像
    # 使用 matplotlib.pyplt 的命令 plot, xlabel, ylabel 等。
    # 提示：可以使用 'rx' 选项使数据点显示为红色的 "x"，
    #       使用 "markersize=8, markeredgewidth=2" 使标记更大

    # 给制数据
    plt.plot(x, y, 'rx')
    # 设置y轴标题为 'Profit in $10,000s'
    plt.ylabel('Profit in $10000s')
    # 设置x轴标题为 'Population of City in 10,000s'
    plt.xlabel('Population of City in 10000s')
    # =========================================================
    plt.show()


def compute_cost(X, y, theta, lmb):
    """计算线性回归的代价。"""
    m = len(y)
    n = len(theta)
    J = 0.0
    regular = 0.0
    # 计算给定 theta 参数下线性回归的代价
    # 将正确的代价赋值给 J
    regular = lmb/(2.0*m) * np.sum(theta[:(n-1), :]**2)
    J = (1.0/(2.0*m))*np.sum( (np.dot(X.T, theta) - y.reshape(m,1)) ** 2 )
    J_regular = J+regular
    # =========================================================
    return J_regular


def gradient_descent(X, y, theta, alpha, lmb, num_iters):
    """执行梯度下降算法来学习参数 theta。"""
    m = len(y)
    n = len(theta)
    J_history = np.zeros((num_iters,))
    theta_history = np.zeros((num_iters, n))

    for iter in range(num_iters):
        # 计算给定 theta 参数下线性回归的梯度，实现梯度下降算法
        theta = theta - alpha*( (1.0/m) * np.dot( X, np.dot(X.T, theta)-y.reshape(m,1) ) + (lmb/m)*theta )
        # =========================================================
        # 将各次迭代后的代价进行记录
        J_history[iter] = compute_cost(X, y, theta, lmb)
        theta_history[iter, :] = theta.T 

    return theta, J_history, theta_history


def plot_linear_fit(X, y, x, yy, theta):
    """在绘制数据点的基础上绘制回归得到直线"""
    # 绘制x与y的图像
    # 使用 matplotlib.pyplt 的命令 plot, xlabel, ylabel 等。
    # 提示：可以使用 'rx' 选项使数据点显示为红色的 "x"，
    #      使用 "markersize=8, markeredgewidth=2" 使标记更大
    #      使用"label=<your label>"设置数据标识，
    #      如 "label='Data'" 表示原始数据点
    #      "label='Ridge Regression'" 表示线性回归的结果

    # 给制数据
    plt.figure()
    plt.plot(X[0,:], y, 'gx', label='Linear Regression')
    plt.plot(x[0,:], yy, 'rx', label='Data')
    # 使用 legned 命令显示图例，图例显示位置为 "loc='lower right'"
    plt.legend(loc="upper right")
    # 设置y轴标题为 'Profit in $10,000s'
    plt.ylabel('Y')
    # 设置x轴标题为 'Population of City in 10,000s'
    plt.xlabel('X')
    # =========================================================
    plt.show()


def pearson_coefficient(y,yy):
    m = len(y)
    correct = np.dot(y.reshape(1,m),yy.reshape(m,1))/(np.sqrt(np.sum(y**2))*np.sqrt(np.sum(yy**2)))
    return correct



if __name__ == '__main__':
    
    data_train = np.loadtxt('train_data.txt', delimiter=',')
    data_test = np.loadtxt('test_data.txt')
    y_real = np.loadtxt('real_data.txt')
    x_train, y_train = data_train[:, 0], data_train[:, 1]
    m = len(y_train)
    
    x_train2 = x_train**2
    x_train3 = x_train**3
    x_train4 = x_train**4
    x_train5 = x_train**5
#    x_train6 = x_train**6
#    x_train7 = x_train**7
#    x_train8 = x_train**8
#    x_train9 = x_train**9
    
    dim = 6
    xx_train = np.vstack((x_train,x_train2,x_train3,x_train4,x_train5))
    X_train = np.ones((dim, m))
    X_train[:dim-1, :] = xx_train[:, :]
    theta = np.zeros((dim, 1))
    iterations = 2000
    alpha = 0.15
    lmb = 1
    J0 = compute_cost(X_train, y_train, theta, lmb)
    print (J0)
    theta, J_history, theta_history = gradient_descent(X_train, y_train, theta,
                                        alpha, lmb, iterations)
    
    x_test = data_test
    x_test2 = x_test**2
    x_test3 = x_test**3
    x_test4 = x_test**4
    x_test5 = x_test**5
#    x_test6 = x_test**6
#    x_test7 = x_test**7
#    x_test8 = x_test**8
#    x_test9 = x_test**9
    
    xx_test = np.vstack((x_test,x_test2,x_test3,x_test4,x_test5))
    X_test = np.ones((dim, m))
    X_test[:dim-1, :] = xx_test[:, :]
    yp_train = np.dot(X_train.T, theta)
    yp_test = np.dot(X_test.T, theta)
    
    correct = 0.0
    correct = pearson_coefficient(yp_train, y_train)[0,0]
    print('y_train_correct = ')
    print(correct)
    correct = 0.0
    correct = pearson_coefficient(yp_test, y_real)[0,0]
    print('y_test_correct = ')
    print(correct)
    plot_linear_fit(X_train, yp_train, X_train, y_train, theta)
    plt.show()
    plot_linear_fit(X_test, yp_test, X_test, y_real, theta)
    plt.show()
