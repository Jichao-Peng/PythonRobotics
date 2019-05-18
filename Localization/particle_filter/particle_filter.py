"""

Particle Filter localization sample

author: Atsushi Sakai (@Atsushi_twi)

"""

import numpy as np
import math
import matplotlib.pyplot as plt

# Estimation parameter of PF
Q = np.diag([0.1])**2  # range error
R = np.diag([1.0, np.deg2rad(40.0)])**2  # input error

#  Simulation parameter
Qsim = np.diag([0.2])**2
Rsim = np.diag([1.0, np.deg2rad(30.0)])**2

DT = 0.1  # time tick [s]
SIM_TIME = 50.0  # simulation time [s]
MAX_RANGE = 20.0  # maximum observation range

# Particle filter parameter
NP = 100  # Number of Particle
NTh = NP / 2.0  # Number of particle for re-sampling

show_animation = True


def calc_input():
    v = 1.0  # [m/s]
    yawrate = 0.1  # [rad/s]
    u = np.array([[v, yawrate]]).T
    return u


def observation(xTrue, xd, u, RFID):

    xTrue = motion_model(xTrue, u) # 这里其实就是不带噪声的运动学方程

    # add noise to gps x-y
    z = np.zeros((0, 3)) # 这里申明一个列数为3，行数为0的矩阵

    for i in range(len(RFID[:, 0])): # 遍历所有RFID点

        dx = xTrue[0, 0] - RFID[i, 0]
        dy = xTrue[1, 0] - RFID[i, 1]
        d = math.sqrt(dx**2 + dy**2) # 这里求得当前的位置距离各个RFID点的位置
        if d <= MAX_RANGE: # 如果再测量范围内的话
            dn = d + np.random.randn() * Qsim[0, 0]  # 对所测得的距离增加噪声
            zi = np.array([[dn, RFID[i, 0], RFID[i, 1]]]) # 记录带噪声的距离和RFID的xy坐标
            z = np.vstack((z, zi)) # 将这些点存储起来，【---这里相当于是对观测方程增加了噪声---】

    # add noise to input
    ud1 = u[0, 0] + np.random.randn() * Rsim[0, 0] # 对控制变量增加噪声， 实际上就是对运动方程增加噪声
    ud2 = u[1, 0] + np.random.randn() * Rsim[1, 1]
    ud = np.array([[ud1, ud2]]).T

    xd = motion_model(xd, ud) # 【---这里相当于是对运动方程增加了噪声---】

    return xTrue, z, xd, ud # 这里的返回值分别是真实的位置，带噪声的测量，带噪声的先验和带噪声的控制


def motion_model(x, u):

    F = np.array([[1.0, 0, 0, 0],
                  [0, 1.0, 0, 0],
                  [0, 0, 1.0, 0],
                  [0, 0, 0, 0]])

    B = np.array([[DT * math.cos(x[2, 0]), 0],
                  [DT * math.sin(x[2, 0]), 0],
                  [0.0, DT],
                  [1.0, 0.0]])

    x = F.dot(x) + B.dot(u) # 这里实际上就是状态转移方程Ax+Bu，这个A和B都是可以通过运动学方程推导出来的，另外这个方程目前是不带噪声的

    return x


def gauss_likelihood(x, sigma):
    p = 1.0 / math.sqrt(2.0 * math.pi * sigma ** 2) * \
        math.exp(-x ** 2 / (2 * sigma ** 2))

    return p


def calc_covariance(xEst, px, pw):
    cov = np.zeros((3, 3)) # 方差矩阵

    for i in range(px.shape[1]):
        dx = (px[:, i] - xEst)[0:3] # 整个粒子对估计值求差
        cov += pw[0, i] * dx.dot(dx.T) # 然后求所有粒子关于这三个状态的方差

    return cov


def pf_localization(px, pw, xEst, PEst, z, u): # 这里输入的分别是粒子，粒子的权值，估计的位置，带噪声的测量和带噪声的控制
    """
    Localization with Particle filter
    """

    for ip in range(NP): # 遍历所有的粒子
        x = np.array([px[:, ip]]).T # 提取出第ip个粒子
        w = pw[0, ip] # 提取出第ip个粒子的权重

        #  【---第一步---粒子采样过程---P(x_k|x_k-1)---】
        ud1 = u[0, 0] + np.random.randn() * Rsim[0, 0] # 这里加噪声的原因是实现随机采样（个人理解，这里不一定对）
        ud2 = u[1, 0] + np.random.randn() * Rsim[1, 1]
        ud = np.array([[ud1, ud2]]).T
        x = motion_model(x, ud) # 粒子在带噪声的状态转移方程下更新

        #  【---第二步---权值更新过程---P(y_k|x_k)---】
        for i in range(len(z[:, 0])): # 遍历所有的测量
            dx = x[0, 0] - z[i, 1]
            dy = x[1, 0] - z[i, 2]
            prez = math.sqrt(dx**2 + dy**2) # 这里计算我采样后的状态（位置）距离RFID有多远
            dz = prez - z[i, 0] # 这里是测量的误差
            w = w * gauss_likelihood(dz, math.sqrt(Q[0, 0])) # 跟新权重

        px[:, ip] = x[:, 0] # 更新粒子的状态
        pw[0, ip] = w # 跟新粒子的权重

    #  【---第三步---权值归一化过程---】
    pw = pw / pw.sum()  # 对权值进行归一化

    #  【---第四步---状态估计过程---】
    xEst = px.dot(pw.T) # 加权平均就获得了估计的位置
    PEst = calc_covariance(xEst, px, pw) # 获得粒子的方差

    #  【---第五步---重采样过程---】
    px, pw = resampling(px, pw) # 重采样，SIR粒子滤波中必须有的步骤

    return xEst, PEst, px, pw


def resampling(px, pw):
    """
    low variance re-sampling
    """

    Neff = 1.0 / (pw.dot(pw.T))[0, 0]  # 按照公式判定那些粒子是低效的
    if Neff < NTh:
        wcum = np.cumsum(pw) # 将权值累加起来，例如 [1,2,3] 在进行cumsum之后变成 [1,3,6]，这其实就是轮盘采样的方法
        base = np.cumsum(pw * 0.0 + 1 / NP) - 1 / NP # 同上理解，会形成一个阶梯状的数列
        resampleid = base + np.random.rand(base.shape[0]) / NP # 加上噪声，形成均匀分布的随机采样值

        inds = []
        ind = 0
        for ip in range(NP):
            while resampleid[ip] > wcum[ind]:
                ind += 1
            inds.append(ind) # 这里存储的是冲采样后的id

        px = px[:, inds] # 将id对应的粒子提取出来
        pw = np.zeros((1, NP)) + 1.0 / NP  # 所有的粒子的权值初始化为1/N

    return px, pw


def plot_covariance_ellipse(xEst, PEst):  # pragma: no cover
    Pxy = PEst[0:2, 0:2] # 取xy部分的方差
    eigval, eigvec = np.linalg.eig(Pxy) # 求特征和特征向量

    if eigval[0] >= eigval[1]:
        bigind = 0
        smallind = 1
    else:
        bigind = 1
        smallind = 0

    t = np.arange(0, 2 * math.pi + 0.1, 0.1)

    # eigval[bigind] or eiqval[smallind] were occassionally negative numbers extremely
    # close to 0 (~10^-20), catch these cases and set the respective variable to 0
    try:
        a = math.sqrt(eigval[bigind]) # 对特征值开方，如果特征值为负则设为0
    except ValueError:
        a = 0

    try:
        b = math.sqrt(eigval[smallind])
    except ValueError:
        b = 0

    x = [a * math.cos(it) for it in t]
    y = [b * math.sin(it) for it in t]
    angle = math.atan2(eigvec[bigind, 1], eigvec[bigind, 0])
    R = np.array([[math.cos(angle), math.sin(angle)],
                  [-math.sin(angle), math.cos(angle)]])
    fx = R.dot(np.array([[x, y]]))
    px = np.array(fx[0, :] + xEst[0, 0]).flatten()
    py = np.array(fx[1, :] + xEst[1, 0]).flatten() # 上面的步骤没有仔细去研究，这里其实就是根据求得的协方差矩阵的两个特征值画一个代表协方差的椭圆的步骤
    # 这里可以观察下这个协方差矩阵的变换情况，刚开始是非常小的，后来慢慢变大，因为刚刚开始所有的粒子是初始化为0的，这个圈圈不代表粒子的实际位置，仅仅是一个分布的展示
    plt.plot(px, py, "--r")


def main():
    print(__file__ + " start!!")

    time = 0.0

    # RFID positions [x, y]
    RFID = np.array([[10.0, 0.0],
                     [10.0, 10.0],
                     [0.0, 15.0],
                     [-5.0, 20.0]]) # RFID的位置，就是图中那四个小星星

    # State Vector [x y yaw v]'
    xEst = np.zeros((4, 1)) # 预测值
    xTrue = np.zeros((4, 1)) # 真实值
    PEst = np.eye(4)

    px = np.zeros((4, NP))  # 存储的粒子，NP指有多少个粒子，一个粒子有四个特征，因此这个维度是4*100的
    pw = np.zeros((1, NP)) + 1.0 / NP  # 存储的粒子的权重，一个粒子只有一个权重，这个维度是1*100的，然后初始值是1/N,这和重采样的结果是一致的
    xDR = np.zeros((4, 1))  # Dead reckoning 航迹预测值

    # history
    hxEst = xEst
    hxTrue = xTrue
    hxDR = xTrue

    while SIM_TIME >= time:
        time += DT
        u = calc_input() # 相当于是输入控制参数

        xTrue, z, xDR, ud = observation(xTrue, xDR, u, RFID) # 这里输入的四个值分别是真实值， 航迹推算值， 输入的控制参数， RFID的位置
        # z是带噪声的测量值[d，x，y],其中d是距离RFID的距离，x是RFID横坐标，y是RFID纵坐标
        # xDR是带噪声的先验，或者可以理解为没有观测，仅仅通过运动方程获得的结果
        # ud是带噪声的控制量


        # 粒子滤波算法就是在这个里面实现的
        xEst, PEst, px, pw = pf_localization(px, pw, xEst, PEst, z, ud) # 这里输入的分别是粒子，粒子的权值，估计的位置，带噪声的测量和带噪声的控制

        # store data history 将两个结果水平方向叠加起来，其实就是把新的结果加到历史结果的后面
        hxEst = np.hstack((hxEst, xEst))
        hxDR = np.hstack((hxDR, xDR))
        hxTrue = np.hstack((hxTrue, xTrue))

        if show_animation:
            plt.cla()

            for i in range(px.shape[1]):
                dx = (px[:, i] - xEst)[0:3] # 整个粒子对估计值求差

            for i in range(len(z[:, 0])):
                plt.plot([xTrue[0, 0], z[i, 1]], [xTrue[1, 0], z[i, 2]], "-k")
            plt.plot(RFID[:, 0], RFID[:, 1], "*k")
            plt.plot(px[0, :], px[1, :], ".r") # 粒子的分布
            plt.plot(np.array(hxTrue[0, :]).flatten(), # 真实值x
                     np.array(hxTrue[1, :]).flatten(), "-b") # 真实值y
            plt.plot(np.array(hxDR[0, :]).flatten(), # 航迹推算值x
                     np.array(hxDR[1, :]).flatten(), "-k") # 航迹推算值y
            plt.plot(np.array(hxEst[0, :]).flatten(), # 预测值x
                     np.array(hxEst[1, :]).flatten(), "-r") # 预测值y
            plot_covariance_ellipse(xEst, PEst)
            plt.axis("equal")
            plt.grid(True)
            plt.pause(0.001)


if __name__ == '__main__':
    main()
