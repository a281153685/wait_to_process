# -*- coding: utf-8 -*-

"""
    @Author  : Mumu
    @Time    : 2022/7/14 10:02
    @Function: 

"""
import math
import pandas as pd
from numpy.linalg import det
from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

'''
定义复制动态方程求解过程中的部分函数，以及变量初始化
其中N为阶段数，将模型推导到n维向量
'''


class Dynamic:
    def __init__(self, N, ra, qa, rb, qb, alpha, ea, eb):
        self.ra = ra
        self.qa = qa
        self.rb = rb
        self.qb = qb
        self.alpha = alpha
        self.ea = ea
        self.eb = eb

        # 定义w\v
        self.w = []
        self.v = []
        for i in range(1, N + 1):
            if i % 2 != 0:
                wi = math.pow(self.ea, i - 1) * self.ra * self.qa
                vi = math.pow(self.eb, i - 1) * self.qa * (self.alpha - self.ra)
            else:
                wi = math.pow(self.ea, i - 1) * self.rb * self.qb
                vi = math.pow(self.eb, i - 1) * self.qb * (self.alpha - self.rb)
            self.w.append(wi)
            self.v.append(vi)

        # 定义维度N
        self.N = N

    # 计算E6   计算出负数概率就是不满足条件，，alpha > r1 > r2
    def e6(self):
        x_list, y_list = [], []
        for i in range(self.N):
            x_i_molecular, y_i_molecular = 1, 1
            x_i_bother, y_i_bother = 1, 1
            x_i_accumulate, y_i_accumulate = 0, 0
            for j in range(self.N):
                if i != j:
                    x_i_molecular *= self.v[j]
                    y_i_molecular *= self.w[j]
                x_i_bother *= self.v[j]
                y_i_bother *= self.w[j]
                x_i_accumulate += (1 / self.v[j])
                y_i_accumulate += (1 / self.w[j])
            x_i = x_i_molecular / (x_i_bother * x_i_accumulate)
            y_i = y_i_molecular / (y_i_bother * y_i_accumulate)
            x_list.append(x_i)
            y_list.append(y_i)

        statStar = []
        statStar.extend(x_list)
        statStar.extend(y_list)
        statStar = tuple(statStar)
        return statStar, self.calculateTraceDet(x_list, y_list)

    # 计算XYW的累加
    def sumXYW(self, X, Y):
        sum_ = 0.
        for i in range(self.N):
            sum_ += X[i] * Y[i] * self.w[i]
        return sum_

    # 计算XYV的累加
    def sumXYV(self, X, Y):
        sum_ = 0.
        for i in range(self.N):
            sum_ += X[i] * Y[i] * self.v[i]
        return sum_

    # 计算行列式的trace 和 det  输入参数是长度为N的列表X 、Y
    def calculateTraceDet(self, X, Y):
        mat_XX = np.mat(np.arange(self.N * self.N).reshape(self.N, self.N))
        mat_XY = np.mat(np.arange(self.N * self.N).reshape(self.N, self.N))
        mat_YY = np.mat(np.arange(self.N * self.N).reshape(self.N, self.N))
        mat_YX = np.mat(np.arange(self.N * self.N).reshape(self.N, self.N))
        mat_XX = mat_XX.astype('float')
        mat_XY = mat_XY.astype('float')
        mat_YY = mat_YY.astype('float')
        mat_YX = mat_YX.astype('float')
        for i in range(self.N):
            for j in range(self.N):
                if i == j:
                    sum_xyw = self.sumXYW(X, Y)
                    ki = Y[i] * self.w[i] - sum_xyw - X[i] * Y[i] * self.w[i]
                    mat_XX[i, j] = ki
                else:
                    mat_XX[i, j] = -1.0 * X[i] * Y[j] * self.w[j]

        for i in range(self.N):
            for j in range(self.N):
                if i == j:
                    ki = X[i] * self.w[i] * (1.0 - X[i])
                    mat_XY[i, j] = ki
                else:
                    mat_XY[i, j] = -1.0 * X[i] * X[j] * self.w[j]

        for i in range(self.N):
            for j in range(self.N):
                if i == j:
                    ki = Y[i] * self.v[i] * (1.0 - Y[i])
                    mat_YX[i, j] = ki
                else:
                    mat_YX[i, j] = (1.0 * Y[i] * Y[j] * self.v[j]) * -1

        for i in range(self.N):
            for j in range(self.N):
                if i == j:
                    sum_xyw = self.sumXYV(X, Y)
                    ki = X[i] * self.v[i] - sum_xyw - Y[i] * X[i] * self.v[i]
                    mat_YY[i, j] = ki
                else:
                    mat_YY[i, j] = -1.0 * Y[i] * Y[j] * self.v[j]

        mat_J = np.bmat("mat_XX mat_XY; mat_YX mat_YY")
        mat_J = mat_J.A
        detM = det(mat_J)
        traceM = mat_J.trace()
        if detM > 0 and traceM < 0:
            msg = '该点为稳定点'
        elif detM > 0 and traceM > 0:
            msg = '该点为不稳定点'
        else:
            msg = '该点为鞍点'
        return detM, traceM, msg

    # 定义n维输入动态方程
    def NDynamicEquation(self, w, t):
        X = w[:self.N]
        Y = w[self.N:]
        res_Equation = []
        sum_xyw = self.sumXYW(X, Y)
        sum_xyv = self.sumXYV(X, Y)
        for i in range(self.N):
            fxi = X[i] * (Y[i] * self.w[i] - sum_xyw)
            res_Equation.append(fxi)
        for i in range(self.N):
            fyi = Y[i] * (X[i] * self.v[i] - sum_xyv)
            res_Equation.append(fyi)

        return np.array(res_Equation)


def loopfind(ra, qa, rb, qb, alpha, e, list_stat):
    """
        定义相关初始变量
        """
    t = np.arange(0, 50, 1)  # 定义时间变化范围及步长
    detEquation = Dynamic(ra, qa, rb, qb, alpha, e)
    statStar6, (detResult, traceResult, msgStart) = detEquation.e6()

    list_stat.append(statStar6)

    for statStar in list_stat:
        dynamicResult = odeint(detEquation.fourDynamicEquation, statStar, t)  # 解4阶动态方程
        k = len(t) - 1
        finalStar = (round(dynamicResult[k, 0], 2), round(dynamicResult[k, 1], 2), round(dynamicResult[k, 2], 2),
                     round(dynamicResult[k, 3], 2))
        if finalStar[0] == round(statStar6[0], 2) and finalStar[2] == round(statStar6[2], 2):
            print('该点能均衡到E6点,该点的初始点为：{}， 结束点为：{}'.format(statStar, finalStar))
            print('此时的ra, qa, rb, qb, alpha, e状态为{}'.format((ra, qa, rb, qb, alpha, e)))
            save(ra, qa, rb, qb, alpha, e, statStar)


savefile = 'stateSave.txt'


def save(ra, qa, rb, qb, alpha, e, statStar):
    with open(savefile, 'a+') as f:
        linetxt = "ra:" + str(ra) + "; qa:" + str(qa) + "; rb:" + str(rb) + "; qb" + str(qb) + "; alpha: " + str(
            alpha) + "; e: " + str(e) + "; start: " + str(statStar) + '\n'
        f.writelines(linetxt)
    f.close()


def make_list_start():
    list_start = []
    for x1 in np.arange(0.1, 1, 0.1):
        for y1 in np.arange(0.1, 1, 0.1):
            x1 = round(x1, 2)
            y1 = round(y1, 2)
            start = (x1, 1 - x1, y1, 1 - y1)
            list_start.append(start)
    return list_start


def loopFindFinal(N, t, detEquation, statStar):
    detResult, traceResult, msgStart = detEquation.calculateTraceDet(statStar[:N], statStar[N:])  # 计算det 和 trace
    dynamicResult = odeint(detEquation.NDynamicEquation, statStar, t)  # 解n阶动态方程
    k = len(t) - 1
    finalStar = (dynamicResult[k, :])  # 演化结束状态
    return finalStar


def plotFinalStar():
    fileName = 'final_star.csv'
    df = pd.read_csv(fileName)
    startStar_list = df.iloc[:, 1].values
    finalStar_list = df.iloc[:, 2].values
    list_large = []
    list_small = []
    list_tmp = []
    tmp_x = ''
    for i in range(len(startStar_list)):
        res = startStar_list[i].replace('(', '').replace(')', '')
        res = tuple(map(float, res.split(',')))
        x1, x2, x3, y1, y2, y3 = res
        x1, x2, x3, y1, y2, y3 = round(x1, 1), round(x2, 1), round(x3, 1), round(y1, 1), round(y2, 1), round(y3, 1)
        x = str(x1) + str(x2) + str(x3)
        if x != tmp_x:
            list_large.append(list_small)
            list_tmp.append(tmp_x)
            list_small = []
            tmp_x = x
            list_small.append(i)
        else:
            list_small.append(i)
    list_large.append(list_small)
    list_tmp.append(tmp_x)
    print(len(list_large))

    #  11 个子图  3*4
    ax1 = plt.subplot(3, 4, 1)
    plt.sca(ax1)

    ax1 = plt.subplot(3, 4, 2)
    plt.sca(ax1)
    for ind, i in enumerate(list_large):
        if ind == 0:
            continue
        ax1 = plt.subplot(3, 4, ind)
        plt.sca(ax1)
        x1_list, x2_list, x3_list, y1_list, y2_list, y3_list = [], [], [], [], [], []
        for j in i:
            res = finalStar_list[j].replace('[', '').replace(']', '').replace(' ', ',').replace('\n', '')
            if res[0] == ',':
                res = res[1:]
            try:
                res = tuple(map(float, res.split(',')))
            except:
                res = res.replace(',,', ',')
                print('序号为：', j, '处理后的：', res)
                res = tuple(map(float, res.split(',')))
            x1, x2, x3, y1, y2, y3 = res
            x1, x2, x3, y1, y2, y3 = round(x1, 1), round(x2, 1), round(x3, 1), round(y1, 1), round(y2, 1), round(y3, 1)
            x1_list.append(x1)
            x2_list.append(x2)
            x3_list.append(x3)
            y1_list.append(y1)
            y2_list.append(y2)
            y3_list.append(y3)
        x_axis = range(len(i))
        # plt.plot(x_axis, x1_list, "v:", label='x1')
        # plt.plot(x_axis, x2_list, "o:", label='x2')
        # plt.plot(x_axis, x3_list, ">:", label='x3')
        plt.plot(x_axis, y1_list, "<:", label='y1')
        plt.plot(x_axis, y2_list, "s:", label='y2')
        plt.plot(x_axis, y3_list, "D:", label='y3')
        plt.title(list_tmp[ind])
        plt.legend()
    plt.show()


# 寻找N阶段中演化速度最快的N
def findNFast(k):
    r1, q1, r2, q2, alpha, ea, eb = 2, 20, 1, 15, 3.2, 0.8, 0.8  # 实验2   # e4  确认初始状态
    t = np.arange(0, 100, 0.1)  # 定义时间变化范围及步长
    k_stable = []
    point_stable = []
    point_stable_0 = []
    for n in range(2, k + 1):
        N = n
        list_n = []
        for i in range(2 * N):
            list_n.append(1 / n)
        statStar = tuple(list_n)
        detEquation = Dynamic(N, r1, q1, r2, q2, alpha, ea, eb)
        detResult, traceResult, msgStart = detEquation.calculateTraceDet(statStar[:N], statStar[N:])

        dynamicResult = odeint(detEquation.NDynamicEquation, statStar, t)  # 解n阶动态方程
        stable = True
        for i in range(len(t)):
            if dynamicResult[i, 0] == 0 or dynamicResult[i, 0] == 1:
                stable = False
                k_stable.append(i)
                point_stable.append(tuple(dynamicResult[i, :]))
                point_stable_0.append(1)
                break
        if stable:
            k_stable.append(len(t) - 1)
            point_stable.append(tuple(dynamicResult[len(t) - 1, :]))
            point_stable_0.append(2)
    return k_stable, point_stable, point_stable_0


'''
# 循环查找不同N情况下，E6演化结果
N=2  E6
N=3  1
N=4  2
N=5  E6
N=6  4
N=7  2
N=8  2
N=9  4
N=10 4
更换初始状态，同样N阶段
1. ea，eb
2. r1,r2,q1,q2
3, alpha
'''

def findNE6(k):
    for i in range(2, k + 1):
        t = np.arange(0, 10, 0.1)  # 定义时间变化范围及步长
        r1, q1, r2, q2, alpha, ea, eb = 2, 20, 1, 15, 3.2, 0.8, 0.8  # 实验2   # e4  确认初始状态
        # r1, q1, r2, q2, alpha, ea, eb = 3, 20, 1, 15, 3.2, 0.8, 0.7  # 实验2   # e4  确认初始状态
        # statStar = (0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25)  # 定义初始状态
        N = i
        # 定义方程并初始化
        detEquation = Dynamic(N, r1, q1, r2, q2, alpha, ea, eb)
        statStar, (detM, traceM, msg) = detEquation.e6()
        print('----------      N={}      -----------'.format(N))
        # print('E6初始点为：{}, \n---\ndetM:{} \n----\ntraceM:{} \n---\n msg:{}'.format(statStar, detM, traceM, msg))
        statStar_2 = []
        for ind in statStar:
            statStar_2.append(round(ind, 2))
        print('E6初始点为：{}, \n---\n msg:{}'.format(statStar_2, msg))
        dynamicResult = odeint(detEquation.NDynamicEquation, statStar, t)  # 解n阶动态方程
        # 计算演化后的状态
        k = len(t) - 1
        finalStar = (dynamicResult[k, :])
        finalStar_2 = []
        for ind in finalStar:
            finalStar_2.append(round(ind, 2))
        print('演化结束状态为：{}'.format(finalStar_2))
        print('----------      N={}      -----------'.format(N))


def findNR(N):
    r1, q1, r2, q2, alpha, ea, eb = 14, 20, 1, 15, 15, 0.8, 0.8  # 实验2   # e4  确认初始状态
    for j in range(1, 9 + 1):
        t = np.arange(0, 10, 0.1)  # 定义时间变化范围及步长
        # r1 递增
        eb = 0.1 + j * 0.1
        changePara = eb
        changePara_str = 'eb'
        # 定义方程并初始化
        detEquation = Dynamic(N, r1, q1, r2, q2, alpha, ea, eb)
        statStar, (detM, traceM, msg) = detEquation.e6()
        statStar_2 = []
        for ind in statStar:
            statStar_2.append(round(ind, 2))
        print('----------      N={}, {}={}     -----------'.format(N, changePara_str, changePara))
        # print('E6初始点为：{}, \n---\ndetM:{} \n----\ntraceM:{} \n---\n msg:{}'.format(statStar, detM, traceM, msg))
        print('E6初始点为：{}, \n---\n msg:{}'.format(statStar_2, msg))
        dynamicResult = odeint(detEquation.NDynamicEquation, statStar, t)  # 解n阶动态方程
        # 计算演化后的状态
        k = len(t) - 1
        finalStar = (dynamicResult[k, :])
        finalStar_2 = []
        for ind in finalStar:
            finalStar_2.append(round(ind, 2))
        print('演化结束状态为：{}'.format(finalStar_2))
        print('----------      N={}, {}={}     -----------'.format(N, changePara_str, changePara))

        # 画图
        ax1 = plt.subplot(3, 4, j)
        plt.sca(ax1)
        x_axis = range(len(dynamicResult))
        y_axis = dynamicResult[:, 0]  # x_1的趋势
        plt.ylim(ymax=1, ymin=0)
        plt.plot(x_axis, y_axis, "<:", label='x1')
        title_con = changePara_str + '=' + str(changePara) + ',x1 trend'
        plt.title(title_con)
    plt.show()


'''
初始状态变，梳理一下。 
发现规律
1. 固定q1, r2, q2, alpha, ea, eb  递增 r1  E6以及稳定点逐渐增大，当r1大于13时 稳定点不为E6
2. 固定r1, r2, q2, alpha, ea, eb  递增 q1  E6以及稳定点逐渐减小，当q1在12、13时 稳定点不为E6
3. 固定r1, q1, q2, alpha, ea, eb  递增 r2  E6以及稳定点逐渐减小，当r2在3、4时 稳定点不为E6
'''


def findDifferentStat():
    # 固定阶段为3
    r1, q1, r2, q2, alpha, ea, eb = 14, 20, 1, 15, 15, 0.8, 0.8  # 实验2   # e4  确认初始状态
    statStart_list = [[0.1, 0.1, 0.8, 0.1, 0.1, 0.8], [0.1, 0.4, 0.5, 0.1, 0.4, 0.5], [0.1, 0.7, 0.2, 0.1, 0.7, 0.2],
                      [0.4, 0.4, 0.2, 0.4, 0.4, 0.2], [0.7, 0.1, 0.2, 0.7, 0.1, 0.2], [0.7, 0.2, 0.1, 0.1, 0.2, 0.7],
                      [0.7, 0.1, 0.2, 0.1, 0.2, 0.7], [0.4, 0.2, 0.4, 0.4, 0.2, 0.4], [0.4, 0.1, 0.5, 0.4, 0.5, 0.1],
                      [0.1, 0.4, 0.5, 0.1, 0.5, 0.4]]
    t = np.arange(0, 10, 0.1)  # 定义时间变化范围及步长
    for j, statStar in enumerate(statStart_list):
        detEquation = Dynamic(3, r1, q1, r2, q2, alpha, ea, eb)
        dynamicResult = odeint(detEquation.NDynamicEquation, statStar, t)  # 解n阶动态方程
        # 计算演化后的状态
        k = len(t) - 1
        finalStar = (dynamicResult[k, :])
        statStar_2 = []
        for ind in statStar:
            statStar_2.append(round(ind, 2))
        print('初始点为：{}, '.format(statStar_2))
        finalStar_2 = []
        for ind in finalStar:
            finalStar_2.append(round(ind, 2))
        print('演化结束状态为：{} \n\n\n'.format(finalStar_2))

    #     # 画图
    #     ax1 = plt.subplot(3, 3, j+1)
    #     plt.sca(ax1)
    #     x_axis = range(len(dynamicResult))
    #     y_axis = dynamicResult[:, 0]  # x_1的趋势
    #     plt.ylim(ymax=1, ymin=0)
    #     plt.plot(x_axis, y_axis, "<:", label='x1')
    #     title_con = str(statStar)
    #     plt.title(title_con)
    # plt.show()


def add(c):
    return c + 2


if __name__ == '__main__':
    '''
    定义相关初始变量
    qa >= qb > 0
    ra >= rb > 0
    1 > e > 0
    alpha > rb
    '''
    # t = np.arange(0, 10, 1)  # 定义时间变化范围及步长
    #
    # r1, q1, r2, q2, alpha, ea, eb = 2, 20, 1, 15, 3.2, 0.8, 0.8  # 实验2   # e4  确认初始状态
    # # r1, q1, r2, q2, alpha, ea, eb = 3, 20, 1, 15, 3.2, 0.8, 0.7  # 实验2   # e4  确认初始状态
    # # statStar = (0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25)  # 定义初始状态
    # statStar = (0.5, 0.5, 0.5, 0.5)
    # N = 2
    # # 定义方程并初始化
    # detEquation = Dynamic(N, r1, q1, r2, q2, alpha, ea, eb)
    # # x = (0.83, 0.17)
    # # y = (0.15, 0.85)
    # # star = (x, y)
    #
    # detResult, traceResult, msgStart = detEquation.calculateTraceDet(statStar[:N], statStar[N:])
    # print('初始状态为：{}'.format(statStar))
    # print('初始状态下行列式的det为：{:.2f}， trace为：{:.2f}, 该点的状态为：{}'.format(detResult, traceResult, msgStart))
    #
    # dynamicResult = odeint(detEquation.NDynamicEquation, statStar, t)  # 解n阶动态方程
    # # 计算演化后的状态
    # k = len(t) - 1
    # finalStar = (dynamicResult[k, :])
    # print('演化结束状态为：{}'.format(finalStar))
    # detResultFinal, traceResultFinal, msgFinal = detEquation.calculateTraceDet(finalStar[:N], finalStar[N:])
    # print('演化结束后行列式的det为：{:.2f}， trace为：{:.2f}, 该点的状态为：{}'.format(detResultFinal, traceResultFinal, msgFinal))

    '''
    循环查找输出终点
    '''
    # t = np.arange(0, 10, 1)  # 定义时间变化范围及步长
    # r1, q1, r2, q2, alpha, ea, eb = 3, 20, 1, 15, 3.2, 0.8, 0.7  # 实验2   # e4  确认初始状态
    # N = 3
    # # 定义方程并初始化
    # detEquation = Dynamic(N, r1, q1, r2, q2, alpha, ea, eb)
    #
    # statStart_list = []
    # stopStart_list = []
    # rangelist = [0, 0.3, 0.5, 0.7, 0.9]
    # for i in rangelist:
    #     for j in rangelist:
    #         for z in rangelist:
    #             for y2 in rangelist:
    #                 if i < 1 and j < 1 and i + j < 1 and z + y2 < 1:
    #                     x1, x2, x3, y1, y2, y3 = i, j, 1 - i - j, z, y2, 1 - z - y2
    #                     x1, x2, x3, y1, y2, y3 = round(x1, 1), round(x2, 1), round(x3, 1), round(y1, 1), round(y2,
    #                                                                                                            1), round(
    #                         y3, 1)
    #                     ss = (x1, x2, x3, y1, y2, y3)
    #                     statStart_list.append(ss)
    #
    # print(len(statStart_list))
    #
    # for i in tqdm(statStart_list):
    #     res_finalStart = loopFindFinal(N, t, detEquation, i)
    #     x1, x2, x3, y1, y2, y3 = res_finalStart[0], res_finalStart[1], res_finalStart[2], res_finalStart[3], res_finalStart[4], res_finalStart[5]
    #     x1, x2, x3, y1, y2, y3 = round(x1, 1), round(x2, 1), round(x3, 1), round(y1, 1), round(y2, 1), round(y3, 1)
    #     res_finalStart = [x1,x2,x3, y1,y2, y3]
    #     stopStart_list.append(res_finalStart)
    #
    # dict_start = {
    #     'startStar': statStart_list,
    #     'finalStar': stopStart_list
    # }
    #
    # df_star = pd.DataFrame(dict_start)
    # df_star.to_csv('final_star.csv')

    '''
    画出final_star中分组的数据
    '''
    # plotFinalStar()

    '''
    尝试N阶段时，固定初始值(1/N,....,1/N...),然后查看每个N的演化快慢
    '''
    # k = 50
    # # k_stable  稳定点稳定的次数   point_stable 稳定点结果    point_stable_0  第一的概率是否为1  1为是 2为不是
    # k_stable, point_stable, point_stable_0 = findNFast(k)
    # print(k_stable)
    # x = list(range(len(k_stable)))
    # x = [add(c) for c in x]
    # plt.xlabel('N')
    # plt.ylabel('Iterate')
    # plt.plot(x, k_stable)
    # for ind, i in enumerate(x):
    #     plt.text(i, k_stable[ind], k_stable[ind])
    # plt.show()

    '''
    查询E6点是否符合
    '''
    # t = np.arange(0, 100, 0.1)  # 定义时间变化范围及步长
    # r1, q1, r2, q2, alpha, ea, eb = 2, 20, 1, 15, 3.2, 0.8, 0.8  # 实验2   # e4  确认初始状态
    # # r1, q1, r2, q2, alpha, ea, eb = 3, 20, 1, 15, 3.2, 0.8, 0.7  # 实验2   # e4  确认初始状态
    # # statStar = (0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25)  # 定义初始状态
    # N = 5
    # # 定义方程并初始化
    # detEquation = Dynamic(N, r1, q1, r2, q2, alpha, ea, eb)
    # statStar, (detM, traceM, msg) = detEquation.e6()
    # print('E6初始点为：{}, \n---\ndetM:{} \n----\ntraceM:{} \n---\n msg:{}'.format(statStar, detM, traceM, msg))
    # dynamicResult = odeint(detEquation.NDynamicEquation, statStar, t)  # 解n阶动态方程
    # # 计算演化后的状态
    # k = len(t) - 1
    # finalStar = (dynamicResult[k, :])
    # print('演化结束状态为：{}'.format(finalStar))

    '''
    循环查找不同N情况下，E6演化结果
    '''
    # findNE6(10)

    '''
    讨论不同 r1, q1, r2, q2, alpha, ea, eb的值对演化速度的影响
    '''
    # findNR(4)

    findDifferentStat()