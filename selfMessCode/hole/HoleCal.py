# -*- coding: utf-8 -*-

"""
    @Author  : Mumu
    @Time    : 2022/8/4 18:34
    @Function: 

"""

import cv2, math
import numpy as np



# 拟合图片地址

# 这个方法没有用到
def fillHole(im_in):
    im_floodfill = im_in.copy()  # 浅拷贝数据
    h, w = im_in.shape[:2]   # 输出数据的维度
    mask = np.zeros((h + 2, w + 2), np.uint8)  # 定义大两个维度的0矩阵
    cv2.floodFill(im_floodfill, mask, (0, 0), 255) #利用泛洪填充算法填充0矩阵
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)  #对数据进行按位取反运算
    # Combine the two images to get the foreground.
    im_out = (im_in.all() or im_floodfill_inv.all())
    return im_out


def Hole(img_cv, filename):
    # img_cv = cv2.imread(dirpath)
    # img_cv = cv2.cvtColor(img_cv, cv2.COLOR_GRAY2BGR)
    gray_img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)  # 将灰度图改为BGR图
    retval, dst = cv2.threshold(gray_img, 23, 255, cv2.THRESH_BINARY)  # 对图片进行阈值化  小于23 为0 大于23 为255
    # imf = fillHole(dst)
    dst1 = dst.copy()  # 浅拷贝数据
    cv2.bitwise_not(dst1, dst)    #对数据进行按位取反运算
    extended = cv2.copyMakeBorder(dst, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[255, 255, 255])   # 对图像增加一个边框
    mh, mw = extended.shape[:2]   # 对扩展前图像的维度进行取值
    mask = np.zeros([mh + 2, mw + 2], np.uint8)  # 定义大原图两个维度的0矩阵
    cv2.floodFill(extended, mask, (5, 5), (0, 0, 0), flags=cv2.FLOODFILL_FIXED_RANGE)  # 利用泛洪填充算法填充0矩阵
    cropImg = extended[10:10 + img_cv.shape[0], 10:10 + img_cv.shape[1]]   # 取出截图的部分图像
    # ret, labels, stats, centroid = cv2.connectedComponentsWithStats(cropImg)
    contours, hierarchy = cv2.findContours(cropImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)   # 查找物体的轮廓信息
    # cnt, hierarchy = cv2.findContours(cropImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_area = []
    # calculate area and filter into new array
    for con in contours:  # 将检测出的轮廓信息一一列出
        # area = cv2.contourArea(con)
        contours_area.append(con)

    contours_cirles = []
    circularity_a = []
    perimeter_a = []
    ellipse_a = []

    ellipse_x = []
    ellipse_y = []
    ellipse_min = []
    ellipse_max = []
    ellipse_ro = []
    # check if contour is of circular shape
    a = 0
    rate = 2.7

    # 对每个轮廓信息进行处理
    for con in contours_area:

        perimeter = cv2.arcLength(con, True)  # 计算轮廓长度
        perimeter_a.append(perimeter)
        area = cv2.contourArea(con)  # 计算轮廓面积

        # 先看是否满足周长和面积的条件，再对成圆率做限定，
        if len(con) <= 4 or area < 40:
            pass
        else:
            # 计算长度
            circularity = 4 * math.pi * (area / (perimeter * perimeter))
            circularity_a.append(circularity)
            if 0.65 <= circularity <= 1.3:  # 给定长度限定条件
                # if 0.2<= circularity <= 1.3:
                a = a + 1  # 累加，记录轮廓个数
                contours_cirles.append(con)
                ellips = cv2.fitEllipse(con)  # 返回椭圆轮廓点参数

                # 绘制椭圆 设置边值
                ellipse = list([list(ellips[0]), list(ellips[1]), (ellips[2])])
                ellipse[1][0] = ellips[1][0] + 4
                ellipse[1][1] = ellips[1][1] + 4
                ellipse[0][0] = ellips[0][0]
                ellipse[0][1] = ellips[0][1]
                ellipse[2] = ellips[2]
                cv2.ellipse(img_cv, ellipse, (0, 0, 255), 1)
                ellipse_a.append(ellipse)
                ellipse_x.append(ellipse[0][0])
                ellipse_y.append(ellipse[0][1])
                ellipse_min.append(ellipse[1][0] / 2.7)
                ellipse_max.append(ellipse[1][1] / 2.7)
                ellipse_ro.append(ellipse[2])
                ellipse[1][0] = (ellipse[1][0]) / rate
                ellipse[1][1] = (ellipse[1][1]) / rate

    # 设置范围0-5,5-7,7-9,9-11,11+
    rangeA = [5, 7, 9, 11]
    rangeX = np.zeros((len(rangeA) + 1, 1))
    maxa = 0

    # 获取最大周长的值列表
    for i in ellipse_max:
        for j in range(len(rangeA) + 1):
            if j == 0:
                if i < rangeA[j]:
                    rangeX[j] = rangeX[j] + 1

            elif j == len(rangeA):
                if i >= rangeA[j - 1]:
                    rangeX[j] = rangeX[j] + 1
            else:
                if rangeA[j - 1] <= i < rangeA[j]:
                    rangeX[j] = rangeX[j] + 1

    new_list = [x[0] / len(ellipse_max) for x in rangeX]

    for x in ellipse_max:
        maxa = x + maxa
    meanmax = maxa / len(ellipse_max)

    # save pic
    cv2.imwrite(filename, img_cv)
    print('保存成功，path:', filename)
    return meanmax, new_list


# if __name__ == '__main__':
#     dirpath = r'C:\Users\Administrator\Desktop\predict\8_2_1_00_1.jpeg'
#     img_cv = cv2.imread(dirpath)
#     meanmax, new_list = Hole(img_cv)
#     print('-------------', meanmax, new_list)
#     line_date = ['dsad.jpg', meanmax]
#     line_date.extend(new_list)
#     line_date1 = line_date
#     line_date2 = line_date1
#     dic = {
#         'a': line_date,
#         'b': line_date1,
#         'c': line_date2,
#     }
#     df = pd.DataFrame(dic)
#     df2 = df.T
#     print(df2.head())
