# -*- coding: utf-8 -*-

"""
    @Author  : Mumu
    @Time    : 2022/8/6 20:54
    @Function: 

"""

import cv2, math
import numpy as np
import xlsxwriter as xw

dirpath = r'1.jpg'


# 拟合图片地址


def fillHole(im_in):
    im_floodfill = im_in.copy()

    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = im_in.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0, 0), 255);

    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    # Combine the two images to get the foreground.
    im_out = im_in | im_floodfill_inv

    return im_out


img_cv = cv2.imread(dirpath)
# cv2.imshow("img_cvmage", img_cv)
gray_img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
retval, dst = cv2.threshold(gray_img, 23, 255, cv2.THRESH_BINARY)
# cv2.imshow("binary",dst)
imf = fillHole(dst)
# cv2.imshow("imfmage", imf)
dst1 = dst.copy()
cv2.bitwise_not(dst1, dst)
extended = cv2.copyMakeBorder(dst, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[255, 255, 255])
mh, mw = extended.shape[:2]
mask = np.zeros([mh + 2, mw + 2], np.uint8)
cv2.floodFill(extended, mask, (5, 5), (0, 0, 0), flags=cv2.FLOODFILL_FIXED_RANGE)
cropImg = extended[10:10 + img_cv.shape[0], 10:10 + img_cv.shape[1]]
# cv2.imshow("cropImgmage", cropImg)

ret, labels, stats, centroid = cv2.connectedComponentsWithStats(cropImg)
# get contours

contours, hierarchy = cv2.findContours(cropImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnt, hierarchy = cv2.findContours(cropImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

contours_area = []

# calculate area and filter into new array

for con in contours:
    area = cv2.contourArea(con)

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
fileName = "统计.xlsx"
workbook = xw.Workbook(fileName)  # 创建工作簿
worksheet1 = workbook.add_worksheet("sheet1")  # 创建子表
worksheet1.activate()  # 激活表
title = ['ellipse_x', 'ellipse_y', 'ellipse_min', 'ellipse_max', 'ellipse_rot']  # 设置表头
worksheet1.write_row('A1', title)  # 从A1单元格开始写入表头
# check if contour is of circular shape
a = 0
rate = 2.7

# def cal_distance(p1, p2):
#     return math.sqrt(math.pow((p1 - p2),2) + math.pow(p1 - p2,2))


for con in contours_area:

    perimeter = cv2.arcLength(con, True)
    perimeter_a.append(perimeter)
    area = cv2.contourArea(con)

    # 先看是否满足周长和面积的条件，再对成圆率做限定，
    if len(con) <= 4 or area < 40:

        1
    else:

        circularity = 4 * math.pi * (area / (perimeter * perimeter))
        circularity_a.append(circularity)
        if 0.05 <= circularity <= 1.3:
            a = a + 1;
            contours_cirles.append(con)
            ellips = cv2.fitEllipse(con)
            ellipse = list([list(ellips[0]), list(ellips[1]), (ellips[2])]);
            ellipse[1][0] = ellips[1][0] + 4;
            ellipse[1][1] = ellips[1][1] + 4;
            ellipse[0][0] = ellips[0][0]
            ellipse[0][1] = ellips[0][1]
            ellipse[2] = ellips[2]
            # 绘制椭圆
            cv2.ellipse(img_cv, ellipse, (0, 0, 255), 1)
            ellipse_a.append(ellipse)
            ellipse_x.append(ellipse[0][0])
            ellipse_y.append(ellipse[0][1])
            ellipse_min.append(ellipse[1][0] / 2.7)
            ellipse_max.append(ellipse[1][1] / 2.7)
            ellipse_ro.append(ellipse[2])
            row = 'A' + str(a + 1)
            ellipse[1][0] = (ellipse[1][0]) / rate
            ellipse[1][1] = (ellipse[1][1]) / rate
            worksheet1.write_row(row, [ellipse[0][0], ellipse[0][1], (ellipse[1][0]), (ellipse[1][1]), ellipse[2]])
            # p1 = ellipse[0][0]
            # p2 = ellipse[0][1]
            # p3 = ellipse[2]
            # di = cal_distance(p1, p2)
            # print(ellipse[0],ellipse[1][0])

# ellipse[1][0] = (ellipse[1][0])/2.7
# ellipse[1][1] = (ellipse[1][1])/2.7

cv2.imshow("output", img_cv)
# 设置范围0-5,5-7,7-9,9-11,11+
# rangeA = [5, 7, 9, 11];
# rangeX = np.zeros((len(rangeA) + 1, 1));
# maxa = 0;
# print(range(len(rangeA)))
# # n1=0
#
# for i in ellipse_max:
#     for j in range(len(rangeA) + 1):
#         if j == 0:
#             if i < rangeA[j]:
#                 # print('a')
#                 rangeX[j] = rangeX[j] + 1
#
#         elif j == len(rangeA):
#             if i >= rangeA[j - 1]:
#                 # print('b')
#                 rangeX[j] = rangeX[j] + 1
#         else:
#             if (i >= rangeA[j - 1] and i < rangeA[j]):
#                 rangeX[j] = rangeX[j] + 1
#                 # print([rangeA[j - 1],i, rangeA[j]])
#                 # print(rangeX[j])
#
# worksheet2 = workbook.add_worksheet("sheet2")  # 创建子表
# worksheet2.activate()  # 激活表
# worksheet2.write_row("B1", rangeA)
#
# new_list = [x / len(ellipse_max) for x in rangeX]
# # new_list = new_list / 2.7
# worksheet2.write_row("B2", str(new_list))
#
# worksheet2.write(0, 0, "mean")
# for x in ellipse_max:
#     maxa = x + maxa
# meanmax = maxa / len(ellipse_max)
# meanmax = meanmax
# worksheet2.write(1, 0, meanmax)
# workbook.close()  # 关闭表
# print(len(contours_cirles))
cv2.waitKey(0)
cv2.destroyAllWindows()
