# -*- coding: utf-8 -*-

"""
    @Author  : Mumu
    @Time    : 2022/8/8 17:22
    @Function: 

"""
import numpy

import cv2
import pandas as pd
from PIL import  Image
from PIL import ImageEnhance
from tqdm import tqdm
from HoleCal import Hole
from unet import myUnet


import os
dir_path = r'C:\Users\Administrator\Desktop\8_2'   # 读取数据源路径
# 解析路径底下各图片路径和图片名称
path_f = os.listdir(dir_path)
img_path_list = []
for dirs in path_f:
    filepath = dir_path + '/' + dirs
    path_img_file2 = os.listdir(filepath)
    for path_img_file in path_img_file2:
        path_img = filepath + '/' + path_img_file
        path_img_list = os.listdir(path_img)
        for pathImg in path_img_list:
            imgPath = path_img + '/' + pathImg
            img_path_list.append(imgPath)

print('numbers:',len(img_path_list))

model_512 = r'C:\Users\Administrator\Desktop\data\sample\2_120.hdf5'
# 定义模型并读取模型参数
modelUnet = myUnet(img_cols=1024, img_rows=1024)
model = modelUnet.get_unet()
model.load_weights(model_512)

# 预测函数，输入参数为 图片内容和文件名称
def predict(image, filename):
    # 修改图片size
    image_new = cv2.resize(image, (1024, 1024))
    # 将图片转为灰度图
    image_gray = cv2.cvtColor(image_new, cv2.COLOR_RGB2GRAY)
    # image = image[:, :, 0]
    # 归一化
    image = image_gray/255
    # 取平均值，并标准化到-0.5 ~ 0.5
    mean = image.mean(axis=0)
    image -= mean
    # 创建1通道0矩阵并赋值给imaes
    images = numpy.zeros(shape=(1024, 1024, 1))
    images[:, :, 0] = image
    test_image = numpy.expand_dims(images, axis=0)  # 扩展维度

    pred = model.predict(test_image, batch_size=1, verbose=1)  # 对处理后的数据进行模型预测
    pred_save = pred[0]  # 获取预测结果
    pred_save = pred_save.reshape(1,1024,1024)  # 进行图像维度转化
    # pred_save = pred_save.reshape(1024,1024,1)
    pred_save = pred_save[0]  # 获取单通道图像
    # pred_save += mean
    pred_save = pred_save*255  # 反归一化
    pred_save = cv2.resize(pred_save, (1600, 1200))  # 对图像进行resize处理

    savepath = r'C:\Users\Administrator\Desktop\predict'  # 目标存储路径
    newfilelist = filename.split('\\')[-1].split('/')  # 接卸图片存储路径
    newfilename = ''
    # 对新存储的路径如果不存在该路径则新建对应路径文件夹
    for ind, i in enumerate(newfilelist):
        if ind == 0:
            newfilename = savepath + '/' + i
        elif ind == len(newfilelist)-2:
            newfilename = newfilename + '/' + i
            try:
                os.makedirs(newfilename)
                print('ok')
            except:
                pass

        else:
            newfilename = newfilename + '/' + i

    # 存储图片
    cv2.imwrite(newfilename, pred_save)
    img = Image.open(newfilename) # 读取图片
    enh_con = ImageEnhance.Contrast(img)  # 对图像进行增强
    contrast = 1.5
    img_contrasted = enh_con.enhance(contrast)  # 根据阈值进行推向增强
    img_contrasted.save(newfilename)  # 保存图像增强后的图像
    img = cv2.imread(newfilename)  # 读取图像数据
    meanmax, new_list = Hole(img, newfilename)  # 调用Hole方法获取椭圆拟合数据
    line_path_data = [filename, meanmax]
    line_path_data.extend(new_list)
    return line_path_data  # 返回更新后的文件名称和平均值等数据


if __name__ == '__main__':
    hoal_list = []
    # 循环预测图片
    for img in tqdm(img_path_list):
        image = cv2.imread(img)
        line_data = predict(image, img)
        hoal_list.append(line_data)
    # 对更新后的文件名称和平均值等数据进行保存
    df_hoal = pd.DataFrame(hoal_list)
    # df_hoal.to_csv('hoalTotal.csv', index=False)
    df_hoal.to_excel('hoalTotal.xls', index=False)
    print('ok')



