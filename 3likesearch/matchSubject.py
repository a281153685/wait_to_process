# -*- coding: utf-8 -*-

"""
    @Author  : Mumu
    @Time    : 2022/8/11 15:00
    @Function: 

"""
import pandas as pd
import difflib

file_lightwork = '2021lightWork.xlsx'
file_enterprise = 'enterprise_level.xlsx'
df_light = pd.read_excel(file_lightwork)
df_light_columns = df_light.shape[1]

# 把所有公司进行编码，获取企业级次对应的字典。
df_enterprise = pd.read_excel(file_enterprise)
enterpriseList = df_enterprise.iloc[:, 1].tolist()  # 第2列为公司名称
enterpriseDict = {word: ind for ind, word in enumerate(enterpriseList)}  # 获取公司对序号的字典{公司：序号}
enterpriseDictTranslate = {v: k for k, v in enterpriseDict.items()}  # 根据序号对应公司的字典{序号：公司}


