# -*- coding: utf-8 -*-

"""
    @Author  : Mumu
    @Time    : 2022/7/28 15:54
    @Function: 

"""
import pandas as pd
import difflib

file_lightwork = '2021lightWork.xlsx'
# file_lightwork = 'test_light.xlsx'
file_enterprise = 'enterprise_level.xlsx'
df_light = pd.read_excel(file_lightwork)
df_light_columns = df_light.shape[1]

'''
实现删除无效列
'''
del_columns = list(range(df_light_columns))
del del_columns[10]
del del_columns[8]
del del_columns[6]
del del_columns[4]
# 删除所有其他列
df_light.drop(df_light.columns[del_columns], axis=1, inplace=True)

# 把所有公司进行编码，获取企业级次对应的字典。
df_enterprise = pd.read_excel(file_enterprise)
list_enterprise = df_enterprise.iloc[:, 1].tolist()
dict_enterprise = {word: ind for ind, word in enumerate(list_enterprise)}


def del_transform_enterprise(x):
    x = x.split('-')[0]
    res = difflib.get_close_matches(x, list_enterprise)
    encode_enterprise = dict_enterprise[res[0]]
    return encode_enterprise


# 去除单位名称中的‘-’ 并且对企业名称进行编码
df_light['填报单位名称'] = df_light['填报单位名称'].apply(lambda x: del_transform_enterprise(x))
df_light['对方单位名称'] = df_light['对方单位名称'].apply(lambda x: del_transform_enterprise(x))
# jiaFang = df_light['填报单位名称']  # 4
# yiFang = df_light['对方单位名称']  # 6
# incomeSubject = df_light['收入科目']  # 8
# inCome = df_light['收入金额（不含税）']  # 10

df_light.to_csv('test.csv', index=False)


menu_file = 'menu.csv'
df_menu = pd.read_csv(menu_file, encoding='gbk')
jiaFang = df_menu.iloc[:, 0].tolist()
yiFang = df_menu.iloc[:, 1].tolist()
# set_jia