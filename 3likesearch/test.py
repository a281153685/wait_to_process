# -*- coding: utf-8 -*-

"""
    @Author  : Mumu
    @Time    : 2022/8/10 15:56
    @Function: 

"""
import random
import pandas as pd

# 固定随机种子
random.seed(1)

# 读取菜单列表
menu_file = 'menu.csv'
df_menu = pd.read_csv(menu_file, encoding='gbk')
jiaFang = df_menu.iloc[:, 0].tolist()
yiFang = df_menu.iloc[:, 1].tolist()
set_jia = set(jiaFang)
set_yi = set(yiFang)
len_sum = len(set_yi) + len(set_jia)
# print(len_sum)
list_all = list(set_jia) + list(set_yi)
# print(list_all)
dict_all = {word: ind for ind, word in enumerate(list_all)}
# print(dict_all)


def del_transform_enterprise(x):
    encode_enterprise = dict_all[x]
    return encode_enterprise


df_menu['1'] = df_menu['1'].apply(lambda x: del_transform_enterprise(x))
df_menu['2'] = df_menu['2'].apply(lambda x: del_transform_enterprise(x))

# print(df_menu.head())

numpy_len = [[0 for i in range(len_sum)] for j in range(len_sum)]

for i in range(len_sum):
    row = df_menu.iloc[i, 0]
    colunm = df_menu.iloc[i, 1]
    numpy_len[row][colunm] = 1
# print(numpy_len)

df_test = pd.read_csv('test.csv')
df_test['收入科目'] = df_test['收入科目'].apply(lambda x: del_transform_enterprise(x))
# print(df_test.head())

df_test['填报单位名称1'] = ['' for i in range(len(df_test))]
df_test['对方单位名称1'] = ['' for i in range(len(df_test))]
df_test['收入科目1'] = ['' for i in range(len(df_test))]
df_test['收入金额1'] = ['' for i in range(len(df_test))]
print(df_test.head())

def_list_index = []
add_list = []
for ind in range(len(df_test)):
    if ind in def_list_index:
        continue
    jia = df_test.iloc[ind, 0]
    yi = df_test.iloc[ind, 1]
    kemu_1 = df_test.iloc[ind, 2]
    count_1 = df_test.iloc[ind, 3]

    kemu_zuiyou = ''
    count_zuiyou = 0.
    inds_zuiyou = 0
    overlap = 0
    opendown = False
    for inds in range(ind, len(df_test)):
        if inds in def_list_index:
            continue
        if df_test.iloc[inds, 1] == jia and df_test.iloc[inds, 0] == yi:
            opendown = True
            kemu_2 = df_test.iloc[inds, 2]
            if numpy_len[kemu_1][kemu_2] == 1:
                count = df_test.iloc[inds, 3]
                diffirence = abs(count - count_1)
                if overlap == 0 or overlap > diffirence:
                    kemu_zuiyou = kemu_2
                    count_zuiyou = count
                    inds_zuiyou = inds
                    overlap = diffirence

        # 查到了最后一行，进行数据更新
        if inds == len(df_test) - 1 and opendown:
            def_list_index.append(inds_zuiyou)
            add_list.append((ind, kemu_zuiyou, count_zuiyou))

print(add_list)

new_dict = {v: k for k, v in dict_all.items()}
for ind, kemu_zuiyou, count_zuiyou in add_list:
    df_test.iloc[ind, 6] = new_dict[kemu_zuiyou]
    df_test.iloc[ind, 7] = count_zuiyou
df_test.drop(def_list_index, axis=0, inplace=True)
print(df_test.head())


