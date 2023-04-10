import os
import pandas as pd
import json
import numpy as np
import random
import re
from tqdm import tqdm
from copy import copy
from utils import resetLabelLv2, resetLabelLv3

# 从config中读出数据集路径
with open('./config.json', 'r') as f:
    s = f.read()
    config = json.loads(s)
data_path = config['data_path']

# 数据预处理，从文件中读取了原始文本之后，进行文本内容的修改或添加。
def preProcess(text: str):
    # 保留
    text = re.sub(r'[^\u4e00-\u9fa50-9a-zA-z]+', '', text)
    return text

def readTrainData(need_dev_set = True):
    total_item = []
    label_dict = {}
    for filename in os.listdir(data_path):
        if(filename.lower().endswith('.xls') or filename.lower().endswith('xlsx')):
            try:
                df = pd.read_excel(data_path + filename)
                df.columns = [s.strip() for s in df.columns.tolist()]
                for row in tqdm(df.index.values, desc='reading {}:'.format(filename)):
                    id = str(df.loc[row, '工单编号'])
                    title = df.loc[row, '诉求标题']
                    if(not pd.isnull(df.loc[row, '市民原始诉求'])):
                        text_raw = str(df.loc[row, '市民原始诉求'])
                    else:
                        text_raw = ''
                    text_merge = ('' if pd.isnull(df.loc[row, '涉事主体']) else '涉事主体:' + str(df.loc[row, '涉事主体'])) + '\n' +\
                        ('' if pd.isnull(df.loc[row, '主体地址']) else '主体地址:' + str(df.loc[row, '主体地址'])) + '\n' +\
                        ('' if pd.isnull(df.loc[row, '事发地点']) else '事发地点:' + str(df.loc[row, '事发地点'])) + '\n' +\
                        ('' if pd.isnull(df.loc[row, '标签组']) else '标签组:' + str(df.loc[row, '标签组'])) + '\n' +\
                        ('' if pd.isnull(df.loc[row, '市民诉求']) else '市民诉求:' + str(df.loc[row, '市民诉求'])) + '\n' +\
                        ('' if pd.isnull(df.loc[row, '补充信息']) else '补充信息:' + str(df.loc[row, '补充信息']))
                    text_raw = preProcess(text_raw)
                    text_merge = preProcess(text_merge)
                    label_level_1 = '无对应一级办理部门' if pd.isnull(df.loc[row, '办理部门一级']) else str(df.loc[row, '办理部门一级'])
                    label_level_2 = resetLabelLv2('无对应二级办理部门' if pd.isnull(df.loc[row, '办理部门二级']) else str(df.loc[row, '办理部门二级']))
                    label_level_3 = resetLabelLv3('无对应三级办理部门' if pd.isnull(df.loc[row, '办理部门三级']) else str(df.loc[row, '办理部门三级']))
                    total_item.append({
                        'id': id,
                        'title': title,
                        'text': text_raw,
                        'label_level_1': label_level_1,
                        'label_level_2': label_level_2,
                        'label_level_3': label_level_3
                    })
                    total_item.append({
                        'id': id,
                        'title': title,
                        'text': text_merge,
                        'label_level_1': label_level_1,
                        'label_level_2': label_level_2,
                        'label_level_3': label_level_3
                    })
                    # label树形式: {label_1: {label2: [label3, ...], ...}}
                    if label_level_1 not in label_dict:
                        label_dict[label_level_1] = {}
                    if label_level_2 not in label_dict[label_level_1]:
                        label_dict[label_level_1][label_level_2] = []
                    if label_level_3 not in label_dict[label_level_1][label_level_2]:
                        label_dict[label_level_1][label_level_2].append(label_level_3)
            except Exception as e:
                print(e)
                print('文件{}内容有误，请检查'.format(data_path + filename))

    print('文件统计出训练样本{}条'.format(len(total_item)))
    
    train_item = {}
    dev_item = {}
    if need_dev_set:
        random.shuffle(total_item)
        train_item = total_item[:len(total_item) * 4 // 5]
        dev_item = total_item[len(total_item) * 4 // 5:]

        del total_item

    print('划分成 训练集样本{}条，测试集样本{}条'.format(len(train_item), len(dev_item)))

    # 写训练用文件
    with open(data_path + 'label.txt', 'w') as f:
        for label_1 in label_dict:
            f.write(label_1 + '\n')
            for label_2 in label_dict[label_1]:
                f.write(label_1 + '##' + label_2 + '\n')
                for label_3 in label_dict[label_1][label_2]:
                    f.write(label_1 + '##' + label_2 + '##' + label_3 + '\n')
    
    with open(data_path + 'train.txt', 'w') as f:
        for item in train_item:
            if item['text'] == '':
                continue
            f.write(item['text'] + '\t' +\
                    item['label_level_1'] + ',' + \
                    item['label_level_1'] + '##' + item['label_level_2'] + ',' + \
                    item['label_level_1'] + '##' + item['label_level_2'] + '##' + item['label_level_3'] + '\n'
            )

    with open(data_path + 'dev.txt', 'w') as f:
        for item in dev_item:
            if item['text'] == '':
                continue
            f.write(item['text'] + '\t' +\
                    item['label_level_1'] + ',' + \
                    item['label_level_1'] + '##' + item['label_level_2'] + ',' + \
                    item['label_level_1'] + '##' + item['label_level_2'] + '##' + item['label_level_3'] + '\n'
            )

# ====数据增强（适配特定数据集）====
'''暂时不使用手动数据增强'''
# 输入：train_item字典，输出：3个数据集的下标列表
def dataEnhance(train_item: dict):
    label_count1 = {}
    label_count2 = {}
    label_count3 = {}

    train_set_id_1 = []
    train_set_id_2 = []
    train_set_id_3 = []

    # 调整一级标签===
    # too_much_tag_id = []
    # for id in train_item:
    #     if train_item[id]['label_level_1'] == '荔湾区政府':
    #         too_much_tag_id.append(id)
    #     else:
    #         train_set_id_1.append(id)
    # random.shuffle(too_much_tag_id)
    # train_set_id_1 += too_much_tag_id[0:10000]
    train_set_id_1 = train_item.keys()

    #==============

    # 调整二级标签===
    train_set_id_2 = train_item.keys()

    #==============

    # 调整三级标签===
    train_set_id_3 = train_item.keys()

    #==============
    return train_set_id_1, train_set_id_2, train_set_id_3

# ====END 数据增强部分====

if __name__ == '__main__':
    readTrainData(True)