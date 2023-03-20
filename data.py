import os
import pandas as pd
import json
import numpy as np
import random
from copy import copy

# 从config中读出数据集路径
with open('./config.json', 'r') as f:
    s = f.read()
    config = json.loads(s)
data_path = config['data_path']
dataset_label_1_name =  config['dataset_label_1_name']
dataset_label_2_name =  config['dataset_label_2_name']
dataset_label_3_name =  config['dataset_label_3_name']

# 数据预处理，从文件中读取了原始文本之后，进行文本内容的修改或添加。
def preProcess(text: str):
    # 暂时没有预处理过程
    return text

def readTrainData(need_test_set = False):
    total_item = {}
    for filename in os.listdir(data_path):
        if(filename.lower().endswith('.xls') or filename.lower().endswith('xlsx')):
            try:
                df = pd.read_excel(data_path + filename)
                df.columns = [s.strip() for s in df.columns.tolist()]
                for row in df.index.values:
                    id = str(df.loc[row, '工单编号'])
                    title = df.loc[row, '诉求标题']
                    if(not pd.isnull(df.loc[row, '市民原始诉求'])):
                        text = str(df.loc[row, '市民原始诉求'])
                    else:
                        text = ('' if pd.isnull(df.loc[row, '涉事主体']) else '涉事主体:' + str(df.loc[row, '涉事主体'])) + '\n' +\
                            ('' if pd.isnull(df.loc[row, '主体地址']) else '主体地址:' + str(df.loc[row, '主体地址'])) + '\n' +\
                            ('' if pd.isnull(df.loc[row, '事发地点']) else '事发地点:' + str(df.loc[row, '事发地点'])) + '\n' +\
                            ('' if pd.isnull(df.loc[row, '标签组']) else '标签组:' + str(df.loc[row, '标签组'])) + '\n' +\
                            ('' if pd.isnull(df.loc[row, '市民诉求']) else '市民诉求:' + str(df.loc[row, '市民诉求'])) + '\n' +\
                            ('' if pd.isnull(df.loc[row, '补充信息']) else '补充信息:' + str(df.loc[row, '补充信息']))
                    text = preProcess(text)
                    label_level_1 = 'undefine' if pd.isnull(df.loc[row, '办理部门一级']) else str(df.loc[row, '办理部门一级'])
                    label_level_2 = 'undefine' if pd.isnull(df.loc[row, '办理部门二级']) else str(df.loc[row, '办理部门二级'])
                    label_level_3 = 'undefine' if pd.isnull(df.loc[row, '办理部门三级']) else str(df.loc[row, '办理部门三级'])
                    total_item[str(id)] = {
                        'title': title,
                        'text': text,
                        'label_level_1': label_level_1,
                        'label_level_2': label_level_2,
                        'label_level_3': label_level_3
                    }
            except Exception as e:
                print(e)
                print('文件{}内容有误，请检查'.format(data_path + filename))

    print('文件统计出训练样本{}条'.format(len(total_item)))
    
    if need_test_set:
        ids = list(total_item.keys())
        random.shuffle(ids)
        train_item = {}
        for id in ids[:len(ids) * 4 // 5]:
            train_item[id] = total_item[id]
        
        test_item = {}
        for id in ids[len(ids) * 4 // 5:]:
            test_item[id] = total_item[id]

        del total_item

    train_set_id_1, train_set_id_2, train_set_id_3 = dataEnhance(train_item)

    train_set_1 = {}
    train_set_2 = {}
    train_set_3 = {}
    for id in train_set_id_1:
        train_set_1[id] = copy(train_item[id])
        train_set_1[id].pop('label_level_2')
        train_set_1[id].pop('label_level_3')
    for id in train_set_id_2:
        train_set_2[id] = copy(train_item[id])
        train_set_2[id].pop('label_level_1')
        train_set_2[id].pop('label_level_3')
    for id in train_set_id_3:
        train_set_3[id] = copy(train_item[id])
        train_set_3[id].pop('label_level_1')
        train_set_3[id].pop('label_level_2')
    print('数据增强后1级标签训练集样本{}条'.format(len(train_set_1)))
    print('数据增强后2级标签训练集样本{}条'.format(len(train_set_2)))
    print('数据增强后3级标签训练集样本{}条'.format(len(train_set_3)))

    with open(data_path + dataset_label_1_name, 'w') as f:
        s = json.dumps(train_set_1, ensure_ascii=False)
        f.write(s)
    with open(data_path + dataset_label_2_name, 'w') as f:
        s = json.dumps(train_set_2, ensure_ascii=False)
        f.write(s)
    with open(data_path + dataset_label_3_name, 'w') as f:
        s = json.dumps(train_set_3, ensure_ascii=False)
        f.write(s)

    if need_test_set:
        with open(data_path + 'test_set.json', 'w') as f:
            s = json.dumps(test_item, ensure_ascii=False)
            f.write(s)

# ====数据增强（适配特定数据集）====

# 输入：train_item字典，输出：3个数据集的下标列表
def dataEnhance(train_item: dict):
    label_count1 = {}
    label_count2 = {}
    label_count3 = {}

    train_set_id_1 = []
    train_set_id_2 = []
    train_set_id_3 = []

    # 调整一级标签===
    too_much_tag_id = []
    for id in train_item:
        if train_item[id]['label_level_1'] == '荔湾区政府':
            too_much_tag_id.append(id)
        else:
            train_set_id_1.append(id)
    random.shuffle(too_much_tag_id)
    train_set_id_1 += too_much_tag_id[0:10000]

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