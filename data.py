import os
import pandas as pd
import numpy as np
import random
import re
import json
from tqdm import tqdm
from copy import copy
from utils import resetLabelLv2, resetLabelLv3
from config import config
 
def preProcess(text: str):
    '''
    数据预处理，从文件中读取了原始文本之后，进行文本内容的修改或添加。
    目前暂时是一个保留接口，功能仅保留文本中的中文字符、数字、英文字母
    Input:
        - text: 待处理文本
    Ouput:
        - text: 处理后文本
    '''
    text = re.sub(r'[^\u4e00-\u9fa50-9a-zA-z]+', '', text)
    return text

def readTrainData(need_test_set = True):
    total_item = []
    label_dict = {}
    for filename in os.listdir(config.path.data):
        if(filename.lower().endswith('.xls') or filename.lower().endswith('xlsx')):
            try:
                df = pd.read_excel(os.path.join(config.path.data, filename))
                for row in tqdm(df.index.values):
                    id = str(df.loc[row, '工单编号'])
                    title = str(df.loc[row, '标题'])
                    text = preProcess('' if pd.isnull(df.loc[row, '内容']) else str(df.loc[row, '内容']))
                    label_1 = '无' if pd.isnull(df.loc[row, '一级']) else str(df.loc[row, '一级'])
                    label_2 = '无' if pd.isnull(df.loc[row, '二级']) else str(df.loc[row, '二级'])
                    label_3 = '无' if pd.isnull(df.loc[row, '三级']) else str(df.loc[row, '三级'])
                    label_4 = '无' if pd.isnull(df.loc[row, '四级']) else str(df.loc[row, '四级'])
                    total_item.append({
                        'title': title,
                        'text': text,
                        'label_1': label_1,
                        'label_2': label_2,
                        'label_3': label_3,
                        'label_4': label_4,
                    })
                    # label树形式: {label_1: {label2: [label3, ...], ...}}
                    if label_1 not in label_dict:
                        label_dict[label_1] = {}
                    if label_2 not in label_dict[label_1]:
                        label_dict[label_1][label_2] = {}
                    if label_3 not in label_dict[label_1][label_2]:
                        label_dict[label_1][label_2][label_3] = []
                    if label_4 not in label_dict[label_1][label_2][label_3]:
                        label_dict[label_1][label_2][label_3].append(label_4)
            except Exception as e:
                print(e)
                print('文件{}内容有误，请检查'.format(os.path.join(config.path.data, filename)))

    print('文件统计出训练样本{}条'.format(len(total_item)))
    
    train_item = []
    dev_item = []
    test_item = []
    random.shuffle(total_item)
    if need_test_set:
        train_item = total_item[:len(total_item) * 8 // 10]
        dev_item = total_item[len(total_item) * 8 // 10: len(total_item) * 9 // 10]
        test_item = total_item[len(total_item) * 9 // 10:]
        print('划分成 训练集-样本{}条，验证集-样本{}条，测试集-样本{}条'.format(len(train_item), len(dev_item), len(test_item)))
    else:
        train_item = total_item[:len(total_item) * 8 // 10]
        dev_item = total_item[len(total_item) * 9 // 10:]
        print('划分成 训练集-样本{}条，验证集-样本{}条'.format(len(train_item), len(dev_item)))

    # 写训练用文件
    with open(os.path.join(config.path.data, 'label.txt'), 'w') as f:
        for label_1 in label_dict:
            f.write(label_1 + '\n')
            for label_2 in label_dict[label_1]:
                f.write(label_1 + '##' + label_2 + '\n')
                for label_3 in label_dict[label_1][label_2]:
                    f.write(label_1 + '##' + label_2 + '##' + label_3 + '\n')
                    for label_4 in label_dict[label_1][label_2][label_3]:
                        f.write(label_1 + '##' + label_2 + '##' + label_3 + '\n')
    
    with open(os.path.join(config.path.data, 'train.txt'), 'w') as f:
        for item in train_item:
            if item['text'] == '':
                continue
            f.write(
                item['text'] + '\t' +\
                item['label_1'] + ',' + \
                item['label_1'] + '##' + item['label_2'] + ',' + \
                item['label_1'] + '##' + item['label_2'] + '##' + item['label_3'] + ',' + \
                item['label_1'] + '##' + item['label_2'] + '##' + item['label_3'] + '##' + item['label_4'] + '\n'
            )

    with open(os.path.join(config.path.data, 'dev.txt'), 'w') as f:
        for item in dev_item:
            if item['text'] == '':
                continue
            f.write(
                item['text'] + '\t' +\
                item['label_1'] + ',' + \
                item['label_1'] + '##' + item['label_2'] + ',' + \
                item['label_1'] + '##' + item['label_2'] + '##' + item['label_3'] + ',' + \
                item['label_1'] + '##' + item['label_2'] + '##' + item['label_3'] + '##' + item['label_4'] + '\n'
            )

    if need_test_set:
        with open(os.path.join(config.path.data, 'test.txt'), 'w') as f:
            for item in test_item:
                if item['text'] == '':
                    continue
                f.write(
                    item['text'] + '\t' +\
                    item['label_1'] + ',' + \
                    item['label_1'] + '##' + item['label_2'] + ',' + \
                    item['label_1'] + '##' + item['label_2'] + '##' + item['label_3'] + ',' + \
                    item['label_1'] + '##' + item['label_2'] + '##' + item['label_3'] + '##' + item['label_4'] + '\n'
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
    #     if train_item[id]['label_1'] == '荔湾区政府':
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

def readTestSet():
    test_item = []
    with open(os.path.join(config.path.data, 'test_set.json'), 'r') as f:
        content = json.loads(f.read())
        for id in content:
            test_item.append({
                'id': id,
                'title': content[id]['title'],
                'text': ''.join(preProcess(content[id]['text'])),
                'label_1': content[id]['label_1'],
                'label_2': resetLabelLv2(content[id]['label_2']),
                'label_3': resetLabelLv3(content[id]['label_3'])
            })
    if item['label_2'] == "undefine":
        item['label_2'] = "无对应二级办理部门"  
    if item['label_3'] == "undefine":
        item['label_3'] = "无对应三级办理部门"  

    with open(os.path.join(config.path.data, 'test.txt'), 'w') as f:
        for item in test_item:
            if item['text'] == '':
                continue
            f.write(item['text'] + '\t' +\
                    item['label_1'] + ',' + \
                    item['label_1'] + '##' + item['label_2'] + ',' + \
                    item['label_1'] + '##' + item['label_2'] + '##' + item['label_3'] + '\n'
            )
    with open(os.path.join(config.path.data, 'data.txt'), 'w') as f:
        for item in test_item:
            if item['text'] == '':
                continue
            f.write(item['text'] + '\n')

class LabelTreeNode:
    def __init__(self, label) -> None:
        self.label = label
        self.children = {}

    def add_child(self, label):
        if(label in list(self.children.keys())):
            return
        else:
            self.children[label] = LabelTreeNode(label)

    def get_child_by_label(self, label):
        if label in list(self.children.keys()):
            return self.children[label]
        else:
            return None
        
def dfs_write_label(cur_node: LabelTreeNode, cur_label, filestream):
    cur_label += cur_node.label
    filestream.write(cur_label + '\n')
    cur_label += '##'
    for label in cur_node.children.keys():
        dfs_write_label(cur_node.children[label], cur_label, filestream)

def dfs_get_label(target, cur_node: LabelTreeNode, cur_label, label_list) -> bool:
    cur_label += cur_node.label
    if cur_node.label == target:
        label_list.append(cur_label)
        return True
    
    next_label = cur_label
    next_label += '##'
    
    for label in cur_node.children.keys():
        if dfs_get_label(target, cur_node.children[label], next_label, label_list):
            label_list.append(cur_label)
            return True
    
    return False



if __name__ == '__main__':
    readTrainData(True)
    # readTestSet()
