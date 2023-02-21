import os
import pandas as pd
import json
import numpy as np
import random
data_path = './data/'

# 数据预处理
def preProcess(text: str):
    
    return text

if __name__ == '__main__':
    from data import preProcess
    total_item = {}
    for filename in os.listdir(data_path):
        if(filename.lower().endswith('.xls') or filename.lower().endswith('xlsx')):
            df = pd.read_excel(data_path + filename)
            df.columns = [s.strip() for s in df.columns.tolist()]
            for row in df.index.values:
                id = str(df.loc[row, '工单编号'])
                title = df.loc[row, '诉求标题']
                if(not pd.isnull(df.loc[row, '市民原始诉求'])):
                    text = str(df.loc[row, '市民原始诉求'])
                else:
                    text = ('' if pd.isnull(df.loc[row, '涉事主体']) else '涉事主体:' + str(df.loc[row, '涉事主体'])) + \
                        ('' if pd.isnull(df.loc[row, '主体地址']) else '主体地址:' + str(df.loc[row, '主体地址'])) + \
                        ('' if pd.isnull(df.loc[row, '事发地点']) else '事发地点:' + str(df.loc[row, '事发地点'])) + \
                        ('' if pd.isnull(df.loc[row, '标签组']) else '标签组:' + str(df.loc[row, '标签组'])) + \
                        ('' if pd.isnull(df.loc[row, '市民诉求']) else '市民诉求:' + str(df.loc[row, '市民诉求'])) + \
                        ('' if pd.isnull(df.loc[row, '补充信息']) else '补充信息:' + str(df.loc[row, '补充信息']))
                text = preProcess(text)
                tag_level_1 = 'undefine' if pd.isnull(df.loc[row, '办理部门一级']) else str(df.loc[row, '办理部门一级'])
                tag_level_2 = 'undefine' if pd.isnull(df.loc[row, '办理部门二级']) else str(df.loc[row, '办理部门二级'])
                tag_level_3 = 'undefine' if pd.isnull(df.loc[row, '办理部门三级']) else str(df.loc[row, '办理部门三级'])
                total_item[str(id)] = {
                    'title': title,
                    'text': text,
                    'tag_level_1': tag_level_1,
                    'tag_level_2': tag_level_2,
                    'tag_level_3': tag_level_3
                }
    ids = list(total_item.keys())
    random.shuffle(ids)
    train_item = {}
    for id in ids[:len(ids) * 4 // 5]:
        train_item[id] = total_item[id]
    test_item = {}
    for id in ids[len(ids) * 4 // 5:]:
        test_item[id] = total_item[id]
    del total_item

    with open(data_path + 'train_set.json', 'w') as f:
        s = json.dumps(train_item)
        f.write(s)
    with open(data_path + 'test_set.json', 'w') as f:
        s = json.dumps(test_item)
        f.write(s)