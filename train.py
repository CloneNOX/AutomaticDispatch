import os
import json
import re
from utils import fixText, resetLabelLv2, resetLabelLv3, split_text
from data import readTrainData
from model import MyFastText

TEMP_DIR = './tmp/'

def getConfig():
    config = {}
    with open('./config.json', 'r') as f:
        s = f.read()
        config = json.loads(s)
    return config

# 读入数据集的json文件，处理成fasttext接口使用的"文本__label__标签"形式，以txt文件存储
def readDataSet(config):
    set1 = []
    with open(config['data_path'] + config['dataset_label_1_name'], 'r') as f:
        s = f.read()
        data_set = json.loads(s)
        for id in list(data_set.keys()):
            text = data_set[id]['text']
            splited_text = split_text(text)
            for text in splited_text:
                set1.append('__label__' + data_set[id]['label_level_1'] + ' ' + fixText(text))
    
    set2 = []
    with open(config['data_path'] + config['dataset_label_2_name'], 'r') as f:
        s = f.read()
        data_set = json.loads(s)
        for id in list(data_set.keys()):
            text = data_set[id]['text']
            splited_text = split_text(text)
            for text in splited_text:
                set2.append('__label__' + resetLabelLv2(data_set[id]['label_level_2']) + ' ' + fixText(text))
    
    set3 = []
    with open(config['data_path'] + config['dataset_label_3_name'], 'r') as f:
        s = f.read()
        data_set = json.loads(s)
        for id in list(data_set.keys()):
            text = data_set[id]['text']
            splited_text = split_text(text)
            for text in splited_text:
                set3.append('__label__' + resetLabelLv3(data_set[id]['label_level_3']) + ' ' + fixText(text))
        
    try:
        os.mkdir(TEMP_DIR)
    except:
        pass
    with open(TEMP_DIR + 'set1.txt', 'w') as f:
        print('训练集1拆分后共{}条文本'.format(len(set1)))
        for l in set1:
            f.write(l + '\n')
    with open(TEMP_DIR + 'set2.txt', 'w') as f:
        print('训练集2拆分后共{}条文本'.format(len(set2)))
        for l in set2:
            f.write(l + '\n')
    with open(TEMP_DIR + 'set3.txt', 'w') as f:
        print('训练集3拆分后共{}条文本'.format(len(set2)))
        for l in set3:
            f.write(l + '\n')

if __name__ == '__main__':
    # 读入配置文件
    config = getConfig()
    # 生成训练数据
    readTrainData(need_test_set=True)

    # 读入训练数据集
    readDataSet(config)

    # 训练标签1的模型
    model_label_1 = MyFastText()
    model_label_1.train(
        input = TEMP_DIR + 'set1.txt',
        lr = config['lr'],
        dim = config['hidden_dim'],
        epoch = config['epoch'],
    )
    model_label_1.save(config['model_path'] + config['model_label_1_name'])
    
    # 训练标签2的模型
    model_label_2 = MyFastText()
    model_label_2.train(
        input = TEMP_DIR + 'set2.txt',
        lr = config['lr'],
        dim = config['hidden_dim'],
        epoch = config['epoch']
    )
    model_label_2.save(config['model_path'] + config['model_label_2_name'])
    
    # 训练标签3的模型
    model_label_3 = MyFastText()
    model_label_3.train(
        input = TEMP_DIR + 'set3.txt',
        lr = config['lr'],
        dim = config['hidden_dim'],
        epoch = config['epoch']
    )
    model_label_3.save(config['model_path'] + config['model_label_3_name'])

    # 保存模型
    model_label_1.save(config['model_path'] + config['model_label_1_name'])
    model_label_2.save(config['model_path'] + config['model_label_2_name'])
    model_label_3.save(config['model_path'] + config['model_label_3_name'])

    try:
        os.remove(TEMP_DIR + 'set1.txt')
        os.remove(TEMP_DIR + 'set2.txt')
        os.remove(TEMP_DIR + 'set3.txt')
        os.removedirs(TEMP_DIR)  
    except:
        pass