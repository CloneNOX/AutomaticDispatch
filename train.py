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

if __name__ == '__main__':
    # 读入配置文件
    config = getConfig()
    # 生成训练数据
    readTrainData(need_test_set=True)

    # 读入训练数据集

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