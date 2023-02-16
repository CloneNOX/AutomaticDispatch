import fasttext
import os
import json
from utils import fixText

TEMP_DIR = './tmp/'

def getConfig():
    config = {}
    with open('./config.json', 'r') as f:
        s = f.read()
        config = json.loads(s)
    return config

# 读入数据集的json文件，处理成fasttext接口使用的"文本__label__标签"形式，以txt文件存储
def readDataSet(path):
    with open(path, 'r') as f:
        s = f.read()
        data_set = json.loads(s)
    set1 = []
    set2 = []
    set3 = []
    for id in list(data_set.keys()):
        set1.append('__label__' + data_set[id]['tag_level_1'] + ' ' + fixText(data_set[id]['text']))
        set2.append('__label__' + data_set[id]['tag_level_2'] + ' ' + fixText(data_set[id]['text']))
        set3.append('__label__' + data_set[id]['tag_level_3'] + ' ' + fixText(data_set[id]['text']))
    try:
        os.mkdir(TEMP_DIR)
    except:
        pass
    with open('./tmp/set1.txt', 'w') as f:
        for l in set1:
            f.write(l + '\n')
    with open('./tmp/set2.txt', 'w') as f:
        for l in set2:
            f.write(l + '\n')
    with open('./tmp/set3.txt', 'w') as f:
        for l in set3:
            f.write(l + '\n')

if __name__ == '__main__':
    config = getConfig()
    readDataSet(config['data_path'] + config['data_set_path'])
    model_tag1 = fasttext.train_supervised(
        input = TEMP_DIR + 'set1.txt',
        lr = config['lr'],
        dim = config['hidden_dim'],
        epoch = config['epoch']
    )
    model_tag2 = fasttext.train_supervised(
        input = TEMP_DIR + 'set2.txt',
        lr = config['lr'],
        dim = config['hidden_dim'],
        epoch = config['epoch']
    )
    model_tag3 = fasttext.train_supervised(
        input = TEMP_DIR + 'set3.txt',
        lr = config['lr'],
        dim = config['hidden_dim'],
        epoch = config['epoch']
    )

    try:
        os.remove(TEMP_DIR + 'set1.txt')
        os.remove(TEMP_DIR + 'set2.txt')
        os.remove(TEMP_DIR + 'set3.txt')
        os.removedirs(TEMP_DIR)  
    except:
        pass