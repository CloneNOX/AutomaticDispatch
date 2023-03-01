import re
import jieba

# 整理输入的文本，通过只保留中文和数字部分，通过jieba库分词，用于生成喂给模型的数据
def fixText(text):
    text = re.sub(r'[^\u4e00-\u9fa50-9]+', '', text)
    text = ' '.join(jieba.lcut(text))
    return text

# 重设办理部门2级，减少类别数量
# 当前做法，2级标签保留街道信息，去除街道后面的实体
def resetLabelLv2(label):
    if(re.search(r'[\u4e00-\u9fa50-9]+街(道)*', label)):
        label = re.search(r'[\u4e00-\u9fa50-9]+街(道)*', label).group()
        if(re.search(r'[\u4e00-\u9fa50-9]+区', label)):
            label = re.sub(re.search(r'[\u4e00-\u9fa50-9]+区', label).group(), '', label)
    return label

# 重设办理部门3级，减少类别数量
# 当前做法，3级标签去除街道信息，保留街道后面的实体
def resetLabelLv3(label):
    if(re.search(r'[\u4e00-\u9fa50-9]+街(道)*', label)):
        label = re.search(r'街(道)*[\u4e00-\u9fa50-9]*', label).group()
    return label