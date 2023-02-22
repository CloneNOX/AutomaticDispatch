import re
import jieba

# 整理输入的文本，通过只保留中文部分，通过jieba库分词
def fixText(text):
    text = re.sub(r'[^\u4e00-\u9fa5]+', '', text)
    text = ' '.join(jieba.lcut(text))
    return text

# 重设办理部门2级，减少类别数量
def resetLabelLv2(label):
    return label

# 重设办理部门3级，减少类别数量
def resetLabelLv3(label):
    return label