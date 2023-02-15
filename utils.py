import re
import jieba

# 整理
def fixText(text):
    text = re.sub('[^\u4e00-\u9fa5]+', '', text)
    text = ' '.join(jieba.lcut(text))
    return text