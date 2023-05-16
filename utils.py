import re
import jieba
jieba.lcut('测试')

# 整理输入的文本，通过只保留中文和数字部分，通过jieba库分词，用于生成喂给模型的数据
def fixText(text):
    text = re.sub(r'[^\u4e00-\u9fa50-9a-zA-z]+', '', text)
    text = ' '.join(jieba.lcut(text))
    return text

# 重设办理部门2级，减少类别数量
# 当前做法，2级标签保留街道信息，去除街道后面的实体
import re
def resetLabelLv2(label):
    if(re.search(r'.+街(道)*', label)):
        label = re.search(r'.+街(道)*', label).group() # 去除街道后面的内容
        if "街道" not in label:
            label = label.replace("街", "街道") # 统一为“街道”
    if(re.search(r'.+[区]', label)):
        label = re.sub(re.search(r'.+[区]', label).group(), '', label) # 去除市、区
    return label

# 重设办理部门3级，减少类别数量
# 当前做法，3级标签去除街道信息，保留街道后面的实体，去除"()"中的描述内容
def resetLabelLv3(label):
    if(re.search(r'[\u4e00-\u9fa50-9]+街(道)*', label)):
        label = re.search(r'街(道)*.*', label).group() # 保留街(道)后面的内容
        if "街道" not in label: 
            label = label.replace("街", "街道") # 统一为“街道”
        label = re.sub(r'[(|（][\u4e00-\u9fa50-9]*[)|）]', '', label) # 去括号及括号里内容
    return label

# 根据换行符切分句子
# 输入：一段文本
# 输出：切分后的
def split_text(text):
    splited_text = re.split(r'\n+|\r+|\r\n|<br>', text)
    while '' in splited_text:
        splited_text.remove('')
    return splited_text
