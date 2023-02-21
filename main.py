# Flask框架API main.py代码，请确保本文文件和其他依赖文件放在同一个目录下

from flask import Flask, render_template, request
import json
import pandas as pd
import numpy as np
import time
import torch
import fasttext

# 声明APP
app = Flask(__name__)

# 无参数路由，打开初始页面
@app.route('/')
def start():
    return render_template('index.html')


# 单条文本处理的API接口
@app.route('/api/<colName>')
def api(colName):
    element_out, element_out2, start_id, end_id, kind, is_in, sents_rec, main_word = dataele.Output(colName)

    result = {
        'origional_text': colName,
        'standardized_text': sents_rec,
        'sjy_out': element_out,
        'sjy_out2': element_out2,
        'start_id': start_id,
        'end_id': end_id,
        'kind': kind,
        'is_in':is_in,
        'sents': sents_rec,
        'main_word': main_word
    }
    return json.dumps(result, ensure_ascii=False)

if __name__ == '__main__':
    app.run(debug = True)