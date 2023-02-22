# Flask框架API main.py代码，请确保本文文件和其他依赖文件放在同一个目录下
# coding:utf-8
from flask import Flask, render_template, request,jsonify
from flask_restful import Resource, Api
import json
import pandas as pd
import numpy as np
import time
import fasttext
import utils
import os
import signal

# 放在header中的apptoken的值,用于验证客户端身份
APP_TOKEN = 'DAT3X71FH87_2sB'
# 声明APP
app = Flask(__name__)
API = Api(app)

# 无参数路由，打开初始页面
@app.route('/')
def start():
    return render_template('index.html')

# 文本分类模型
model_1 = fasttext.load_model('./models/no2_lr_10e1_epoch100_tag1')
model_2 = fasttext.load_model('./models/no2_lr_10e1_epoch100_tag2')
model_3 = fasttext.load_model('./models/no2_lr_10e1_epoch100_tag3')

class Dispatch(Resource):
    def post(self):
        # 客户端身份验证
        apptoken = request.headers['apptoken']
        if apptoken != APP_TOKEN:
            return {
                'success': False,
                'msg': '无效的 apptoken'
            }

        # 读取文件
        tid = request.form['tid']
        f = request.files['file']
        # 原始工单内容
        f_content = f.read().decode('utf-8')

        # 提取工单内容，分词（按字符分词，字符之间用空格隔开）
        content = utils.fixText(f_content)
        
        # 分类预测
        result_1 = model_1.predict(content)
        result_2 = model_2.predict(content)
        result_3 = model_3.predict(content)
        department_1 = result_1[0][0][9:]
        department_2 = result_2[0][0][9:]
        department_3 = result_3[0][0][9:]
        probs_1 = result_1[1][0]
        probs_2 = result_2[1][0]
        probs_3 = result_3[1][0]

        # 返回派单结果
        return {
            'tid': tid,
            'success': True,
            'label_1': department_1,
            'probs_1': probs_1,
            'label_2': department_2,
            'probs_2': probs_2,
            'label_3': department_3,
            'probs_3': probs_3,
        }, 200, {'Content-Type': 'application/json; charset=utf-8'}


# 绑定接口地址
API.add_resource(Dispatch, '/dispatch')


def kill_pid_if_exists():
    """如果之前有一个进程在运行，就杀掉该进程
    """
    try:
        pid = open('run.pid', 'r').read()
        os.kill(int(pid), signal.SIGTERM)
        print('KILLING PREVIOUS PROCESS')
        time.sleep(1)
    except:
        pass

    # 将当前PID写入文件
    with open('run.pid', 'w') as f:
        f.write(str(os.getpid()))
        f.flush()

if __name__ == '__main__':
    kill_pid_if_exists()
    # 必须加上 host='0.0.0.0' 否则只能通过 127.0.0.1 访问接口
    app.run(host='0.0.0.0')
