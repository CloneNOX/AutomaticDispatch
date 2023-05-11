# Flask框架API main.py代码，请确保本文文件和其他依赖文件放在同一个目录下
# coding:utf-8
from flask import Flask, render_template, request,jsonify
from flask_restful import Resource, Api
import json
import pandas as pd
import numpy as np
import time
import utils
import os
import signal
from datetime import datetime
from model import myPaddleHierarchical

# 放在header中的apptoken的值,用于验证客户端身份
APP_TOKEN = 'DAT3X71FH87_2sB'
# 声明APP
app = Flask(__name__)
app.config.update(RESTFUL_JSON=dict(ensure_ascii=False))
API = Api(app)
# 装载配置文件
config = {}
with open('./config.json', 'r') as f:
    config = json.loads(f.read())

# 无参数路由，打开初始页面
@app.route('/')
def start():
    return render_template('index.html')

# 文本分类模型
model = myPaddleHierarchical()
model.load_model('./model/checkpoint/')

class Dispatch(Resource):
    def post(self):
        # 客户端身份验证
        if 'Apptoken' not in list(request.headers.keys()):
            print(list(request.headers.keys()))
            return {
                'success': 'False',
                'msg': '没有收到token'
            }
        apptoken = request.headers['Apptoken']
        if apptoken != APP_TOKEN:
            return {
                'success': 'False',
                'msg': '无效的 apptoken'
            }
        

        # 读取文件
        tid = request.form['tid']
        f = request.files['file']
        # 原始工单内容
        f_content = f.read().decode('utf-8')

        # 提取工单内容，分词（按字符分词，字符之间用空格隔开）
        content = ''.join(utils.fixText(f_content))

        # 分类预测
        result = model.predict(content)
        department_1 = result[0][0]
        department_2 = result[1][0]
        department_3 = result[2][0]
        probs_1 = result[0][1]
        probs_2 = result[1][1]
        probs_3 = result[2][1]

        with open('./ticket_info.txt', 'a') as file:
            file.write('[' + datetime.now().strftime(r'%Y-%m-%d %H:%M:%S') + ']\t')
            file.write(tid + '\t')
            file.write(content + '\t')
            file.write(
                'department 1:' + department_1 +\
                '\tdepartment 2: ' + department_2 +\
                '\tdepartment 3: ' + department_3 + '\n'            
            )

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
    # 如果之前有一个进程在运行，就杀掉该进程
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
