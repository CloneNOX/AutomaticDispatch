# Flask框架API main.py代码，请确保本文文件和其他依赖文件放在同一个目录下
# coding:utf-8
from flask import Flask, render_template, request,jsonify
from flask_restful import Resource, Api
import json
import numpy as np
import time
import utils
import os
import signal
from datetime import datetime
from onnxModel import OnnxHierarchical

# 放在header中的apptoken的值,用于验证客户端身份
APP_TOKEN = 'DAT3X71FH87_2sB'
# 声明APP
app = Flask(__name__)
app.config.update(RESTFUL_JSON=dict(ensure_ascii=False))
API = Api(app)

# 无参数路由，打开Index页面
@app.route('/')
def start():
    return render_template('index.html')

# 文本分类模型
model = OnnxHierarchical()

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
        result = model(content) # result: [(label_1, prob_1), (label_2, prob_2), (label_3, prob_3)]
        error_info = []
        try: 
            department_1 = result[0][0]
            probs_1 = float(result[0][1])
            if 'undefine' in department_1:
                department_1 = '无对应一级办理部门'
        except Exception as e:
            error_info.append('level 1 raise error: ' + str(e))
            department_1 = '无对应一级办理部门'
            probs_1 = 0.99

        try:
            department_2 = result[1][0]
            department_2 = department_2.split('##')[1]
            if 'undefine' in department_2:
                department_2 = '无对应二级办理部门'
            probs_2 = float(result[1][1])
        except Exception as e:
            error_info.append('level 2 raise error: ' + str(e))
            department_2 = '无对应二级办理部门'
            probs_2 = 0.99

        try:
            department_3 = result[2][0]
            department_3 = department_3.split('##')[2]
            if 'undefine' in department_3:
                department_3 = '无对应三级办理部门'
            probs_3 = float(result[2][1])
        except Exception as e:
            error_info.append('level 3 raise error: ' + str(e))
            department_3 = '无对应三级办理部门'
            probs_3 = 0.99

        with open('./ticket_info.txt', 'a') as file:
            file.write('[' + datetime.now().strftime(r'%Y-%m-%d %H:%M:%S') + ']\t')
            file.write(tid + '\t')
            file.write(content + '\t')
            file.write(
                'department 1: ' + department_1 + '\t' + \
                'probs_1: ' + str(probs_1) + '\t' + \
                'department 2: ' + department_2 + '\t' + \
                'probs_2: ' + str(probs_2) + '\t' + \
                'department 3: ' + department_3 + '\t' + \
                'probs_3: ' + str(probs_3) + '\t' + \
                'result: ' + str(result) + '\t' + \
                'error info: ' + str(error_info) + '\n'
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
    # 初始化记录文档
    with open('./ticket_info.txt', 'w') as file:
        pass
    with open('./app_run_log.txt', 'w') as file:
        pass
    app.run(host='0.0.0.0')
