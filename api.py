from flask import Flask, request, render_template,jsonify
import nltk
from autocorrect import spell
import json
from flask import send_from_directory
from werkzeug.utils import secure_filename
import os
from predict import get_entity_lst
import sys
from models import get_entity_rec_model



app = Flask(__name__)




# 配置信息
app.config['UPLOAD_FOLDER'] = 'upload/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1000 * 1000  # 上传文件大小限制为16M，如果超过会抛出异常


@app.route('/')
@app.route('/home')
def home():
    return render_template('index.html')


@app.route('/entityRec', methods=["GET"])
def entity_recognize():
    """
    实体识别页面
    :return:
    """
    return render_template('entityRec.html')


@app.route("/query", methods=["POST"])
def query():
    """
    查询实体接口
    :return: [ TODO
      "技研株式会社",
      "大地熊"
    ]
    """
    text = request.values['text']
    disambiMethod = request.values['disambiMethod']
 
    if not text:
        return jsonify({"result": "error"})

    recognized_entity = get_entity_lst(text, disambiMethod)
    return jsonify(recognized_entity)


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    """
    上传文件接口
    :return:
    """
    if request.method == 'POST':
        f = request.files['file']
        print(request.files)
        # secure_filename检查客户端上传的文件名，确保安全，注意文件名称不要全中文！！！
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename)))
        # return render_template('upload.html')# 上传成功界面
        return "<script>alert('上传成功'); window.location = '/keyExt';</script>"
    # else:
        # return render_template('index.html')# 未上传的界面


@app.route('/upload/<filename>', methods=['GET', 'POST'])
def download(filename):
    """
    下载文件接口
    :param filename: 文件名（文件存在/upload/）
    :return:
    """
    # as_attachment=True 表示文件作为附件下载
    return send_from_directory('./upload', filename, as_attachment=True)


@app.route('/about', methods=["GET"])
def about():
    """
    关于我们页面
    :return:
    """
    return render_template('about.html')


@app.route('/version', methods=["GET"])
def version():
    """
    版本信息页面
    :return:
    """
    return render_template('version.html')


if __name__ == '__main__':
    # sys.path.append(r"D:\BERT_Chinese\Finish_API\Flink")
    app.run(debug=True, port=5000)

