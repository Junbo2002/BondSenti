from flask import Flask, request, render_template,jsonify
import nltk
from autocorrect import spell
import json
from flask import send_from_directory
from werkzeug.utils import secure_filename
import os
from message import test
import sys
from models import get_model
from utils import get_args


app = Flask(__name__)
args = get_args()
model = get_model(args)


# 配置信息
app.config['UPLOAD_FOLDER'] = 'upload/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1000 * 1000  # 上传文件大小限制为16M，如果超过会抛出异常


@app.route('/')
@app.route('/home')
def home():
    return render_template('index.html')


@app.route('/entityRec', methods=["GET"])
def keyword_extraction():
    return render_template('entityRec.html')

@app.route("/query", methods=["POST"])
def query():
    res = {}
    text = request.values['text']
    disambiMethod = request.values['disambiMethod']
 
    if not text:
        res["result"] = "error"
        return jsonify(res)
    
    # sentences_map_list = [{"text":text} ]# 本项目不需要分割句子，直接产生如[{"text":全文本}]

    # path_test = '../test.json'
    
    # with open(path_test, 'a+', encoding='utf-8') as f:
    #     json.dump({"text":text}, f, ensure_ascii=False)

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(BASE_DIR)
        # TODO：【【【已有模型和web的API】】】实体抽取关系任务的调用test()!!!
    recognized_entity = test(text, model, args, disambiMethod)#【【需要调用已有模型，返回的可能是字符串 or pickle文件的列表】】
    res = jsonify(recognized_entity)
    # print("result:", recognized_entity)
    return res


@app.route('/upload', methods=['GET', 'POST'])
def upload():
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
    # as_attachment=True 表示文件作为附件下载
    return send_from_directory('./upload', filename, as_attachment=True)


@app.route('/result',methods = ['POST', 'GET'])
def result():
   if request.method == 'POST':
      result = request.form #拿到前端传输的表单数据
      return render_template("result.html",result = result)


@app.route('/about', methods=["GET"])
def about():
    return render_template('about.html')


@app.route('/version', methods=["GET"])
def version():
    return render_template('version.html')



@app.route('/sent_tokenize', methods=["GET", "POST"])
def sent_tokenize():
    text = request.form['text']
    sent_tokenize = nltk.sent_tokenize(text)
    result = {
        "result": str(sent_tokenize) #remove str() if you want the output as list
    }
    result = {str(key): value for key, value in result.items()}
    return jsonify(result=result)


if __name__ == '__main__':
    sys.path.append(r"D:\BERT_Chinese\Finish_API\Flink")
    app.run(debug=True, port=5000)

