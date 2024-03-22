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
# text = "改革创新强主业 凝心聚力促发展 - 光明网2021年9月24日  浙报传媒控股集团有限公司 2020年，浙报传媒控股集团有限公司不断提升媒体融合传播能力,优化完善“一核多平台多集群”媒体布局，主流媒体传播阵地进一步拓展，新媒体用户规模达1.34"
# res = test(text, model, args)
@app.route('/')
@app.route('/home')
def home():
    return render_template('index.html')


@app.route('/keyExt', methods=["GET"])
def keyword_extraction():
    return render_template('keyExt.html')

@app.route("/query", methods=["POST"])
def query():
    res = {}
    text = request.values['text']
 
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
    recognized_entity = test(text, model, args)#【【需要调用已有模型，返回的可能是字符串 or pickle文件的列表】】
    res = jsonify(recognized_entity)
    # print("result:", recognized_entity)
    return res


app.config['UPLOAD_FOLDER'] = 'upload/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1000 * 1000  # 上传文件大小限制为16M，如果超过会抛出异常

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


@app.route('/preproc', methods=["GET"])
def pre_process():
    return render_template('preproc.html')


@app.route('/others', methods=["GET"])
def others():
    return render_template('others.html')


@app.route('/summary', methods=["GET"])
def summary():
    return render_template('text_Summarization.html')


@app.route('/installation', methods=["GET"])
def installation():
    return render_template('installation.html')


@app.route('/lower', methods=["GET", "POST"])
def lower_case():
    text1 = request.form['text']
    word = text1.lower()
    result = {
        "result": word
    }
    result = {str(key): value for key, value in result.items()}
    return jsonify(result=result)

@app.route('/sent_tokenize', methods=["GET", "POST"])
def sent_tokenize():
    text = request.form['text']
    sent_tokenize = nltk.sent_tokenize(text)
    result = {
        "result": str(sent_tokenize) #remove str() if you want the output as list
    }
    result = {str(key): value for key, value in result.items()}
    return jsonify(result=result)

@app.route('/word_tokenize', methods=["GET", "POST"])
def word_tokenize():
    text = request.form['text']
    word_tokenize = nltk.word_tokenize(text)
    result = {
        "result": str(word_tokenize) #remove str() if you want the output as list
    }
    result = {str(key): value for key, value in result.items()}
    return jsonify(result=result)

@app.route('/spell_check', methods=["GET", "POST"])
def spell_check():
    text = request.form['text']
    spells = [spell(w) for w in (nltk.word_tokenize(text))]
    result = {
        "result": " ".join(spells)
    }
    result = {str(key): value for key, value in result.items()}
    return jsonify(result=result)

@app.route('/lemmatize', methods=["GET", "POST"])
def lemmatize():
    from nltk.stem import WordNetLemmatizer
    wordnet_lemmatizer = WordNetLemmatizer()

    text = request.form['text']
    word_tokens = nltk.word_tokenize(text)
    lemmatized_word = [wordnet_lemmatizer.lemmatize(word) for word in
                       word_tokens]
    result = {
        "result": " ".join(lemmatized_word)
    }
    result = {str(key): value for key, value in result.items()}
    return jsonify(result=result)

@app.route('/stemming', methods=["GET", "POST"])
def stemming():
    from nltk.stem import SnowballStemmer
    snowball_stemmer = SnowballStemmer('english')

    text = request.form['text']
    word_tokens = nltk.word_tokenize(text)
    stemmed_word = [snowball_stemmer.stem(word) for word in word_tokens]
    result = {
        "result": " ".join(stemmed_word)
    }
    result = {str(key): value for key, value in result.items()}
    return jsonify(result=result)

@app.route('/remove_tags', methods=["GET", "POST"])
def remove_tags():
    import re
    text = request.form['text']
    cleaned_text = re.sub('<[^<]+?>', '', text)
    result = {
        "result": cleaned_text
    }
    result = {str(key): value for key, value in result.items()}
    return jsonify(result=result)


@app.route('/remove_numbers', methods=["GET", "POST"])
def remove_numbers():
    text = request.form['text']
    remove_num = ''.join(c for c in text if not c.isdigit())
    result = {
        "result": remove_num
    }
    result = {str(key): value for key, value in result.items()}
    return jsonify(result=result)

@app.route('/remove_punct', methods=["GET", "POST"])
def remove_punct():
    from string import punctuation
    def strip_punctuation(s):
        return ''.join(c for c in s if c not in punctuation)

    text = request.form['text']
    text = strip_punctuation(text)
    result = {
        "result": text
    }
    result = {str(key): value for key, value in result.items()}
    return jsonify(result=result)

@app.route('/remove_stopwords', methods=["GET", "POST"])
def remove_stopwords():
    from nltk.corpus import stopwords
    stopword = stopwords.words('english')
    text = request.form['text']
    word_tokens = nltk.word_tokenize(text)
    removing_stopwords = [word for word in word_tokens if word not in stopword]
    result = {
        "result": " ".join(removing_stopwords)
    }
    result = {str(key): value for key, value in result.items()}
    return jsonify(result=result)


@app.route("/keyword", methods=["GET","POST"])
def keyword():
    text = request.form['text']
    word = nltk.word_tokenize(text)
    pos_tag = nltk.pos_tag(word)
    chunk = nltk.ne_chunk(pos_tag)
    NE = [" ".join(w for w, t in ele) for ele in chunk if isinstance(ele, nltk.Tree)]
    result = {
        "result": NE
    }
    result = {str(key): value for key, value in result.items()}
    return jsonify(result=result)


@app.route("/summarize", methods=["GET","POST"])
def summarize():
    text = request.form['text']
    sent = nltk.sent_tokenize(text)
    if len(sent) < 2:
        summary1 =  "please pass more than 3 sentences to summarize the text"
    else:
        #summary = gensim.summarization.summarize(text)
        summary = text
        summ = nltk.sent_tokenize(summary)
        summary1 = (" ".join(summ[:2]))
    result = {
        "result": summary1
    }
    result = {str(key): value for key, value in result.items()}
    return jsonify(result=result)

if __name__ == '__main__':
    sys.path.append(r"D:\BERT_Chinese\Finish_API\Flink")
    app.run(debug=True)

