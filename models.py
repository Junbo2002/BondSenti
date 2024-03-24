import torch
import torch.nn as nn
import torch.functional as F
from torchcrf import CRF
from pytorch_transformers import BertPreTrainedModel, BertModel, BertConfig



""" 
创建一个自定义的模型类，继承自BertPreTrainedModel，这个类是Hugging Face库中的BERT模型的扩展类。
"""
class BERT_BiLSTM_CRF(BertPreTrainedModel):
    def __init__(self, config, need_birnn=False, rnn_dim=128):
        super(BERT_BiLSTM_CRF, self).__init__(config)
        self.num_tags = config.num_labels
        self.bert = BertModel(config)  # 初始化BERT模型
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        out_dim = config.hidden_size  # 输出维度初始化为BERT隐层大小
        self.need_birnn = need_birnn  # 如果need_birnn为True，则添加一个BiLSTM层
        # 如果为False，则不要BiLSTM层
        if need_birnn:
            self.birnn = nn.LSTM(config.hidden_size, rnn_dim, num_layers=1, bidirectional=True, batch_first=True)
            out_dim = rnn_dim*2 # 更新输出维度为BiLSTM的输出大小（双向）
        self.hidden2tag = nn.Linear(out_dim, config.num_labels)
        self.crf = CRF(config.num_labels, batch_first=True)
    
# 计算CRF层的损失
    def forward(self, input_ids, tags, token_type_ids=None, input_mask=None):
        return -1*self.crf(self.tag_outputs(input_ids, token_type_ids, input_mask), \
            tags, mask=input_mask.byte())


# 将处理后的输出映射到标签空间
    def tag_outputs(self, input_ids, token_type_ids=None, input_mask=None):
        outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=input_mask)
        sequence_output = outputs[0]
        if self.need_birnn:
            sequence_output, _ = self.birnn(sequence_output)
        sequence_output = self.dropout(sequence_output)
        return self.hidden2tag(sequence_output)
    
# CRF输出的是int类型，之后再转换成label。
#【需检测在此处CRF是否就吃掉了部分字符】
    def predict(self, input_ids, token_type_ids=None, input_mask=None):
        # 预测函数，返回通过CRF解码后的预测结果
        emissions = self.tag_outputs(input_ids, token_type_ids, input_mask)
        temp = self.crf.decode(emissions, input_mask.byte())
        return temp


def get_entity_rec_model(args):
    # num_labels = len(args.label_list)
    # config = BertConfig.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
    #                                     num_labels=num_labels)
    # print(args.config_name, args.model_name_or_path, num_labels)
    # model = BERT_BiLSTM_CRF.from_pretrained(args.output_dir, config=config,
    #                                         need_birnn=args.need_birnn, rnn_dim=args.rnn_dim)
    # print(args.output_dir, args.need_birnn, args.rnn_dim)
    args.need_birnn = True
    args.rnn_dim = 256
    model = BERT_BiLSTM_CRF.from_pretrained(args.output_dir, need_birnn=args.need_birnn, rnn_dim=args.rnn_dim)
    return model


class MLP(torch.nn.Module):
    def __init__(self, output_n=5, layer_list=[768, 256], dropout=0.3):
        """
        :param output_n: int 输出神经元个数
        :param layer_list: list(int) 每层隐藏层神经元个数
        :param dropout: float 训练完丢掉多少
        """
        super(MLP, self).__init__()
        self.output_n = output_n
        self.num_layer = len(layer_list)
        self.layer_list = layer_list

        # 隐藏层
        self.hidden_layer = nn.ModuleList(
            [nn.Sequential(nn.Linear(layer_list[i], layer_list[i+1], bias=True),
                          torch.nn.BatchNorm1d(layer_list[i+1]),
                          nn.ReLU(),
                          nn.Dropout(dropout)
            ) for i in range(self.num_layer-1)]
        )

        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(layer_list[-1], output_n, bias=True),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        for layer in self.hidden_layer:
            x = layer(x)
        x = self.output_layer(x)
        return x


def get_sentiment_model(args):
    model = MLP()
    params = torch.load(args.sentiment_model_path, map_location=torch.device('cpu'))
    model.load_state_dict(params)
    return model
