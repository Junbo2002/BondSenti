import logging
import os
import sys
import torch
import pickle
import argparse
from torch.utils.data import TensorDataset
from tqdm import tqdm


"""部分数据预处理的过程，包括读取数据、转换为特征、构建数据集等。
1. NerProcessor 类，通过 read_data 方法读取BIO标注的数据文件，
2.通过 get_labels 方法获取标签集合，并通过 get_examples 方法获取示例。
3.convert_examples_to_features 方法将示例转换为模型输入特征。
4.get_Dataset 方法用于获取数据集，根据 mode 参数决定使用哪个数据文件（train、eval、test）
"""

logger = logging.getLogger(__name__)

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text, label=None):
        self.guid = guid
        self.text = text 
        self.label = label

# 定义一个表示数据特征的类
class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, input_ids, input_mask, segment_ids, label_id, ori_tokens):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.ori_tokens = ori_tokens

# NERProcessor类负责处理数据，读取和转换示例
class NerProcessor(object):
    def read_from_file(self, file_path):
        # 读取BIO标注的数据文件
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        return lines

    def read_data(self, data):
        lines = []
        words = []
        labels = []
        for line in data:
            contends = line.strip()
            tokens = line.strip().split(" ")
            # print("tokens: ", tokens)
            if len(tokens) == 2:
                words.append(tokens[0])
                labels.append(tokens[1])
            else:
                if len(contends) == 0 and len(words) > 0:
                    label = []
                    word = []
                    for l, w in zip(labels, words):
                        if len(l) > 0 and len(w) > 0:
                            label.append(l)
                            word.append(w)
                    lines.append([' '.join(label), ' '.join(word)])
                    words = []
                    labels = []

        return lines
    
    def get_labels(self, args):
        labels = set()
        if os.path.exists(os.path.join(args.output_dir, "label_list.pkl")):
             # 如果已经存在标签列表文件，从中加载标签信息
            logger.info(f"loading labels info from {args.output_dir}")
            with open(os.path.join(args.output_dir, "label_list.pkl"), "rb",) as f:
                labels = pickle.load(f)
        else:
            # 若不存在，从训练数据中提取标签信息
            logger.info(f"loading labels info from train file and dump in {args.output_dir}")
            print(args.train_file)
            with open(args.train_file, encoding='utf-8') as f:
                for line in f.readlines():
                    tokens = line.strip().split(" ")
                    if len(tokens) == 2:
                        labels.add(tokens[1])

            if len(labels) > 0:
                with open(os.path.join(args.output_dir, "label_list.pkl"), "wb") as f:
                    pickle.dump(labels, f)
            else:
                logger.info("loading error and return the default labels B,I,O")
                labels = {"O", "B", "I"}
        return labels 

    def get_examples(self, input_file, data=None):
        examples = []
        if data is None:
            data = self.read_from_file(input_file)
        lines = self.read_data(data)

        for i, line in enumerate(lines):
            guid = str(i)
            text = line[1]
            label = line[0]
            examples.append(InputExample(guid=guid, text=text, label=label))
        return examples

# 将示例转换为特征，用于模型输入
# TODO 跳过这一步
def convert_examples_to_features(args, examples, label_list, max_seq_length, tokenizer):
    label_map = {label : i for i, label in enumerate(label_list)}
    features = []
    for (ex_index, example) in tqdm(enumerate(examples), desc="convert examples"):
        # if ex_index % 10000 == 0:
        #     logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        textlist = example.text.split(" ")
        labellist = example.label.split(" ")
        assert len(textlist) == len(labellist)
        tokens = []
        labels = []
        ori_tokens = []
        for i, word in enumerate(textlist):
            # 防止wordPiece情况出现，不过貌似不会
            token = tokenizer.tokenize(word)
            tokens.extend(token)
            label_1 = labellist[i]
            ori_tokens.append(word)
            # 单个字符不会出现wordPiece
            for m in range(len(token)):
                if m == 0:
                    labels.append(label_1)
                else:
                    if label_1 == "O":
                        labels.append("O")
                    else:
                        labels.append("I")
        if len(tokens) >= max_seq_length - 1:
            tokens = tokens[0:(max_seq_length - 2)]  # -2 的原因是因为序列需要加一个句首和句尾标志
            labels = labels[0:(max_seq_length - 2)]
            ori_tokens = ori_tokens[0:(max_seq_length - 2)]

        ori_tokens = ["[CLS]"] + ori_tokens + ["[SEP]"]
        ntokens = []
        segment_ids = []
        label_ids = []
        # 构建输入特征
        ntokens.append("[CLS]")
        segment_ids.append(0)
        label_ids.append(label_map["O"])

        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
            label_ids.append(label_map[labels[i]])
        ntokens.append("[SEP]")
        segment_ids.append(0)
        label_ids.append(label_map["O"])
        input_ids = tokenizer.convert_tokens_to_ids(ntokens)   
        input_mask = [1] * len(input_ids)
        assert len(ori_tokens) == len(ntokens), f"{len(ori_tokens)}, {len(ntokens)}, {ori_tokens}"
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            # we don't concerned about it!
            label_ids.append(0)
            ntokens.append("**NULL**")
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in ntokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))
        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_ids,
                              ori_tokens=ori_tokens))
    return features

# 获取数据集
def get_Dataset(args, processor, tokenizer, mode="train", data=None):
    if mode == "train":
        filepath = args.train_file
    elif mode == "eval":
        filepath = args.eval_file
    elif mode == "test":
        filepath = args.test_file
        print("load test file from:", filepath)
    else:
        raise ValueError("mode must be one of train, eval, or test")

    if data is None:
        examples = processor.get_examples(filepath)
    else:
        examples = processor.get_examples(filepath, data=data.split('\n'))
    # print("examples: ", examples)
    label_list = args.label_list
    # print('label_list: ', label_list)
    features = convert_examples_to_features(
        args, examples, label_list, args.max_seq_length, tokenizer
    )
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    return examples, features, data


def get_args():
    parser = argparse.ArgumentParser()

    ## 必须的参数
    parser.add_argument("--train_file", default=None, type=str)
    parser.add_argument("--eval_file", default=None, type=str)
    parser.add_argument("--test_file", default='../test_text.txt', type=str)
    parser.add_argument("--model_name_or_path", default="../bert-base-chinese", type=str)
    parser.add_argument("--output_dir", default="../output", type=str)

    ## 可选参数
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=256, type=int)
    parser.add_argument("--do_train", default=False, type=bool)
    parser.add_argument("--do_eval", default=False, type=bool)
    parser.add_argument("--do_test", default=True, type=bool)
    parser.add_argument("--train_batch_size", default=8, type=int)
    parser.add_argument("--eval_batch_size", default=8, type=int)
    parser.add_argument("--learning_rate", default=3e-5, type=float)
    parser.add_argument("--num_train_epochs", default=10, type=float)
    parser.add_argument("--warmup_proprotion", default=0.1, type=float)
    parser.add_argument("--use_weight", default=1, type=int)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--fp16", default=False)
    parser.add_argument("--loss_scale", type=float, default=0)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument("--warmup_steps", default=0, type=int)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--max_steps", default=-1, type=int)
    parser.add_argument("--do_lower_case", action='store_true')
    parser.add_argument("--logging_steps", default=500, type=int)
    parser.add_argument("--clean", default=False, type=bool, help="clean the output dir")
    parser.add_argument("--need_birnn", default=False, type=bool)
    parser.add_argument("--rnn_dim", default=128, type=int)
    parser.add_argument("--sentiment_model_path", default="../output/MLP.pth", type=str)
    parser.add_argument("--sentiment_nums", default=5, type=int)

    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    args.device = device

    processor = NerProcessor()
    label_list = processor.get_labels(args)
    # num_labels = len(label_list)
    args.label_list = label_list
    return args


def make_seq(test_data):
    """
    将输入的文本转换为模型输入格式
    :param test_data: 输入文本
    :return:
    """
    seq = ''
    punctuation_set = {"。", "；", ":", "！", "？", "，"}
    # 解决BUG： 最后无标点会导致最后一个句子的实体无法识别
    test_data = test_data.strip() + '。'
    for i in test_data:
        seq += i + ' O\n'
        if i in punctuation_set:
            seq += '\n'
    return seq
