from __future__ import absolute_import, division, print_function
import os
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from tqdm import tqdm, trange
from utils import NerProcessor, get_Dataset, make_seq
from pytorch_transformers import (WEIGHTS_NAME, BertConfig, BertTokenizer)
from const import DISAMBIGUTOR_DICT
from models import get_entity_rec_model, get_sentiment_model
from utils import get_args
from const import TOKENIZER, TEXT_MAX_LENGTH, ENTITY_MAX_LENGTH, ARGS, DISAMBIGUTOR_DICT


args = ARGS
entity_rec_model = get_entity_rec_model(args)
sentiment_model = get_sentiment_model(args)
entity_rec_model.eval()
sentiment_model.eval()

def tokenize(text, max_length):
    tokenizer = TOKENIZER
    encoded_text = tokenizer.encode_plus(
        text,  # Sentence to encode.
        add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
        max_length=max_length,  # Pad & truncate all sentences.
        padding=True,
        return_attention_mask=True,  # Construct attn. masks.
        return_tensors='pt',  # Return pytorch tensors.
        truncation=True
    )

    return encoded_text


def get_sentiment(text, entity):
    encoded_text = tokenize(text, TEXT_MAX_LENGTH)
    encoded_entity = tokenize(entity, ENTITY_MAX_LENGTH)

    with torch.no_grad():
        encoded_text = entity_rec_model.bert(encoded_text['input_ids'], encoded_text['attention_mask'])
        encoded_entity = entity_rec_model.bert(encoded_entity['input_ids'], encoded_entity['attention_mask'])
        sentiment_predict = sentiment_model(encoded_text[1] + encoded_entity[1])
    # print(sentiment_predict)
    return sentiment_predict[0].tolist()


def get_entity_lst(test_data, disambiMethod="open"):
    assert disambiMethod in DISAMBIGUTOR_DICT.keys()
    disambiguator = DISAMBIGUTOR_DICT[disambiMethod]
    # BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # sys.path.append(BASE_DIR)
    seq = make_seq(test_data)
    print("test_data:", test_data)
    # '--train_file', '../test_text.txt', '--eval_file', '../test_text.txt','--test_file', '../test_text.txt',
    # subprocess.run(['python', '../main/ner.py',
    #                 '--model_name_or_path', "../bert-base-chinese", '--output_dir', '../output'])
    case_words_org, case_words_sto = forward(args, entity_rec_model, seq)
    # print(case_words_org, "\n", case_words_sto)
    # case_words_org, case_words_sto = load_from_result_test('../output/token_labels_.txt')

    # 直接返回实体列表

    entity_lst = [item for sublist in case_words_org + case_words_sto for item in sublist if sublist]
    disambiguated_entity_lst = disambiguator.disambiguate(entity_lst)

    sentiment_res = {
        entity: get_sentiment(test_data, entity) for entity in disambiguated_entity_lst
    }
    return sentiment_res


def forward(args, model, text):
    """
    提取实体
    :param args: 配置参数
    :param model: 实体识别模型
    :param text: 输入文本
    :return: （组织、股票）实体列表 [[],[]], [[浙报传媒控股集团有限公司],[]]
    """
    device = args.device
    label_list = args.label_list
    processor = NerProcessor()

    id2label = {i: label for i, label in enumerate(["B-STO", "I-ORG", "B-ORG", "I-STO", "O"])}
    # print(args)
    if args.do_test:
        label_map = {i : label for i, label in enumerate(label_list)}
        tokenizer = BertTokenizer.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        args = torch.load(os.path.join(args.output_dir, 'training_args.bin'))
        model.to(device)

        test_examples, test_features, test_data = get_Dataset(args, processor, tokenizer, mode="test", data=text)
        all_ori_tokens = [f.ori_tokens for f in test_features]
        all_ori_labels = [e.label.split(" ") for e in test_examples]
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.eval_batch_size)

        pred_labels = []
        with torch.no_grad():
            for b_i, (input_ids, input_mask, segment_ids, label_ids) in enumerate(tqdm(test_dataloader, desc="Predicting")):
                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                segment_ids = segment_ids.to(device)
                label_ids = label_ids.to(device)
                with torch.no_grad():
                    logits = model.predict(input_ids, segment_ids, input_mask)
                pred_labels = [[id2label[idx] for idx in l] for l in logits]

        assert len(pred_labels) == len(all_ori_tokens) == len(all_ori_labels)

        all_ori_tokens = [i[1: -1] for i in all_ori_tokens]
        pred_labels = [i[1: -1] for i in pred_labels]

        case_words_org, case_words_sto = load_from_result_test(all_ori_tokens, pred_labels)
        return case_words_org, case_words_sto


def load_from_result_test(case_word_list, case_label_list):
    """
    从带标签的结果中加载实体
    :param case_word_list: 句子列表 [['深','度','观','察'], [], ...
    :param case_label_list: 句子每个字的标签列表 [['O','O','O','O'], [], ...]
    :return: [[],[]], [[浙报传媒控股集团有限公司],[]]
    """
    case_words_org = [[] for i in range(len(case_label_list))]
    case_words_sto = [[] for i in range(len(case_label_list))]
    for sentence_idx in range(len(case_label_list)):
        label_list = case_label_list[sentence_idx]
        word_list = case_word_list[sentence_idx]
        begin_pos = -1
        for word_idx in range(len(label_list)):
            if label_list[word_idx] == 'O' and begin_pos == -1:
                continue
            if label_list[word_idx] == 'O' and begin_pos != -1:
                reg = ''.join(word_list[begin_pos:word_idx])
                sto_count = 0
                org_count = 0
                for j in range(begin_pos, word_idx):
                    ty = label_list[j][2:]
                    if ty == 'STO':
                        sto_count += 1
                    else:
                        org_count += 1
                if sto_count < org_count:
                    case_words_org[sentence_idx].append(reg)
                else:
                    case_words_sto[sentence_idx].append(reg)
                begin_pos = -1
                continue
            if label_list[word_idx][0] == 'B':
                begin_pos = word_idx
            # if label_list[word_idx][0] == 'I':
            #     continue
    return case_words_org, case_words_sto


if __name__ == "__main__":
    # test()
    pass
