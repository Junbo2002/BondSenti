from __future__ import absolute_import, division, print_function


import os
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)

from tqdm import tqdm, trange

from utils import NerProcessor, convert_examples_to_features, get_Dataset


from pytorch_transformers import (WEIGHTS_NAME, BertConfig, BertTokenizer)


""""
训练和测试过程
1.评估函数 evaluate：用于在验证集或测试集上对模型进行评估。该函数读取验证/测试数据并进行推理，
  将预测的标签与真实标签进行比较，然后调用 conlleval 库进行评估计算。最后返回评估的整体指标和按类型指标。
2.解析参数：通过 argparse 解析命令行参数，包括训练文件、验证文件、测试文件等。
3.训练过程：如果需要训练，就设置优化器、损失函数，遍历数据进行训练。
4.在训练过程中，将损失值、学习率等信息写入 TensorBoard 中，同时可以进行周期性的模型验证和保存。
5.测试过程：如果需要测试，加载已经训练好的模型，将测试数据输入模型进行预测，并将预测结果与真实标签进行对比，将评估结果写入文件。

"""


# TODO 替代文件IO
def forward(args, model):
    # if os.path.exists(os.path.join(args.output_dir, "label2id.pkl")):
    #     with open(os.path.join(args.output_dir, "label2id.pkl"), "rb") as f:
    #         label2id = pickle.load(f)
    # else:
    #     label2id = {l: i for i, l in enumerate(label_list)}
    #     with open(os.path.join(args.output_dir, "label2id.pkl"), "wb") as f:
    #         pickle.dump(label2id, f)
    #
    # id2label = {value: key for key, value in label2id.items()}
    device = args.device
    label_list = args.label_list
    processor = NerProcessor()

    id2label = {i: label for i, label in enumerate(["B-STO", "I-ORG", "B-ORG", "I-STO", "O"])}
    # print(args)
    if args.do_test:
        label_map = {i : label for i, label in enumerate(label_list)}
        tokenizer = BertTokenizer.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        args = torch.load(os.path.join(args.output_dir, 'training_args.bin'))
        # model = BERT_BiLSTM_CRF.from_pretrained(args.output_dir, need_birnn=args.need_birnn, rnn_dim=args.rnn_dim)
        model.to(device)

        test_examples, test_features, test_data = get_Dataset(args, processor, tokenizer, mode="test") 
        all_ori_tokens = [f.ori_tokens for f in test_features]
        all_ori_labels = [e.label.split(" ") for e in test_examples]
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.eval_batch_size)
        model.eval()

        pred_labels = []
        for b_i, (input_ids, input_mask, segment_ids, label_ids) in enumerate(tqdm(test_dataloader, desc="Predicting")):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)
            with torch.no_grad():
                logits = model.predict(input_ids, segment_ids, input_mask)
            pred_labels = [[id2label[idx] for idx in l] for l in logits]

        assert len(pred_labels) == len(all_ori_tokens) == len(all_ori_labels)
        # print("all_ori_tokens:")
        # print(all_ori_tokens)
        # print("pred_labels:")
        # print(pred_labels)
        # with open(os.path.join(args.output_dir, "token_labels_.txt"), "w", encoding="utf-8") as f:
        all_ori_tokens = [i[1: -1] for i in all_ori_tokens]
        pred_labels = [i[1: -1] for i in pred_labels]
        
        # for ori_tokens, ori_labels,prel in zip(all_ori_tokens, all_ori_labels, pred_labels):
        #     for ot,ol,pl in zip(ori_tokens, ori_labels, prel):
        #         if ot in ["[CLS]", "[SEP]"]:
        #             continue
        #         else:
        #             f.write(f"{ot} {ol} {pl}\n")
        #     f.write("\n")

        case_words_org, case_words_sto = load_from_result_test(all_ori_tokens, pred_labels)
        return case_words_org, case_words_sto


def load_from_result_test(case_word_list, case_label_list):
    case_words_org = [[] for i in range(len(case_label_list))]
    case_words_sto = [[] for i in range(len(case_label_list))]
    # print("case_word_list:")
    # print(case_word_list)
    # print("case_label_list:")
    # print(case_label_list)
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
