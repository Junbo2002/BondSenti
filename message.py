import sys,os
from test import forward
from utils import StringMatchingDisambiguator, ConnectedComponentsDisambiguator

# disambiguator = StringMatchingDisambiguator()
# disambiguator = ConnectedComponentsDisambiguator()

disambiguator_dict = {
    "close": StringMatchingDisambiguator(),
    "open": ConnectedComponentsDisambiguator()
}

def test(test_data, model, args, disambiMethod="open"):
    assert disambiMethod in disambiguator_dict.keys()
    disambiguator = disambiguator_dict[disambiMethod]
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(BASE_DIR)

    # 指定了测试数据的路径 path_test 和预训练模型的路径 PATH_MODEL
    # path_test = '../test.json'  # 部署的时候，manage.py的路径为当前路径
    # with open(path_test, 'r', encoding='utf-8') as f1:
    #     test_data = json.load(f1)
    seq = ''
    # 解决BUG： 最后无标点会导致最后一个句子的实体无法识别
    test_data = test_data.strip() + '.'
    for i in test_data:
        seq += i + ' O\n'
        if i in ["。", "；", ".", ":"]:
            seq += '\n'

    # with open('../processed_data/test.txt', 'w', encoding='utf-8') as f2:
    #     f2.write(seq)
    print("test_data:", test_data)
    # '--train_file', '../test_text.txt', '--eval_file', '../test_text.txt','--test_file', '../test_text.txt',
    # subprocess.run(['python', '../main/ner.py',
    #                 '--model_name_or_path', "../bert-base-chinese", '--output_dir', '../output'])
    case_words_org, case_words_sto = forward(args, model, seq)
    # print(case_words_org, "\n", case_words_sto)
    # case_words_org, case_words_sto = load_from_result_test('../output/token_labels_.txt')

    # 直接返回实体列表

    entity_lst = [item for sublist in case_words_org + case_words_sto for item in sublist if sublist]
    disambiguated_entity_lst = disambiguator.disambiguate(entity_lst)
    return disambiguated_entity_lst

    
    
if __name__ == '__main__':
    # case_words_org, case_words_sto = load_from_result_test('../output/token_labels_.txt')
    pass
    


