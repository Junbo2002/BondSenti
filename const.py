"""全局用的常量"""
from transformers import BertTokenizer
from utils import get_args
ARGS = get_args()

# ===================
# 消歧模型的阈值
# ===================

CLOSE_DISAMBIGUATOR_THRESHOLD = 0.6
OPEN_DISAMBIGUATOR_THRESHOLD = 0.4


# ===================
# 分词器
# ===================
TOKENIZER = BertTokenizer.from_pretrained(ARGS.model_name_or_path,
                                              do_lower_case=ARGS.do_lower_case)

# 文本。实体分词数
TEXT_MAX_LENGTH = 512
ENTITY_MAX_LENGTH = 16

# 测试用例
EXAMPLES = {
    "龙光集团(3380.hk)盘中涨超8%，合景泰富集团(1813.hk)涨超15%。"
    "有媒体称,长江实业已与汇丰接触,希望接手合景泰富与龙光合伙开发的"
    "香港超级豪宅楼盘凯月项目相应债权": {
        "龙光集团": [0.07, 0.13, 0.80],
        "合景泰富集团": [0.05, 0.23, 0.72],
        "汇丰": [0.15, 0.41, 0.44],
        "长江实业": [0.19, 0.61, 0.20]
    }
}


# ===================
# 这里可能有交叉引用的问题，所以放在最后导入
# 消歧模型
# ===================
from disambiguator import StringMatchingDisambiguator, ConnectedComponentsDisambiguator

DISAMBIGUTOR_DICT = {
    "close": StringMatchingDisambiguator(),
    "open": ConnectedComponentsDisambiguator()
}

