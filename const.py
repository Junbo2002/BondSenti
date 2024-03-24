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
TEXT_MAX_LENGTH = 160
ENTITY_MAX_LENGTH = 16

# ===================
# 这里可能有交叉引用的问题，所以放在最后导入
# 消歧模型
# ===================
from disambiguator import StringMatchingDisambiguator, ConnectedComponentsDisambiguator

DISAMBIGUTOR_DICT = {
    "close": StringMatchingDisambiguator(),
    "open": ConnectedComponentsDisambiguator()
}

