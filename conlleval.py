# Python版本的CoNLL评估脚本，用于评估标签化结果是否符合CoNLL格式
# - 默认情况下接受任何空格作为分隔符
# - 可选的文件参数（默认为标准输入）
# - 设置边界的选项（-b参数）
# - 不支持LaTeX输出（-l参数）
# - 不支持原始标签（-r参数）

import sys
import re
import codecs
from collections import defaultdict, namedtuple

ANY_SPACE = '<SPACE>'


class FormatError(Exception):
    pass

Metrics = namedtuple('Metrics', 'tp fp fn prec rec fscore')


# 用于保存评估统计信息的类
class EvalCounts(object):
    def __init__(self):
        self.correct_chunk = 0    # 正确识别的块数量
        self.correct_tags = 0     # 正确的块标签数量
        self.found_correct = 0    # 数据集中的块数量
        self.found_guessed = 0    # 识别出的块数量
        self.token_counter = 0    # 令牌计数器（忽略句子分隔符）
        # 按类型统计
        self.t_correct_chunk = defaultdict(int)
        self.t_found_correct = defaultdict(int)
        self.t_found_guessed = defaultdict(int)


# 解析命令行参数
def parse_args(argv):
    import argparse
    parser = argparse.ArgumentParser(
        description='evaluate tagging results using CoNLL criteria',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    arg = parser.add_argument
    arg('-b', '--boundary', metavar='STR', default='-X-',
        help='sentence boundary')
    arg('-d', '--delimiter', metavar='CHAR', default=ANY_SPACE,
        help='character delimiting items in input')
    arg('-o', '--otag', metavar='CHAR', default='O',
        help='alternative outside tag')
    arg('file', nargs='?', default=None)
    return parser.parse_args(argv)

# 解析标签，返回标签和类型
def parse_tag(t):
    m = re.match(r'^([^-]*)-(.*)$', t)
    return m.groups() if m else (t, '')

# 评估函数
def evaluate(iterable, options=None):
    if options is None:
        options = parse_args([])    # 使用默认值
    counts = EvalCounts()
    num_features = None       # 每行特征数
    in_correct = False        # 当前处理的块是否正确
    last_correct = 'O'        # 数据集中前一个块的标签
    last_correct_type = ''    # 上一个块的类型
    last_guessed = 'O'        # 之前识别出的块的标签
    last_guessed_type = ''    # 数据集中上一个块的标签类型

    for line in iterable:
        line = line.rstrip('\r\n')
        if options.delimiter == ANY_SPACE:
            features = line.split()
        else:
            features = line.split(options.delimiter)
        if num_features is None:
            num_features = len(features)
        elif num_features != len(features) and len(features) != 0:
            raise FormatError('unexpected number of features: %d (%d)' %
                              (len(features), num_features))
        if len(features) == 0 or features[0] == options.boundary:
            features = [options.boundary, 'O', 'O']
        if len(features) < 3:
            raise FormatError('unexpected number of features in line %s' % line)
        guessed, guessed_type = parse_tag(features.pop())
        correct, correct_type = parse_tag(features.pop())
        first_item = features.pop(0)
        if first_item == options.boundary:
            guessed = 'O'
        end_correct = end_of_chunk(last_correct, correct,
                                   last_correct_type, correct_type)
        end_guessed = end_of_chunk(last_guessed, guessed,
                                   last_guessed_type, guessed_type)
        start_correct = start_of_chunk(last_correct, correct,
                                       last_correct_type, correct_type)
        start_guessed = start_of_chunk(last_guessed, guessed,
                                       last_guessed_type, guessed_type)

        if in_correct:
            if (end_correct and end_guessed and
                last_guessed_type == last_correct_type):
                in_correct = False
                counts.correct_chunk += 1
                counts.t_correct_chunk[last_correct_type] += 1
            elif (end_correct != end_guessed or guessed_type != correct_type):
                in_correct = False
        if start_correct and start_guessed and guessed_type == correct_type:
            in_correct = True
        if start_correct:
            counts.found_correct += 1
            counts.t_found_correct[correct_type] += 1
        if start_guessed:
            counts.found_guessed += 1
            counts.t_found_guessed[guessed_type] += 1
        if first_item != options.boundary:
            if correct == guessed and guessed_type == correct_type:
                counts.correct_tags += 1
            counts.token_counter += 1
        last_guessed = guessed
        last_correct = correct
        last_guessed_type = guessed_type
        last_correct_type = correct_type
    if in_correct:
        counts.correct_chunk += 1
        counts.t_correct_chunk[last_correct_type] += 1
    return counts



def uniq(iterable):
  seen = set()
  return [i for i in iterable if not (i in seen or seen.add(i))]


# 计算指标
def calculate_metrics(correct, guessed, total):
    tp, fp, fn = correct, guessed-correct, total-correct
    p = 0 if tp + fp == 0 else 1.*tp / (tp + fp)
    r = 0 if tp + fn == 0 else 1.*tp / (tp + fn)
    f = 0 if p + r == 0 else 2 * p * r / (p + r)
    return Metrics(tp, fp, fn, p, r, f)

# 计算总体和各类型指标
def metrics(counts):
    c = counts
    overall = calculate_metrics(
        c.correct_chunk, c.found_guessed, c.found_correct
    )
    by_type = {}
    for t in uniq(list(c.t_found_correct) + list(c.t_found_guessed)):
        by_type[t] = calculate_metrics(
            c.t_correct_chunk[t], c.t_found_guessed[t], c.t_found_correct[t]
        )
    return overall, by_type

# 打印评估报告
def report(counts, out=None):
    if out is None:
        out = sys.stdout
    overall, by_type = metrics(counts)
    c = counts
    out.write('processed %d tokens with %d phrases; ' %
              (c.token_counter, c.found_correct))
    out.write('found: %d phrases; correct: %d.\n' %
              (c.found_guessed, c.correct_chunk))
    if c.token_counter > 0:
        out.write('accuracy: %6.2f%%; ' %
                  (100.*c.correct_tags/c.token_counter))
        out.write('precision: %6.2f%%; ' % (100.*overall.prec))
        out.write('recall: %6.2f%%; ' % (100.*overall.rec))
        out.write('FB1: %6.2f\n' % (100.*overall.fscore))
    for i, m in sorted(by_type.items()):
        out.write('%17s: ' % i)
        out.write('precision: %6.2f%%; ' % (100.*m.prec))
        out.write('recall: %6.2f%%; ' % (100.*m.rec))
        out.write('FB1: %6.2f  %d\n' % (100.*m.fscore, c.t_found_guessed[i]))

# 打印评估报告，但不直接打印，而是将结果存储在列表中
def report_notprint(counts, out=None):
    if out is None:
        out = sys.stdout
    overall, by_type = metrics(counts)
    c = counts
    final_report = []
    line = []
    line.append('processed %d tokens with %d phrases; ' %
              (c.token_counter, c.found_correct))
    line.append('found: %d phrases; correct: %d.\n' %
              (c.found_guessed, c.correct_chunk))
    final_report.append("".join(line))
    if c.token_counter > 0:
        line = []
        line.append('accuracy: %6.2f%%; ' %
                  (100.*c.correct_tags/c.token_counter))
        line.append('precision: %6.2f%%; ' % (100.*overall.prec))
        line.append('recall: %6.2f%%; ' % (100.*overall.rec))
        line.append('FB1: %6.2f\n' % (100.*overall.fscore))
        final_report.append("".join(line))
    for i, m in sorted(by_type.items()):
        line = []
        line.append('%17s: ' % i)
        line.append('precision: %6.2f%%; ' % (100.*m.prec))
        line.append('recall: %6.2f%%; ' % (100.*m.rec))
        line.append('FB1: %6.2f  %d\n' % (100.*m.fscore, c.t_found_guessed[i]))
        final_report.append("".join(line))
    return final_report


# 判断块是否结束
def end_of_chunk(prev_tag, tag, prev_type, type_):
    # 检查前一个单词和当前单词之间是否结束了一个块
    # 参数：前一个和当前块的标记，前一个和当前块的类型
    chunk_end = False
    if prev_tag == 'E': chunk_end = True
    if prev_tag == 'S': chunk_end = True
    if prev_tag == 'B' and tag == 'B': chunk_end = True
    if prev_tag == 'B' and tag == 'S': chunk_end = True
    if prev_tag == 'B' and tag == 'O': chunk_end = True
    if prev_tag == 'I' and tag == 'B': chunk_end = True
    if prev_tag == 'I' and tag == 'S': chunk_end = True
    if prev_tag == 'I' and tag == 'O': chunk_end = True
    if prev_tag != 'O' and prev_tag != '.' and prev_type != type_:
        chunk_end = True
    if prev_tag == ']': chunk_end = True
    if prev_tag == '[': chunk_end = True
    return chunk_end


# 判断块是否开始
def start_of_chunk(prev_tag, tag, prev_type, type_):
    # 检查前一个单词和当前单词之间是否开始了一个块
    # 参数：前一个和当前块的标记，前一个和当前块的类型
    if tag == 'B': chunk_start = True
    if tag == 'S': chunk_start = True
    if prev_tag == 'E' and tag == 'E': chunk_start = True
    if prev_tag == 'E' and tag == 'I': chunk_start = True
    if prev_tag == 'S' and tag == 'E': chunk_start = True
    if prev_tag == 'S' and tag == 'I': chunk_start = True
    if prev_tag == 'O' and tag == 'E': chunk_start = True
    if prev_tag == 'O' and tag == 'I': chunk_start = True
    if tag != 'O' and tag != '.' and prev_type != type_:
        chunk_start = True
    # these chunks are assumed to have length 1
    if tag == '[': chunk_start = True
    if tag == ']': chunk_start = True
    return chunk_start

# 返回评估报告的列表，从给定的文件中读取数据并评估
def return_report(input_file):
    with codecs.open(input_file, "r", "utf8") as f:
        counts = evaluate(f)
    return report_notprint(counts)


def main(argv):
    args = parse_args(argv[1:])
    if args.file is None:
        counts = evaluate(sys.stdin, args)
    else:
        with open(args.file) as f:
            counts = evaluate(f, args)
    report(counts)


# 在脚本直接运行时执行主函数
if __name__ == '__main__':
    sys.exit(main(sys.argv))