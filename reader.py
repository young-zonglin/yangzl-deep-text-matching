import io
import re

import nltk

import params


def _load_vectors(filename, head_n=None):
    """
    装载前N个词向量
    :param filename:
    :param head_n: head n word vectors will be loaded
    :return: dict, {word: str => vector: float list}
    """
    fin = io.open(filename, 'r', encoding=params.OPEN_FILE_ENCODING,
                  newline='\n', errors='ignore')
    line_count = 0
    data = {}
    for line in fin:
        # load head n word vectors
        if head_n and head_n.__class__ == int:
            line_count += 1
            if line_count > head_n:
                break
        tokens = line.rstrip().split(' ')
        # map是一个类，Python中的高阶函数，类似于Scala中的array.map(func)
        # 将传入的函数作用于传入的可迭代对象（例如list）的每一个元素之上
        # float也是一个类
        # Convert a string or number to a floating point number, if possible.
        data[tokens[0]] = map(float, tokens[1:])
    return data


def _remove_symbols(seq, pattern_str):
    """
    remove specified symbol from seq
    :param seq:
    :param pattern_str: 例如text-preprocess项目的remove_comma_from_number()
    :return: new seq
    """
    match_symbol_pattern = re.compile(pattern_str)
    while True:
        matched_obj = match_symbol_pattern.search(seq)
        if matched_obj:
            matched_str = matched_obj.group()
            # print('matched_str:', matched_str)
            matched_symbol = matched_obj.group(1)
            seq = seq.replace(matched_str, matched_str.replace(matched_symbol, ''))
        else:
            break
    return seq


def _pre_process_nltk(src_fname, tgt_fname):
    """
    transform raw train data to standard format
    save them to tgt_file
    in lower case
    使用NLTK做英文切词，主要切标点符号和didn't之类
    :return: None
    """
    with open(src_fname, 'r', encoding=params.OPEN_FILE_ENCODING) as src_file, \
            open(tgt_fname, 'w', encoding=params.SAVE_FILE_ENCODING) as tgt_file:
        for line in src_file:
            field_list = line.split('\t')
            for sentence in [field_list[0], field_list[2]]:
                sentence = _remove_symbols(sentence, params.MATCH_SINGLE_QUOTE_STR)
                for token in nltk.word_tokenize(sentence.lower()):
                    if token:
                        tgt_file.write(token + " ")
                tgt_file.write('\t')
            tgt_file.write(field_list[4])


def _pre_process(src_fname, tgt_fname):
    """
    transform raw train data to standard format
    save them to tgt_file
    in lower case
    训练语料有许多错误，例如拼写错误和标点符号错误 => 按照标点符号切词，只保留didn't类似的单引号
    去除数字，因为得不到它们的向量
    去除非法符号，例如"''"
    :return: None
    """
    split_pattern = re.compile(params.SPLIT_PATTERN_STR)
    number_pattern = re.compile(params.MATCH_NUMBER_STR)
    illegal_char_pattern = re.compile(params.MATCH_ILLEGAL_CHAR_STR)
    with open(src_fname, 'r', encoding=params.OPEN_FILE_ENCODING) as src_file, \
            open(tgt_fname, 'w', encoding=params.SAVE_FILE_ENCODING) as tgt_file:
        for line in src_file:
            field_list = line.split('\t')
            for sentence in [field_list[0], field_list[2]]:
                sentence = _remove_symbols(sentence, params.MATCH_SINGLE_QUOTE_STR)
                for token in split_pattern.split(sentence.lower()):
                    if token and not number_pattern.match(token) \
                            and not illegal_char_pattern.match(token):
                        tgt_file.write(token+" ")
                    else:
                        print('===========', token)
                tgt_file.write('\t')
            tgt_file.write(field_list[4])


def _read_words(fname):
    """
    read all distinct words
    :param fname:
    :return: set, {'apple', 'banana', ...}
    """
    ret_words = set()
    with open(fname, 'r', encoding=params.OPEN_FILE_ENCODING) as file:
        for line in file:
            field_list = line.split('\t')
            source1 = field_list[0]
            source2 = field_list[1]
            for word in source1.split(' ')+source2.split(' '):
                if word:
                    ret_words.add(word)
    return ret_words


def _get_needed_vectors():
    pass

if __name__ == '__main__':
    # # ========== test _load_vectors() function ==========
    # word2vec = _load_vectors(params.fastText_EN_PRE_TRAINED_WIKI_WORD_VEC, head_n=4)
    # for word, vector in word2vec.items():
    #     print(word, end=' ')
    #     for value in vector:
    #         print(value, end=' ')
    #     print()

    # # ========== test _remove_symbols() func ==========
    # # str1 = "'Random Number' is what I don't like at all."
    # # str1 = "I don't like 'Random Number'."
    # str1 = "I don't like 'Random Number' at all"
    # print(_remove_symbols(str1, params.MATCH_SINGLE_QUOTE_STR))

    # ========== test _pre_process() func ==========
    _pre_process(params.CIKM_ENGLISH_TRAIN_DATA, params.PROCESSED_EN_TRAIN_DATA)

    # ========== test _read_words() function ==========
    all_distinct_words = _read_words(params.PROCESSED_EN_TRAIN_DATA)
    for word in all_distinct_words:
        print(word)
    print('total distinct words number:', len(all_distinct_words))

    # # ========== test NLTK ==========
    # sentence = "I don't like 'Random Number' when time is eight o'clock."
    # tokens = nltk.word_tokenize(_remove_symbols(sentence, params.MATCH_SINGLE_QUOTE_STR).lower())
    # print(tokens)
    # for token in tokens:
    #     print(token, end=' ')
    # print()

    # ========== other test ==========
