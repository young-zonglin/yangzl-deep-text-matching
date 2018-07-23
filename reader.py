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
    line_count = 0
    data = {}
    try:
        fin = io.open(filename, 'r', encoding=params.OPEN_FILE_ENCODING,
                  newline='\n', errors='ignore')
    except FileNotFoundError as error:
        print(error)
        return data

    for line in fin:
        # load head n word vectors
        if head_n and head_n.__class__ == int:
            line_count += 1
            if line_count > head_n:
                break
        tokens = line.rstrip().replace('\n', '').split(' ')
        # map是一个类，Python中的高阶函数，类似于Scala中的array.map(func)
        # 将传入的函数作用于传入的可迭代对象（例如list）的每一个元素之上
        # float也是一个类
        # Convert a string or number to a floating point number, if possible.
        data[tokens[0]] = map(float, tokens[1:])
    fin.close()
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


# https://zhidao.baidu.com/question/1830830474764728580.html
addr_to_full = {"n't": 'not', "'m": 'am', "'s": 'is', "'re": 'are',
                "'d": 'would', "'ll": 'will', "'ve": 'have'}


def transform_addr_full_format(token):
    if token in addr_to_full:
        return addr_to_full[token]
    else:
        return token


def _pre_process(src_fname, tgt_fname):
    """
    transform raw train data to standard format
    save them to tgt_file
    in lower case
    训练语料有许多错误，例如拼写错误和标点符号错误 => 去除标点符号
    由于fastText预训练的词向量没有诸如didn't之类的词 => 使用NLTK切分
    去除数字，因为得不到它们的向量
    去除非法符号，例如"''"
    :return: None
    """
    punctuation_pattern = re.compile(params.MATCH_PUNCTUATION_STR)
    number_pattern = re.compile(params.MATCH_NUMBER_STR)
    illegal_char_pattern = re.compile(params.MATCH_ILLEGAL_CHAR_STR)
    with open(src_fname, 'r', encoding=params.OPEN_FILE_ENCODING) as src_file, \
            open(tgt_fname, 'w', encoding=params.SAVE_FILE_ENCODING) as tgt_file:
        for line in src_file:
            field_list = line.split('\t')
            for sentence in [field_list[0], field_list[2]]:
                sentence = _remove_symbols(sentence, params.MATCH_SINGLE_QUOTE_STR)
                sentence = punctuation_pattern.sub(' ', sentence.lower())
                for token in nltk.word_tokenize(sentence):
                    # TODO 数字该如何处理
                    # TODO </s>？
                    # TODO 如何做些基本的校对
                    if token and not number_pattern.match(token) \
                            and not illegal_char_pattern.match(token):
                        token = transform_addr_full_format(token)
                        tgt_file.write(token+' ')
                    else:
                        print(token, 'is a NaN or illegal char')
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


def get_needed_vectors(processed_train_fname, fastText_vecs_fname, needed_vecs_fname):
    """
    read all distinct words from processed train file
    if word not in needed word vectors file, get it's vector from fastText word vectors file
    return needed word vectors dict
    :return: dict, {word: str => vector: float list}
    """
    all_words = _read_words(processed_train_fname)
    needed_word2vec = _load_vectors(needed_vecs_fname)

    is_all_in_needed = True
    for word in all_words:
        if word not in needed_word2vec:
            print(word, 'not in needed word2vec')
            is_all_in_needed = False
    if not is_all_in_needed:
        with open(fastText_vecs_fname, 'r', encoding=params.OPEN_FILE_ENCODING) as fastText_file, \
                open(needed_vecs_fname, 'a', encoding=params.SAVE_FILE_ENCODING) as needed_file:
            line_count = 0
            print('============ In get_needed_vectors() func ============')
            for line in fastText_file:
                line_count += 1
                if line_count % 100000 == 0:
                    print(line_count, 'has been processed')
                tokens = line.rstrip().split(' ')
                word = tokens[0]
                if word in all_words and word not in needed_word2vec:
                    for token in tokens:
                        needed_file.write(token+' ')
                    needed_file.write('\n')
        needed_word2vec = _load_vectors(needed_vecs_fname)
    else:
        print('all words in needed word2vec!')
    return needed_word2vec


if __name__ == '__main__':
    # # ========== test _load_vectors() function ==========
    # needed_word2vec = _load_vectors(params.fastText_EN_PRE_TRAINED_WIKI_WORD_VEC, head_n=50)
    # for word, vector in needed_word2vec.items():
    #     print(word, end=' ')
    #     for value in vector:
    #         print(value, end=' ')
    #     print()

    # # ========== test _remove_symbols() func ==========
    # # str1 = "'Random Number' is what I don't like at all."
    # # str1 = "I don't like 'Random Number'."
    # str1 = "I don't like 'Random Number' at all"
    # print(_remove_symbols(str1, params.MATCH_SINGLE_QUOTE_STR))

    # # ========== test _read_words() function ==========
    # all_distinct_words = _read_words(params.PROCESSED_EN_TRAIN_DATA)
    # for word in all_distinct_words:
    #     print(word)
    # print('total distinct words number:', len(all_distinct_words))

    # # ========== test NLTK ==========
    # sentence = "i'v isn't can't haven't aren't won't i'm it's we're who's where's i'd we'll we've he's."
    # tokens = nltk.word_tokenize(_remove_symbols(sentence, params.MATCH_SINGLE_QUOTE_STR).lower())
    # print(tokens)
    # for token in tokens:
    #     print(token, end=' ')
    # print()

    # ========== test _pre_process() func ==========
    _pre_process(params.CIKM_ENGLISH_TRAIN_DATA, params.PROCESSED_EN_TRAIN_DATA)

    # ========== test _get_needed_vectors() func ==========
    needed_word2vec = get_needed_vectors(processed_train_fname=params.PROCESSED_EN_TRAIN_DATA,
                                         fastText_vecs_fname=params.fastText_EN_PRE_TRAINED_WIKI_WORD_VEC,
                                         needed_vecs_fname=params.PROCESSED_EN_WORD_VEC)
    for word, vector in needed_word2vec.items():
        print(word, end=' ')
        print(vector)

    # ========== other test ==========
