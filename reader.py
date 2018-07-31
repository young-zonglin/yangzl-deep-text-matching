import io
import os
import re

import nltk
import numpy as np
import numpy.random as random
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

import params
import tools


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
                sentence = tools.remove_symbols(sentence, params.MATCH_SINGLE_QUOTE_STR)
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
                sentence = tools.remove_symbols(sentence, params.MATCH_SINGLE_QUOTE_STR)
                sentence = punctuation_pattern.sub(' ', sentence.lower())
                for token in nltk.word_tokenize(sentence):
                    # TODO 数字该如何处理
                    # TODO </s>？
                    # TODO 如何做些基本的校对
                    if token and not number_pattern.match(token) \
                            and not illegal_char_pattern.match(token):
                        token = tools.transform_addr_full_format(token)
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
            source1, source2 = field_list[0], field_list[1]
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


def split_train_val_test(raw_fname, train_fname, val_fname, test_fname):
    """
    randomly split raw data to train data, val data and test data
    train : val : test = 8:1:1
    :param raw_fname:
    :param train_fname:
    :param val_fname:
    :param test_fname:
    :return: None
    """
    if os.path.exists(train_fname) and os.path.exists(val_fname) and os.path.exists(test_fname):
        print('\n======== In split_train_val_test function ========')
        print('Train, val and test data already exists')
        return
    with open(raw_fname, 'r', encoding=params.OPEN_FILE_ENCODING) as raw_file, \
            open(train_fname, 'w', encoding=params.SAVE_FILE_ENCODING) as train_file, \
            open(val_fname, 'w', encoding=params.SAVE_FILE_ENCODING) as val_file, \
            open(test_fname, 'w', encoding=params.SAVE_FILE_ENCODING) as test_file:
        for line in raw_file:
            rand_value = random.rand()
            if rand_value >= 0.2:
                train_file.write(line)
            elif 0.1 <= rand_value < 0.2:
                val_file.write(line)
            else:
                test_file.write(line)


def load_pretrained_vecs(fname):
    """
    load needed word vectors
    :param fname:
    :return: dict, {word: str => embedding: numpy array}
    """
    word2vec = {}
    with open(fname, 'r', encoding=params.OPEN_FILE_ENCODING) as vecs_file:
        for line in vecs_file:
            tokens = line.rstrip().replace('\n', '').split(' ')
            word = tokens[0]
            embedding = np.asarray(tokens[1:], dtype=np.float32)
            word2vec[word] = embedding
    return word2vec


def get_embedding_matrix(word2id, word2vec, vec_dim):
    """
    turn word2vec dict to embedding matrix
    :param word2id: dict
    :param word2vec: dict
    :param vec_dim: embedding dim
    :return: embedding matrix
    """
    embedding_matrix = np.zeros((len(word2id)+1, vec_dim))
    for word, index in word2id.items():
        embedding = word2vec.get(word)
        # words not found in word2vec will be all-zeros.
        if embedding is not None:
            embedding_matrix[index] = embedding
    return embedding_matrix


def fit_tokenizer(fname):
    file = open(fname, 'r', encoding=params.OPEN_FILE_ENCODING)
    text = file.read()
    file.close()
    texts = [text]
    # 不过滤低频词
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    return tokenizer


def generate_in_out_pair_file(fname, tokenizer):
    """
    generate func, generate a input-output pair at a time
    yield a tuple at a time
    (source1_word_id_seq, source2_word_id_list, label)
    :param fname:
    :param tokenizer:
    :return: a iterator
    """
    with open(fname, 'r', encoding=params.OPEN_FILE_ENCODING) as file:
        for line in file:
            if line and line != '\n':
                field_list = line.split('\t')
                if len(field_list) != 3:
                    continue
                source1, source2, label = field_list[0], field_list[1], field_list[2]
                encodeds = tokenizer.texts_to_sequences([source1, source2])
                yield encodeds[0], encodeds[1], label


def process_format_model_in(in_out_pairs, max_len, batch_size, pad='pre', cut='pre'):
    """
    处理输入输出对的格式，使得符合模型的输入要求
    :param in_out_pairs: [(s1, s2, label), (word id list, list, str), ...]
    :param max_len: 最长序列（切词之后）的长度
    :param batch_size:
    :param pad:
    :param cut:
    :return: ({'source1': S1, 'source2': S2}, y)
    S1.shape == S2.shape: 2d numpy array
    y.shape == (in_out_pairs len, vocab_size+1)
    """
    S1 = []
    S2 = []
    y = []
    for in_out_pair in in_out_pairs:
        S1.append(in_out_pair[0])
        S2.append(in_out_pair[1])
        y.append(int(in_out_pair[2]))

    # lists of list => 2d numpy array
    S1 = pad_sequences(S1, maxlen=max_len, padding=pad, truncating=cut)
    S2 = pad_sequences(S2, maxlen=max_len, padding=pad, truncating=cut)

    # binary classification problem
    y = np.asarray(y, dtype=np.int16).reshape(batch_size, 1)
    return {'source1': S1, 'source2': S2}, y


def generate_batch_data_file(fname, tokenizer, max_len, batch_size, pad, cut):
    """
    生成器函数，一次生成一个批的数据
    会在数据集上无限循环
    :param fname:
    :param tokenizer:
    :param max_len:
    :param batch_size:
    :param pad:
    :param cut:
    :return: 返回迭代器，可以遍历由fname生成的batch data的集合
    """
    while True:
        batch_samples_count = 0
        in_out_pairs = list()
        for in_out_pair in generate_in_out_pair_file(fname, tokenizer):
            # 每次生成一个批的数据，每次返回固定相同数目的样本
            if batch_samples_count < batch_size - 1:
                in_out_pairs.append(in_out_pair)
                batch_samples_count += 1
            else:
                in_out_pairs.append(in_out_pair)
                X, y = process_format_model_in(in_out_pairs, max_len, batch_size, pad, cut)
                yield X, y
                in_out_pairs = list()
                batch_samples_count = 0


if __name__ == '__main__':
    # # ========== test _load_vectors() function ==========
    # needed_word2vec = _load_vectors(params.fastText_EN_PRE_TRAINED_WIKI_WORD_VEC, head_n=50)
    # for word, vector in needed_word2vec.items():
    #     print(word, end=' ')
    #     for value in vector:
    #         print(value, end=' ')
    #     print()

    # # ========== test _read_words() function ==========
    # all_distinct_words = _read_words(params.PROCESSED_EN_TRAIN_DATA)
    # for word in all_distinct_words:
    #     print(word)
    # print('total distinct words number:', len(all_distinct_words))

    # ========== test NLTK ==========
    sentence = "i'v     isn't   can't haven't aren't won't i'm it's we're who's where's i'd we'll we've he's."
    tokens = nltk.word_tokenize(tools.remove_symbols(sentence, params.MATCH_SINGLE_QUOTE_STR).lower())
    print(tokens)
    for token in tokens:
        print(token, end=' ')
    print()

    # # ========== test _pre_process() func ==========
    # _pre_process(params.CIKM_ENGLISH_TRAIN_DATA, params.PROCESSED_EN_TRAIN_DATA)
    #
    # # ========== test _get_needed_vectors() func ==========
    # needed_word2vec = get_needed_vectors(processed_train_fname=params.PROCESSED_EN_TRAIN_DATA,
    #                                      fastText_vecs_fname=params.fastText_EN_PRE_TRAINED_WIKI_WORD_VEC,
    #                                      needed_vecs_fname=params.PROCESSED_EN_WORD_VEC)
    # for word, vector in needed_word2vec.items():
    #     print(word, end=' ')
    #     print(vector)

    # ========== other test ==========
