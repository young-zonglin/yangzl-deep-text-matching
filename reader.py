import io
import params


def load_vectors(filename, head_n=None):
    """
    装载前N个词向量
    :param filename:
    :param head_n: head n word vectors will be loaded
    :return: dict, {word: str => vector: float list}
    """
    fin = io.open(filename, 'r', encoding='utf-8', newline='\n', errors='ignore')
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


if __name__ == '__main__':
    word2vec = load_vectors(params.fastText_EN_PRE_TRAINED_WIKI_WORD_VEC, head_n=4)
    for word, vector in word2vec.items():
        print(word, end=' ')
        for value in vector:
            print(value, end=' ')
        print()
