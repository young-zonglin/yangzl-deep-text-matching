import os
import re
import time

import matplotlib.pyplot as plt

import params


def remove_symbols(seq, pattern_str):
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

# https://zhidao.baidu.com/question/1830830474764728580.html
addr_to_full = {"n't": 'not', "'m": 'am', "'s": 'is', "'re": 'are',
                "'d": 'would', "'ll": 'will', "'ve": 'have'}


def transform_addr_full_format(token):
    if token in addr_to_full:
        return addr_to_full[token]
    else:
        return token


def get_current_time():
    return time.strftime('%Y-%m-%d %H_%M_%S', time.localtime(time.time()))


def plot_figure(figure_name, *args):
    """
    画图，目前最多画四条曲线，传入一个(x, y)元组，就画一条曲线
    这是一个阻塞函数
    :param figure_name: 图片的名字
    :param args: 变长参数，即参数数目可变
    :return: None
    """
    colors = ['r', 'b', 'g', 'y', 'k']
    styles = ['-', '--', '-.', ':']
    max_args_num = len(styles)
    length = len(args)
    if length > max_args_num:
        print('too much tuple, more than', max_args_num)
        return
    plt.figure(figure_name)
    for i in range(length):
        plt.plot(args[i][0], args[i][1], colors[i]+styles[i], lw=3)
    if not os.path.exists(params.FIGURE_DIR):
        os.makedirs(params.FIGURE_DIR)
    current_time = get_current_time()
    save_url = os.path.join(params.FIGURE_DIR, 'text match_' + current_time + '.png')
    plt.savefig(save_url)
    plt.show()  # it is a blocking function


if __name__ == '__main__':
    # ========== test _remove_symbols() func ==========
    # str1 = "'Random Number' is what I don't like at all."
    # str1 = "I don't like 'Random Number'."
    str1 = "I don't like 'Random Number' at all"
    print(remove_symbols(str1, params.MATCH_SINGLE_QUOTE_STR))
