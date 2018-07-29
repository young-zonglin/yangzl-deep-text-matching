"""
The implementation of Transformer encoder mainly follows to the following link:
https://github.com/Lsdefine/attention-is-all-you-need-keras
"""
import numpy as np
from keras import backend as K
from keras.engine.topology import Layer
from keras.initializers import Ones, Zeros
from keras.layers import Activation, Dense, Dropout, Conv1D
from keras.layers import TimeDistributed, Concatenate, Add

from layers import Reshape1, Reshape2, Repeat, GetPadMask
from layers import ScaledDotProduct, MMask, WeightedSum


class LayerNormalization(Layer):
    def __init__(self, eps=1e-6, **kwargs):
        super(LayerNormalization, self).__init__(**kwargs)
        self.eps = eps
        self.gamma = None
        self.beta = None

    def build(self, input_shape):
        # Create trainable weight variables for this layer.
        # 不同的行共享gamma和beta
        self.gamma = self.add_weight(name='gamma', shape=input_shape[-1:],
                                     initializer=Ones(), trainable=True)
        self.beta = self.add_weight(name='beta', shape=input_shape[-1:],
                                    initializer=Zeros(), trainable=True)
        super(LayerNormalization, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs, **kwargs):
        mean = K.mean(inputs, axis=-1, keepdims=True)
        std = K.std(inputs, axis=-1, keepdims=True)
        # 类似于BN，LN在对样本归一化后也有缩放和平移操作
        # Python中的广播
        return self.gamma * (inputs - mean) / (std + self.eps) + self.beta

    def compute_output_shape(self, input_shape):
        return input_shape


# input tensor通过一系列Keras的操作，变成output tensor
class ScaledDotProductAttention:
    def __init__(self, d_model, attn_dropout=0.1):
        # TODO dk?
        self.temper = np.sqrt(d_model)
        self.dropout = Dropout(attn_dropout)

    def __call__(self, q, k, v, mask=None):
        """
        self attention: Q = K = V
        NLP: K = V
        :param q: 矩阵，shape=(batch_size, seq_len, word_vec_dim)
        :param k: 同上
        :param v: 同上
        :param mask: shape == (batch_size, seq_len, seq_len), [0, 0, 1, 1, ...], 需要mask的地方就标0
        :return: tuple, output: 不同位置序列的局部表示; attn: 不同位置对不同位置的依赖关系
        """
        attn = ScaledDotProduct(self.temper)([q, k])
        output = None
        # Masked operation
        if mask is not None:
            mmask = MMask()(mask)
            attn = Add()([attn, mmask])
            attn = Activation('softmax')(attn)
            attn = self.dropout(attn)
            output = WeightedSum()([attn, v])
        return output, attn


class MultiHeadAttention:
    # mode 0 - big matrices, faster
    # mode 1 - more clear implementation
    def __init__(self, n_head, d_model, d_k, d_v, p_dropout,
                 mode=0, use_norm=True, batch_size=None):
        self.mode = mode
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.p_dropout = p_dropout
        self.batch_size = batch_size

        # d_k = d_v = d_model/n_head
        if mode == 0:
            self.qs_layer = Dense(n_head * d_k, use_bias=False)
            self.ks_layer = Dense(n_head * d_k, use_bias=False)
            self.vs_layer = Dense(n_head * d_v, use_bias=False)
        elif mode == 1:
            self.qs_layers = []
            self.ks_layers = []
            self.vs_layers = []
            for _ in range(n_head):
                self.qs_layers.append(TimeDistributed(Dense(d_k, use_bias=False)))
                self.ks_layers.append(TimeDistributed(Dense(d_k, use_bias=False)))
                self.vs_layers.append(TimeDistributed(Dense(d_v, use_bias=False)))
        self.attention = ScaledDotProductAttention(d_model)
        self.layer_norm = LayerNormalization() if use_norm else None
        self.w_o = TimeDistributed(Dense(d_model))

    def __call__(self, q, k, v, mask=None):
        d_k, d_v = self.d_k, self.d_v
        n_head = self.n_head
        batch_size = self.batch_size

        head = None
        attn = None
        if self.mode == 0:
            qs = self.qs_layer(q)  # qs.shape == (batch_size, len_q, n_head*d_k)
            ks = self.ks_layer(k)
            vs = self.vs_layer(v)

            reshape1 = Reshape1(n_head, d_k, batch_size)
            qs = reshape1(qs)
            ks = reshape1(ks)
            vs = reshape1(vs)

            if mask is not None:
                mask = Repeat(n_head, 0, batch_size)(mask)
            head, attn = self.attention(qs, ks, vs, mask=mask)
            head = Reshape2(n_head, d_v, batch_size)(head)
        elif self.mode == 1:
            heads = []
            attns = []
            for i in range(n_head):
                qs = self.qs_layers[i](q)
                ks = self.ks_layers[i](k)
                vs = self.vs_layers[i](v)
                head, attn = self.attention(qs, ks, vs, mask)
                heads.append(head)
                attns.append(attn)
            head = Concatenate()(heads) if n_head > 1 else heads[0]
            attn = Concatenate()(attns) if n_head > 1 else attns[0]

        outputs = self.w_o(head)
        outputs = Dropout(self.p_dropout)(outputs)
        if not self.layer_norm:
            return outputs, attn
        outputs = Add()([outputs, q])
        return self.layer_norm(outputs), attn


class PositionwiseFeedForward:
    def __init__(self, d_hid, d_inner_hid, p_dropout=0.1):
        self.w_1 = Conv1D(d_inner_hid, 1, activation='relu')
        self.w_2 = Conv1D(d_hid, 1)
        self.layer_norm = LayerNormalization()
        self.dropout = Dropout(p_dropout)

    def __call__(self, x):
        output = self.w_1(x)
        output = self.w_2(output)
        output = self.dropout(output)
        output = Add()([output, x])
        return self.layer_norm(output)


class EncoderLayer:
    def __init__(self, d_model, d_inner_hid, n_head, d_k, d_v,
                 p_dropout=0.1, mode=0, batch_size=None):
        self.self_att_layer = MultiHeadAttention(n_head, d_model, d_k, d_v,
                                                 p_dropout=p_dropout,
                                                 mode=mode, batch_size=batch_size)
        self.pos_ffn_layer = PositionwiseFeedForward(d_model, d_inner_hid, p_dropout=p_dropout)

    def __call__(self, enc_input, mask=None):
        # self attention
        output, slf_att = self.self_att_layer(enc_input, enc_input, enc_input, mask=mask)
        output = self.pos_ffn_layer(output)
        return output, slf_att


def get_pos_enc_matrix(max_len, d_emb):
    """
    位置编号从1开始
    :param max_len: 应为max_seq_len + 1
    :param d_emb: dim of pos emb
    :return: matrix, like word embedding matrix
    """
    # TODO 这个函数还有待进一步学习
    pos2enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / d_emb) for j in range(d_emb)]
        if pos != 0 else np.zeros(d_emb)
        for pos in range(max_len)
    ])
    pos2enc[1:, 0::2] = np.sin(pos2enc[1:, 0::2])  # dim 2i
    pos2enc[1:, 1::2] = np.cos(pos2enc[1:, 1::2])  # dim 2i+1
    return pos2enc


def get_pad_mask(q, k):
    """
    mask占位符，需要mask的地方就标0，[1, 1, ..., 0, 0, ...]
    对于一个句子而言，它的右边可能会有填充
    :param q: shape == (batch_size, seq_len)
    :param k: 自注意力，q=k
    :return: tensor, mask: shape == (batch_size, seq_len, seq_len)
    """
    # expand_dims操作会改变tensor的shape，元素不变
    # (batch_size, seq_len) => (batch_size, seq_len, 1)
    ones = K.expand_dims(K.ones_like(q, 'float32'), -1)
    # cast: True -> 1; False -> 0
    mask = K.cast(K.expand_dims(K.not_equal(k, 0), 1), 'float32')
    # shape为(seq_len, 1)和(1, seq_len)的两个tensor做点积计算
    mask = K.batch_dot(ones, mask, axes=[2, 1])
    return mask


def get_pos_seq(x):
    """
    获得位置序列，位置从1开始编码
    :param x: shape == (batch_size, seq_len)
    :return: tensor, shape同x
    """
    mask = K.cast(K.not_equal(x, 0), 'int32')
    pos = K.cumsum(K.ones_like(x, 'int32'), 1)
    return pos * mask


class Encoder:
    def __init__(self, d_model, d_inner_hid, n_head, d_k, d_v,
                 layers_num=6, p_dropout=0.1, pos_enc_layer=None, mode=0, batch_size=None):
        self.pos_enc_layer = pos_enc_layer
        self.enc_layers = [EncoderLayer(d_model, d_inner_hid, n_head, d_k, d_v,
                                        p_dropout, mode, batch_size)
                           for _ in range(layers_num)]

    def __call__(self, src_word_vec_seq, src_seq, src_pos, return_attn=False, active_layers=999):
        x = src_word_vec_seq
        if src_pos is not None:
            pos_enc = self.pos_enc_layer(src_pos)
            x = Add()([x, pos_enc])
        attns = []
        mask = GetPadMask()(src_seq)
        # 只激活哪些层
        for enc_layer in self.enc_layers[:active_layers]:
            x, attn = enc_layer(x, mask)
            if return_attn:
                attns.append(attn)
        return (x, attns) if return_attn else x
