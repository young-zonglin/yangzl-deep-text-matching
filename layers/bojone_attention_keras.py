#! -*- coding: utf-8 -*-
"""
The bojone's implementation.
https://github.com/bojone/attention
"""
from keras import backend as K
from keras.engine.topology import Layer


class PositionEncoding(Layer):
    def __init__(self, d_pos_enc=None, mode='sum', **kwargs):
        super(PositionEncoding, self).__init__(**kwargs)
        self.d_pos_enc = d_pos_enc  # 必须为偶数
        self.mode = mode

    def call(self, x, **kwargs):
        if not self.d_pos_enc or self.mode == 'sum':
            self.d_pos_enc = int(x.shape[-1])
        position_j = 1. / K.pow(10000., 2 * K.arange(self.d_pos_enc/2, dtype='float32') / self.d_pos_enc)
        position_j = K.expand_dims(position_j, 0)
        position_i = K.cumsum(K.ones_like(x[:, :, 0]), 1)-1  # K.arange不支持变长，只好用这种方法生成
        position_i = K.expand_dims(position_i, 2)
        position_ij = K.dot(position_i, position_j)
        position_ij = K.concatenate([K.cos(position_ij), K.sin(position_ij)], 2)
        if self.mode == 'sum':
            return position_ij + x
        elif self.mode == 'concat':
            return K.concatenate([position_ij, x], -1)
        
    def compute_output_shape(self, input_shape):
        if self.mode == 'sum':
            return input_shape
        elif self.mode == 'concat':
            return input_shape[0], input_shape[1], input_shape[2]+self.d_pos_enc

    def get_config(self):
        config = {'size': self.d_pos_enc,
                  'mode': self.mode}
        base_config = super(PositionEncoding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class MultiHeadAttn(Layer):
    def __init__(self, n_head, size_per_head, **kwargs):
        self.n_head = n_head
        self.size_per_head = size_per_head
        self.output_dim = n_head * size_per_head
        super(MultiHeadAttn, self).__init__(**kwargs)

    def build(self, input_shape):
        self.WQ = self.add_weight(name='WQ', 
                                  shape=(input_shape[0][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WK = self.add_weight(name='WK', 
                                  shape=(input_shape[1][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WV = self.add_weight(name='WV', 
                                  shape=(input_shape[2][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        super(MultiHeadAttn, self).build(input_shape)
        
    @staticmethod
    def mask(inputs, seq_len, mode='mul'):
        if not seq_len:
            return inputs
        else:
            mask = K.one_hot(seq_len[:, 0], K.shape(inputs)[1])
            mask = 1 - K.cumsum(mask, 1)
            for _ in range(len(inputs.shape)-2):
                mask = K.expand_dims(mask, 2)
            if mode == 'mul':
                return inputs * mask
            if mode == 'add':
                return inputs - (1 - mask) * 1e12
                
    def call(self, x, **kwargs):
        # 如果只传入Q_seq, K_seq, V_seq，那么就不做Mask
        # 如果同时传入Q_seq, K_seq, V_seq, Q_len, V_len，那么对多余部分做Mask
        if len(x) == 3:
            Q_seq, K_seq, V_seq = x
            Q_len, V_len = None,None
        elif len(x) == 5:
            Q_seq, K_seq, V_seq, Q_len, V_len = x
        # 对Q、K、V做线性变换
        Q_seq = K.dot(Q_seq, self.WQ)
        Q_seq = K.reshape(Q_seq, (-1, K.shape(Q_seq)[1], self.n_head, self.size_per_head))
        Q_seq = K.permute_dimensions(Q_seq, (0, 2, 1, 3))
        K_seq = K.dot(K_seq, self.WK)
        K_seq = K.reshape(K_seq, (-1, K.shape(K_seq)[1], self.n_head, self.size_per_head))
        K_seq = K.permute_dimensions(K_seq, (0, 2, 1, 3))
        V_seq = K.dot(V_seq, self.WV)
        V_seq = K.reshape(V_seq, (-1, K.shape(V_seq)[1], self.n_head, self.size_per_head))
        V_seq = K.permute_dimensions(V_seq, (0, 2, 1, 3))
        # 计算内积，然后mask，然后softmax
        A = K.batch_dot(Q_seq, K_seq, axes=[3, 3]) / self.size_per_head**0.5
        A = K.permute_dimensions(A, (0, 3, 2, 1))
        A = self.mask(A, V_len, 'add')
        A = K.permute_dimensions(A, (0, 3, 2, 1))
        A = K.softmax(A)
        # 输出并mask
        O_seq = K.batch_dot(A, V_seq, axes=[3, 2])
        O_seq = K.permute_dimensions(O_seq, (0, 2, 1, 3))
        O_seq = K.reshape(O_seq, (-1, K.shape(O_seq)[1], self.output_dim))
        O_seq = self.mask(O_seq, Q_len, 'mul')
        return O_seq
        
    def compute_output_shape(self, input_shape):
        return input_shape[0][0], input_shape[0][1], self.output_dim
