import tensorflow as tf
from keras import backend as K
from keras.engine.topology import Layer
from keras.initializers import Ones, Zeros

from layers import transformer


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

    def get_config(self):
        config = {'eps': self.eps}
        base_config = super(LayerNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class AvgEmb(Layer):
    def __init__(self, word_vec_dim, **kwargs):
        super(AvgEmb, self).__init__(**kwargs)
        self.word_vec_dim = word_vec_dim

    # 模板方法模式
    def call(self, inputs, **kwargs):
        inputs = tf.reduce_mean(inputs, axis=1, keepdims=True)
        # return Reshape([self.word_vec_dim])(X)
        return tf.reshape(inputs, [-1, self.word_vec_dim])

    # 由于是静态计算图的框架，shape都并不可靠，可能不是预期的值
    # 尽量使用已知值
    def compute_output_shape(self, input_shape, **kwargs):
        return input_shape[0], self.word_vec_dim

    # 和保存相关的方法
    # config = layer.get_config() or model.get_config() => 包含这个层配置信息的dict
    # layer = Layer.from_config(config) or
    # model = Model.from_config(config) or Sequential.from_config(config)
    # 由于Keras其他层没有重写from_config方法，我的自定义层也不重写
    def get_config(self):
        config = {'word_vec_dim': self.word_vec_dim}
        base_config = super(AvgEmb, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Repeat(Layer):
    def __init__(self, rep, axis, batch_size, **kwargs):
        Layer.__init__(self, **kwargs)
        self.rep = rep
        self.axis = axis
        self.batch_size = batch_size

    def call(self, inputs, **kwargs):
        return K.repeat_elements(inputs, self.rep, self.axis)

    def compute_output_shape(self, input_shape):
        axis = self.axis
        if axis == 0:
            return self.rep*self.batch_size, input_shape[1], input_shape[2]
        elif axis == 1:
            return self.batch_size, self.rep*input_shape[1], input_shape[2]
        elif axis == 2:
            return self.batch_size, input_shape[1], self.rep*input_shape[2]
        else:
            raise ValueError('axis not in [0, 1, 2]')

    def get_config(self):
        config = {'rep': self.rep,
                  'axis': self.axis,
                  'batch_size': self.batch_size}
        base_config = Layer.get_config(self)
        # dict.items() will return a set-like object
        return dict(base_config.items() | config.items())


class ScaledDotProduct(Layer):
    def __init__(self, temper, **kwargs):
        super(ScaledDotProduct, self).__init__(**kwargs)
        self.temper = temper

    def call(self, inputs, **kwargs):
        q, k = inputs[0], inputs[1]
        # K.batch_dot, batch-wise dot product
        # batch-wise operation, element-wise operation, point-wise operation
        # axes=[2, 2]意味着“砍掉”batch data tensor.shape的第三维
        return K.batch_dot(q, k, axes=[2, 2]) / self.temper

    def compute_output_shape(self, input_shape):
        input_shape = input_shape[0]
        return input_shape[0], input_shape[1], input_shape[1]

    def get_config(self):
        config = {'temper': self.temper}
        base_config = super(ScaledDotProduct, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class MMask(Layer):
    def __init__(self, **kwargs):
        super(MMask, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        return (-1e+10) * (1-inputs)

    def compute_output_shape(self, input_shape):
        return input_shape


class WeightedSum(Layer):
    def __init__(self, **kwargs):
        super(WeightedSum, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        attn, v = inputs[0], inputs[1]
        # 默认axes=[2, 1]
        return K.batch_dot(attn, v)


class Reshape1(Layer):
    def __init__(self, n_head, d_k, batch_size, **kwargs):
        super(Reshape1, self).__init__(**kwargs)
        self.n_head = n_head
        self.d_k = d_k
        self.batch_size = batch_size

    def call(self, inputs, **kwargs):
        n_head = self.n_head
        d_k = self.d_k
        s = tf.shape(inputs)  # [batch_size, len_q, n_head * d_k]
        inputs = tf.reshape(inputs, [s[0], s[1], n_head, d_k])
        inputs = tf.transpose(inputs, [2, 0, 1, 3])
        # -1意味着自动推断
        inputs = tf.reshape(inputs, [-1, s[1], d_k])  # [n_head * batch_size, len_q, d_k]
        return inputs

    def compute_output_shape(self, input_shape):
        return self.n_head*self.batch_size, input_shape[1], self.d_k

    def get_config(self):
        config = {'n_head': self.n_head,
                  'd_k': self.d_k,
                  'batch_size': self.batch_size}
        base_config = super(Reshape1, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Reshape2(Layer):
    def __init__(self, n_head, d_v, batch_size, **kwargs):
        super(Reshape2, self).__init__(**kwargs)
        self.n_head = n_head
        self.d_v = d_v
        self.batch_size = batch_size

    def call(self, inputs, **kwargs):
        n_head = self.n_head
        d_v = self.d_v
        s = tf.shape(inputs)  # [n_head * batch_size, seq_len, d_v]
        inputs = tf.reshape(inputs, [n_head, -1, s[1], s[2]])
        inputs = tf.transpose(inputs, [1, 2, 0, 3])
        # n_head * s[2]会出错！
        inputs = tf.reshape(inputs, [-1, s[1], n_head * d_v])  # [batch_size, seq_len, n_head * d_v]
        return inputs

    def compute_output_shape(self, input_shape):
        return self.batch_size, input_shape[1], self.n_head * self.d_v

    def get_config(self):
        config = {'n_head': self.n_head,
                  'd_v': self.d_v,
                  'batch_size': self.batch_size}
        base_config = super(Reshape2, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class GetPadMask(Layer):
    def __init__(self, **kwargs):
        super(GetPadMask, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        return transformer.get_pad_mask(inputs, inputs)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], input_shape[1]
