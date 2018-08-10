import keras
from keras import backend as K
from keras.layers import Embedding, Lambda, Dense, Dropout, Bidirectional, LSTM

from layers import transformer
from models.basic_model import BasicModel
from utils.tools import UnitReduceDense


class TransformerEncoderBiLSTMDenseModel(BasicModel):
    def __init__(self):
        super(TransformerEncoderBiLSTMDenseModel, self).__init__()

    def _do_build(self, src1_word_vec_seq, src2_word_vec_seq, src1_seq, src2_seq):
        d_model = self.hyperparams.d_model
        d_inner_hid = self.hyperparams.d_inner_hid
        n_head = self.hyperparams.n_head
        d_k = d_v = self.hyperparams.d_k
        d_pos_enc = self.hyperparams.d_pos_enc
        len_limit = self.max_seq_len
        layers_num = self.hyperparams.layers_num
        p_dropout = self.hyperparams.p_dropout

        # 位置编号从1开始
        # word id亦从1开始
        pos_enc_layer = Embedding(len_limit + 1, d_pos_enc, trainable=False,
                                  weights=[transformer.get_pos_enc_matrix(len_limit + 1, d_pos_enc)],
                                  name='pos_enc_layer')
        transformer_encoder = transformer.Encoder(d_model, d_inner_hid, n_head, d_k, d_v,
                                                  layers_num=layers_num,
                                                  p_dropout=p_dropout,
                                                  pos_enc_layer=pos_enc_layer,
                                                  mode=self.hyperparams.transformer_mode,
                                                  batch_size=self.batch_size)
        get_pos_seq = Lambda(transformer.get_pos_seq, name='get_pos_seq')
        src1_pos = get_pos_seq(src1_seq)
        src2_pos = get_pos_seq(src2_seq)

        # TODO Transformer-encoder-based model does not converge
        # 训练集较小，且模型参数较多，学习能力较强，应该不是欠拟合
        # 输入数据有问题？ => 不是的，可以训练LSTM-based model
        # 模型设计有问题？ => 感觉求平均有点问题，试试用LTSM编码 => 失败了
        # 喂给Encoder的数据有问题？ => 打印看看
        # Transformer Encoder实现有问题？ => 试一试tensor2tensor or 原作者的实现
        # 陷入局部最优？ =>
        # 学习率调度策略是否合理？ => warm up step, lr up, then, down
        src1_seq_repr_seq = transformer_encoder(src1_word_vec_seq, src1_seq, src_pos=src1_pos)
        src2_seq_repr_seq = transformer_encoder(src2_word_vec_seq, src2_seq, src_pos=src2_pos)

        # mask操作，只对非占位符的部分求平均
        def masked_avg_emb(src_seq_repr_seq, src_seq):
            mask = K.cast(K.expand_dims(K.not_equal(src_seq, 0), -1), 'float32')
            src_seq_repr_seq = src_seq_repr_seq * mask
            src_seq_repr_seq = K.mean(src_seq_repr_seq, axis=1, keepdims=True)
            return K.reshape(src_seq_repr_seq, [-1, d_model])

        masked_avg_seq = Lambda(lambda x: masked_avg_emb(x[0], x[1]), name='masked_seq_avg')
        # src1_encoding = masked_avg_seq([src1_seq_repr_seq, src1_seq])
        # src2_encoding = masked_avg_seq([src2_seq_repr_seq, src2_seq])

        enc_bilstm = Bidirectional(LSTM(self.hyperparams.state_dim), name='enc_bilstm')
        enc_dropout = Dropout(self.hyperparams.lstm_p_dropout, name='enc_dropout')
        src1_encoding = enc_bilstm(src1_seq_repr_seq)
        src2_encoding = enc_bilstm(src2_seq_repr_seq)
        src1_encoding = enc_dropout(src1_encoding)
        src2_encoding = enc_dropout(src2_encoding)

        # input tensor => 一系列的Keras层 => output tensor
        # 如果使用了backend函数，例如K.concatenate()或tf.reduce_mean()等，需要使用Lambda层封装它们
        merged_vec = keras.layers.concatenate([src1_encoding, src2_encoding])
        middle_vec = UnitReduceDense(self.hyperparams.dense_layer_num,
                                     self.hyperparams.initial_unit_num,
                                     self.hyperparams.dense_p_dropout,
                                     self.hyperparams.unit_reduce)(merged_vec)
        preds = Dense(1, activation='sigmoid', name='logistic_output_layer')(middle_vec)
        return preds
