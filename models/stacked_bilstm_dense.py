import keras
from keras import regularizers
from keras.layers import Dropout, LSTM, Bidirectional, Dense

from models.basic_model import BasicModel


class StackedBiLSTMDenseModel(BasicModel):
    def __init__(self):
        super(StackedBiLSTMDenseModel, self).__init__()

    def _do_build(self, src1_word_vec_seq, src2_word_vec_seq, src1_seq, src2_seq):
        lstm_p_dropout = self.hyperparams.lstm_p_dropout
        input_dropout = Dropout(lstm_p_dropout, name='input_dropout')
        src1_hidden_seq = input_dropout(src1_word_vec_seq)
        src2_hidden_seq = input_dropout(src2_word_vec_seq)

        # 缓解LSTM-based model过拟合的问题 => done
        # 过拟合的症状 => 训练集损失还在下降，val loss开始震荡
        # 正则化技术 => L1/L2 regularization; Max norm constraints; Dropout; LN和BN
        # 更多的数据 => 数据增强 => 生成式模型，GANs？
        # 减少参数数量 => RNN和CNN都有参数共享 => 模型复杂度和特征维数应与数据规模成正比
        # 提前结束训练
        bilstm_retseq_layer_num = self.hyperparams.bilstm_retseq_layer_num
        state_dim = self.hyperparams.state_dim
        for _ in range(bilstm_retseq_layer_num):
            this_lstm = LSTM(state_dim, return_sequences=True,
                             kernel_regularizer=regularizers.l2(self.hyperparams.kernel_l2_lambda),
                             recurrent_regularizer=regularizers.l2(self.hyperparams.recurrent_l2_lambda),
                             bias_regularizer=regularizers.l2(self.hyperparams.bias_l2_lambda),
                             activity_regularizer=regularizers.l2(self.hyperparams.activity_l2_lambda))
            this_bilstm = Bidirectional(this_lstm, merge_mode='concat')
            this_dropout = Dropout(lstm_p_dropout)
            src1_hidden_seq = this_bilstm(src1_hidden_seq)
            src2_hidden_seq = this_bilstm(src2_hidden_seq)
            src1_hidden_seq = this_dropout(src1_hidden_seq)
            src2_hidden_seq = this_dropout(src2_hidden_seq)

        enc_lstm = LSTM(state_dim,
                        kernel_regularizer=regularizers.l2(self.hyperparams.kernel_l2_lambda),
                        recurrent_regularizer=regularizers.l2(self.hyperparams.recurrent_l2_lambda),
                        bias_regularizer=regularizers.l2(self.hyperparams.bias_l2_lambda),
                        activity_regularizer=regularizers.l2(self.hyperparams.activity_l2_lambda))
        enc_bilstm = Bidirectional(enc_lstm, name='enc_bilstm')
        enc_dropout = Dropout(lstm_p_dropout, name='enc_dropout')
        src1_encoding = enc_bilstm(src1_hidden_seq)
        src2_encoding = enc_bilstm(src2_hidden_seq)
        src1_encoding = enc_dropout(src1_encoding)
        src2_encoding = enc_dropout(src2_encoding)

        merged_vec = keras.layers.concatenate([src1_encoding, src2_encoding], axis=-1)
        middle_vec = merged_vec

        # a fusion op needed here
        # must use FC layer
        # two Dense layer => like CNN
        # a feedforward layer => like Transformer
        for _ in range(self.hyperparams.dense_layer_num):
            middle_vec = Dense(self.hyperparams.linear_unit_num, activation='relu',
                               kernel_regularizer=regularizers.l2(self.hyperparams.kernel_l2_lambda),
                               bias_regularizer=regularizers.l2(self.hyperparams.bias_l2_lambda),
                               activity_regularizer=regularizers.l2(self.hyperparams.activity_l2_lambda)
                               )(middle_vec)
            middle_vec = Dropout(self.hyperparams.dense_p_dropout)(middle_vec)

        preds = Dense(1, activation='sigmoid', name='logistic_output_layer')(middle_vec)
        return preds
