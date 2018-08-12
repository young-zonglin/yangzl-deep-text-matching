import keras
from keras import regularizers
from keras.layers import Dropout, Bidirectional, LSTM, Dense

from layers import RNMT_plus
from models.basic_model import BasicModel
from utils.tools import UnitReduceDense


class RNMTPlusEncoderBiLSTMDenseModel(BasicModel):
    def __init__(self):
        super(RNMTPlusEncoderBiLSTMDenseModel, self).__init__()

    def _do_build(self, src1_word_vec_seq, src2_word_vec_seq, src1_seq, src2_seq):
        input_dropout = Dropout(self.hyperparams.lstm_p_dropout, name='input_dropout')
        src1_word_vec_seq = input_dropout(src1_word_vec_seq)
        src2_word_vec_seq = input_dropout(src2_word_vec_seq)

        RNMT_plus_encoder = RNMT_plus.Encoder(self.hyperparams.retseq_layer_num,
                                              self.hyperparams.state_dim,
                                              self.hyperparams.lstm_p_dropout,
                                              self.hyperparams.kernel_l2_lambda,
                                              self.hyperparams.bias_l2_lambda,
                                              self.hyperparams.recurrent_l2_lambda,
                                              self.hyperparams.activity_l2_lambda)
        src1_context_repr_seq = RNMT_plus_encoder(src1_word_vec_seq)
        src2_context_repr_seq = RNMT_plus_encoder(src2_word_vec_seq)

        enc_lstm = LSTM(self.hyperparams.state_dim,
                        kernel_regularizer=regularizers.l2(self.hyperparams.kernel_l2_lambda),
                        recurrent_regularizer=regularizers.l2(self.hyperparams.recurrent_l2_lambda),
                        bias_regularizer=regularizers.l2(self.hyperparams.bias_l2_lambda),
                        activity_regularizer=regularizers.l2(self.hyperparams.activity_l2_lambda))
        enc_bilstm = Bidirectional(enc_lstm, name='enc_bilstm')
        enc_dropout = Dropout(self.hyperparams.lstm_p_dropout, name='enc_dropout')
        src1_encoding = enc_bilstm(src1_context_repr_seq)
        src2_encoding = enc_bilstm(src2_context_repr_seq)
        src1_encoding = enc_dropout(src1_encoding)
        src2_encoding = enc_dropout(src2_encoding)

        merged_vec = keras.layers.concatenate([src1_encoding, src2_encoding], axis=-1)
        middle_vec = UnitReduceDense(self.hyperparams.dense_layer_num,
                                     self.hyperparams.initial_unit_num,
                                     self.hyperparams.dense_p_dropout,
                                     self.hyperparams.kernel_l2_lambda,
                                     self.hyperparams.bias_l2_lambda,
                                     self.hyperparams.activity_l2_lambda,
                                     reduce=self.hyperparams.unit_reduce)(merged_vec)
        preds = Dense(1, activation='sigmoid', name='logistic_output_layer')(middle_vec)
        return preds
