import keras
from keras.layers import Dropout, Dense

from layers.layers import AvgEmb
from models.basic_model import BasicModel


class AvgSeqDenseModel(BasicModel):
    def __init__(self):
        super(AvgSeqDenseModel, self).__init__()

    def _do_build(self, src1_word_vec_seq, src2_word_vec_seq, src1_seq, src2_seq):
        avg_seq = AvgEmb(self.word_vec_dim, name='seq_avg')
        src1_encoding = avg_seq(src1_word_vec_seq)
        src2_encoding = avg_seq(src2_word_vec_seq)
        # assert avg_seq.get_output_shape_at(0) == (self.batch_size, self.word_vec_dim)
        # assert avg_seq.get_output_shape_at(1) == (self.batch_size, self.word_vec_dim)

        merged_vec = keras.layers.concatenate([src1_encoding, src2_encoding], axis=-1)
        p_dropout = self.hyperparams.p_dropout
        middle_vec = Dropout(p_dropout)(merged_vec)

        dense_layer_num = self.hyperparams.dense_layer_num
        linear_unit_num = self.hyperparams.linear_unit_num
        for _ in range(dense_layer_num):
            middle_vec = Dense(linear_unit_num, activation='relu')(middle_vec)
            middle_vec = Dropout(p_dropout)(middle_vec)

        preds = Dense(1, activation='sigmoid', name='sigmoid_output_layer')(middle_vec)
        return preds
