import keras
from keras.layers import Dropout, GlobalAveragePooling1D, Dense

from layers.bojone_attention_keras import MultiHeadAttn, PositionEncoding
from models.basic_model import BasicModel
from utils.tools import UnitReduceDense


class MultiHeadAttnAvgDenseModel(BasicModel):
    def __init__(self):
        super(MultiHeadAttnAvgDenseModel, self).__init__()

    def _do_build(self, src1_word_vec_seq, src2_word_vec_seq, src1_seq, src2_seq):
        pos_enc_layer = PositionEncoding(name='pos_enc_layer')
        input_dropout = Dropout(self.hyperparams.p_dropout, name='input_dropout')
        multi_head_attn_layer = MultiHeadAttn(self.hyperparams.n_head,
                                              self.hyperparams.d_k,
                                              name='multi_head_attn_layer')
        avg_layer = GlobalAveragePooling1D(name='avg_layer')
        enc_dropout = Dropout(self.hyperparams.p_dropout, name='enc_dropout')

        src1_emb_seq = pos_enc_layer(src1_word_vec_seq)
        src2_emb_seq = pos_enc_layer(src2_word_vec_seq)
        src1_emb_seq = input_dropout(src1_emb_seq)
        src2_emb_seq = input_dropout(src2_emb_seq)
        src1_seq_repr_seq = multi_head_attn_layer([src1_emb_seq, src1_emb_seq, src1_emb_seq])
        src2_seq_repr_seq = multi_head_attn_layer([src2_emb_seq, src2_emb_seq, src2_emb_seq])
        src1_encoding = avg_layer(src1_seq_repr_seq)
        src2_encoding = avg_layer(src2_seq_repr_seq)
        src1_encoding = enc_dropout(src1_encoding)
        src2_encoding = enc_dropout(src2_encoding)

        merged_vec = keras.layers.concatenate([src1_encoding, src2_encoding])
        middle_vec = UnitReduceDense(self.hyperparams.dense_layer_num,
                                     self.hyperparams.initial_unit_num,
                                     self.hyperparams.dense_p_dropout,
                                     self.hyperparams.unit_reduce)(merged_vec)
        preds = Dense(1, activation='sigmoid', name='logistic_output_layer')(middle_vec)
        return preds
