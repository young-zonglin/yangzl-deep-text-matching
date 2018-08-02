from keras.layers import Bidirectional, LSTM, Dropout, Add

from layers import LayerNormalization


class EncoderLayer:
    def __init__(self, state_dim, lstm_p_dropout):
        self.retseq_bilstm = Bidirectional(LSTM(state_dim, return_sequences=True), merge_mode='concat')
        self.dropout = Dropout(lstm_p_dropout)
        self.layer_norm = LayerNormalization()

    def __call__(self, enc_input):
        hidden_seq = self.retseq_bilstm(enc_input)
        hidden_seq = self.dropout(hidden_seq)
        if enc_input.shape[-1] == hidden_seq.shape[-1]:
            hidden_seq = Add()([hidden_seq, enc_input])
        return self.layer_norm(hidden_seq)


class Encoder:
    def __init__(self, retseq_layer_num, state_dim, p_dropout):
        self.enc_layers = [EncoderLayer(state_dim, p_dropout) for _ in range(retseq_layer_num)]
        self.enc_bilstm = Bidirectional(LSTM(state_dim), name='enc_bilstm')
        self.enc_dropout = Dropout(p_dropout, name='enc_dropout')

    def __call__(self, word_vec_seq):
        x = word_vec_seq
        for enc_layer in self.enc_layers:
            x = enc_layer(x)
        x = self.enc_bilstm(x)
        x = self.enc_dropout(x)
        return x
