from keras import backend as K
from keras.callbacks import Callback
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

    def __call__(self, word_vec_seq):
        x = word_vec_seq
        for enc_layer in self.enc_layers:
            x = enc_layer(x)
        return x


class LRSchedulerPerStep(Callback):
    def __init__(self, n_model, warmup, start_decay, end_decay):
        super(LRSchedulerPerStep, self).__init__()
        self.basic = 1e-4
        self.n_model = n_model
        self.warmup = warmup
        self.start_decay = start_decay
        self.end_decay = end_decay
        self.step_num = 0

    def on_batch_begin(self, batch, logs = None):
        self.step_num += 1
        t = self.step_num
        n = self.n_model
        p = self.warmup
        s = self.start_decay
        e = self.end_decay
        first = 1+t*(n-1)/(n*p)
        second = n
        third = n*(2*n)**((s-n*t)/(e-s))
        lr = self.basic * min(first, second, third)
        K.set_value(self.model.optimizer.lr, lr)
