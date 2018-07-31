import os
import time

import keras
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.layers import Input, Dense, Dropout, Embedding, Lambda, LSTM, Bidirectional
from keras.models import Model
from keras.models import load_model

import net_conf
import params
import reader
import tools
import transformer
from layers import AvgEmb


class BasicModel:
    def __init__(self):
        self.hyperparams = None
        self.max_seq_len = None
        self.vocab_size = None
        self.word_vec_dim = None
        self.batch_size = None
        self.this_model_save_dir = None

        self.pretrained_word_vecs_fname = None
        self.raw_fname = None
        self.train_fname = None
        self.val_fname = None
        self.test_fname = None

        self.model = None
        self.embedding_matrix = None
        self.tokenizer = None

        self.pad = None
        self.cut = None

        self.total_samples_count = 0
        self.train_samples_count = 0
        self.val_samples_count = 0
        self.test_samples_count = 0

    # 静态工厂方法
    @staticmethod
    def make_model(model_name):
        if model_name == "AvgSeqDenseModel":
            return AvgSeqDenseModel()
        elif model_name == 'StackedBiLSTMDenseModel':
            return StackedBiLSTMDenseModel()
        elif model_name == 'TransformerEncoderDenseModel' or model_name == 'TransformerDenseModelTest':
            return TransformerEncoderDenseModel()
        else:
            return BasicModel()

    def setup(self, raw_fname, train_fname, val_fname, test_fname,
              pretrained_word_vecs_fname, hyperparams):
        self.pretrained_word_vecs_fname = pretrained_word_vecs_fname
        reader.split_train_val_test(raw_fname, train_fname, val_fname, test_fname)
        self.raw_fname = raw_fname
        self.train_fname = train_fname
        self.val_fname = val_fname
        self.test_fname = test_fname

        run_which_model = net_conf.RUN_WHICH_MODEL
        which_language = net_conf.WHICH_LANGUAGE
        setup_time = tools.get_current_time()
        self.this_model_save_dir = \
            params.RESULT_SAVE_DIR + os.path.sep + \
            run_which_model + '_' + which_language + '_' + setup_time
        if not os.path.exists(self.this_model_save_dir):
            os.makedirs(self.this_model_save_dir)
        params.MODEL_SAVE_DIR = self.this_model_save_dir

        self.hyperparams = hyperparams
        self.batch_size = self.hyperparams.batch_size
        self.tokenizer = reader.fit_tokenizer(self.raw_fname)
        self.vocab_size = len(self.tokenizer.word_index)

        self.pad = self.hyperparams.pad
        self.cut = self.hyperparams.cut

        for _ in reader.generate_in_out_pair_file(self.raw_fname, self.tokenizer):
            self.total_samples_count += 1
        for _ in reader.generate_in_out_pair_file(self.train_fname, self.tokenizer):
            self.train_samples_count += 1
        for _ in reader.generate_in_out_pair_file(self.val_fname, self.tokenizer):
            self.val_samples_count += 1
        for _ in reader.generate_in_out_pair_file(self.test_fname, self.tokenizer):
            self.test_samples_count += 1

        # compute max seq len
        max_len = -1
        for in_out_pair in reader.generate_in_out_pair_file(self.raw_fname, self.tokenizer):
            for word_id_list in [in_out_pair[0], in_out_pair[1]]:
                length = len(word_id_list)
                if length > max_len:
                    max_len = length
        self.max_seq_len = max_len

        record_info = list()
        record_info.append('\n================ In setup ================\n')
        record_info.append('Vocab size: %d\n' % self.vocab_size)
        record_info.append('Max sentence length: {}\n'.format(self.max_seq_len))
        record_info.append('Total samples count: %d\n' % self.total_samples_count)
        record_info.append('Train samples count: %d\n' % self.train_samples_count)
        record_info.append('Val samples count: %d\n' % self.val_samples_count)
        record_info.append('Test samples count: %d\n' % self.test_samples_count)
        record_str = ''.join(record_info)
        record_url = params.MODEL_SAVE_DIR + os.path.sep + params.TRAIN_RECORD_FNAME
        tools.print_save_str(record_str, record_url)

    def _do_build(self, src1_word_vec_seq, src2_word_vec_seq, src1_seq, src2_seq):
        raise NotImplementedError()

    def build(self):
        """
        define model
        template method pattern
        :return: Model object using the functional API
        """
        source1 = Input(name='source1', shape=(self.max_seq_len,), dtype='int32')
        source2 = Input(name='source2', shape=(self.max_seq_len,), dtype='int32')

        word2vec = reader.load_pretrained_vecs(self.pretrained_word_vecs_fname)
        self.word_vec_dim = net_conf.fastText_EN_WORD_VEC_DIM \
            if 'en' in self.raw_fname.lower() \
            else net_conf.fastText_ES_WORD_VEC_DIM
        # if 'en' in self.raw_fname.lower():
        #     self.word_vec_dim = net_conf.fastText_EN_WORD_VEC_DIM
        # else:
        #     self.word_vec_dim = net_conf.fastText_ES_WORD_VEC_DIM
        self.embedding_matrix = reader.get_embedding_matrix(word2id=self.tokenizer.word_index,
                                                            word2vec=word2vec,
                                                            vec_dim=self.word_vec_dim)
        embedding = Embedding(input_dim=self.vocab_size+1,
                              output_dim=self.word_vec_dim,
                              weights=[self.embedding_matrix],
                              input_length=self.max_seq_len,
                              name='word_embedding',
                              trainable=False)
        src1_word_vec_seq = embedding(source1)
        src2_word_vec_seq = embedding(source2)
        # print(embedding.get_input_shape_at(0))
        # print(embedding.get_output_shape_at(1))

        preds = self._do_build(src1_word_vec_seq, src2_word_vec_seq, source1, source2)
        self.model = Model(inputs=[source1, source2], outputs=preds)

        record_info = list()
        record_info.append('\n================ In build ================\n')
        record_info.append('Found %d word vectors.' % len(word2vec))
        record_str = ''.join(record_info)
        record_url = params.MODEL_SAVE_DIR + os.path.sep + params.TRAIN_RECORD_FNAME
        tools.print_save_str(record_str, record_url)
        print('\n############### Model summary ##################')
        print(self.model.summary())

        return self.model

    # TODO 优化算法，学习率等
    # TODO 动态学习率
    def compile(self):
        self.model.compile(loss='binary_crossentropy',
                           optimizer='rmsprop',
                           metrics=['accuracy'])

    def fit_generator(self):
        train_start = float(time.time())
        early_stopping = EarlyStopping(monitor='val_loss',
                                       patience=self.hyperparams.early_stop_patience,
                                       min_delta=self.hyperparams.early_stop_min_delta,
                                       verbose=1, mode='min')
        save_model = tools.SaveModel()
        save_model.set_model(self.model)
        history = self.model.fit_generator(reader.generate_batch_data_file(self.train_fname,
                                                                           self.tokenizer,
                                                                           self.max_seq_len,
                                                                           self.batch_size,
                                                                           self.pad,
                                                                           self.cut),
                                           validation_data=reader.generate_batch_data_file(self.val_fname,
                                                                                           self.tokenizer,
                                                                                           self.max_seq_len,
                                                                                           self.batch_size,
                                                                                           self.pad,
                                                                                           self.cut),
                                           validation_steps=self.val_samples_count / self.batch_size,
                                           steps_per_epoch=self.train_samples_count / self.batch_size,
                                           epochs=self.hyperparams.train_epoch_times, verbose=2,
                                           callbacks=[save_model, early_stopping])
        tools.show_save_record(history, train_start)

    def evaluate_generator(self):
        scores = self.model.evaluate_generator(generator=reader.generate_batch_data_file(self.test_fname,
                                                                                         self.tokenizer,
                                                                                         self.max_seq_len,
                                                                                         self.batch_size,
                                                                                         self.pad,
                                                                                         self.cut),
                                               steps=self.test_samples_count / self.batch_size)
        record_info = list()
        record_info.append("\n================== 性能评估 ==================\n")
        record_info.append("%s: %.4f\n" % (self.model.metrics_names[0], scores[0]))
        record_info.append("%s: %.2f%%\n" % (self.model.metrics_names[1], scores[1] * 100))
        record_str = ''.join(record_info)
        record_url = params.MODEL_SAVE_DIR + os.path.sep + params.TRAIN_RECORD_FNAME
        tools.print_save_str(record_str, record_url)

    def save(self, model_url):
        self.model.save(model_url)
        print("\n================== 保存模型 ==================")
        print(net_conf.RUN_WHICH_MODEL, 'has been saved in', model_url)

    def load(self, model_url):
        self.model = load_model(model_url)
        print("\n================== 加载模型 ==================")
        print('Model has been loaded from', model_url)

    def __call__(self, x):
        return self.model(x)


class AvgSeqDenseModel(BasicModel):
    """
    Do not use this model!
    There are some unknown errors with the Lamda layer, which makes model fail to save.
    """
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


class StackedBiLSTMDenseModel(BasicModel):
    def __init__(self):
        super(StackedBiLSTMDenseModel, self).__init__()

    def _do_build(self, src1_word_vec_seq, src2_word_vec_seq, src1_seq, src2_seq):
        p_dropout = self.hyperparams.p_dropout
        input_dropout = Dropout(p_dropout, name='input_dropout')
        src1_hidden_seq = input_dropout(src1_word_vec_seq)
        src2_hidden_seq = input_dropout(src2_word_vec_seq)

        # TODO 解决过拟合的问题
        bilstm_retseq_layer_num = self.hyperparams.bilstm_retseq_layer_num
        state_dim = self.hyperparams.state_dim
        for _ in range(bilstm_retseq_layer_num):
            this_bilstm = Bidirectional(LSTM(state_dim, return_sequences=True), merge_mode='concat')
            this_dropout = Dropout(p_dropout)
            src1_hidden_seq = this_bilstm(src1_hidden_seq)
            src2_hidden_seq = this_bilstm(src2_hidden_seq)
            src1_hidden_seq = this_dropout(src1_hidden_seq)
            src2_hidden_seq = this_dropout(src2_hidden_seq)

        enc_bilstm = Bidirectional(LSTM(state_dim), name='enc_bilstm')
        enc_dropout = Dropout(p_dropout, name='enc_dropout')
        src1_encoding = enc_bilstm(src1_hidden_seq)
        src2_encoding = enc_bilstm(src2_hidden_seq)
        src1_encoding = enc_dropout(src1_encoding)
        src2_encoding = enc_dropout(src2_encoding)

        merged_vec = keras.layers.concatenate([src1_encoding, src2_encoding], axis=-1)
        middle_vec = merged_vec

        dense_layer_num = self.hyperparams.dense_layer_num
        for _ in range(dense_layer_num):
            middle_vec = Dense(self.hyperparams.linear_unit_num, activation='relu')(middle_vec)
            middle_vec = Dropout(p_dropout)(middle_vec)

        preds = Dense(1, activation='sigmoid', name='logistic_output_layer')(middle_vec)
        return preds


class TransformerEncoderDenseModel(BasicModel):
    def __init__(self):
        super(TransformerEncoderDenseModel, self).__init__()

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
        pos_enc_layer = Embedding(len_limit+1, d_pos_enc, trainable=False,
                                  weights=[transformer.get_pos_enc_matrix(len_limit+1, d_pos_enc)],
                                  name='pos_enc_layer')
        transformer_encoder = transformer.Encoder(d_model, d_inner_hid, n_head, d_k, d_v,
                                                  layers_num=layers_num,
                                                  p_dropout=p_dropout,
                                                  pos_enc_layer=pos_enc_layer,
                                                  mode=self.hyperparams.transformer_mode,
                                                  batch_size=self.batch_size)
        get_pos_seq = Lambda(transformer.get_pos_seq)
        src1_pos = get_pos_seq(src1_seq)
        src2_pos = get_pos_seq(src2_seq)
        src1_seq_repr_seq = transformer_encoder(src1_word_vec_seq, src1_seq, src_pos=src1_pos)
        src2_seq_repr_seq = transformer_encoder(src2_word_vec_seq, src2_seq, src_pos=src2_pos)

        # mask操作，只对非占位符的部分求平均
        def masked_avg_emb(src_seq_repr_seq, src_seq):
            mask = K.cast(K.expand_dims(K.not_equal(src_seq, 0), -1), 'float32')
            src_seq_repr_seq = src_seq_repr_seq * mask
            src_seq_repr_seq = K.mean(src_seq_repr_seq, axis=1, keepdims=True)
            return K.reshape(src_seq_repr_seq, [-1, d_model])

        masked_avg_seq = Lambda(lambda x: masked_avg_emb(x[0], x[1]), name='masked_seq_avg')

        # TODO 感觉求平均有点问题，试试用LTSM编码
        # TODO 训练不收敛
        src1_encoding = masked_avg_seq([src1_seq_repr_seq, src1_seq])
        src2_encoding = masked_avg_seq([src2_seq_repr_seq, src2_seq])

        # input tensor => 一系列的Keras层 => output tensor
        # 如果使用了backend函数，例如K.concatenate()或tf.reduce_mean()等，需要使用Lambda层封装它们
        merged_vec = keras.layers.concatenate([src1_encoding, src2_encoding])
        middle_vec = merged_vec

        dense_layer_num = self.hyperparams.dense_layer_num
        dense_p_dropout = self.hyperparams.dense_p_dropout
        for _ in range(dense_layer_num):
            middle_vec = Dense(self.hyperparams.linear_unit_num, activation='relu')(middle_vec)
            middle_vec = Dropout(dense_p_dropout)(middle_vec)

        preds = Dense(1, activation='sigmoid', name='logistic_output_layer')(middle_vec)
        return preds
