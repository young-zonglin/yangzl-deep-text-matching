from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.layers import Input, Dense, Dropout, Embedding, Lambda, Reshape
from keras.layers import concatenate
from keras.models import Model
from keras.models import load_model

import net_conf
import reader
import tools


class BasicModel:
    def __init__(self):
        self.max_seq_len = None
        self.vocab_size = None
        self.word_vec_dim = None
        self.batch_samples_number = None

        self.pretrained_word_vecs_fname = None
        self.raw_fname = None
        self.train_fname = None
        self.val_fname = None
        self.test_fname = None

        self.model = None
        self.embedding_matrix = None
        self.tokenizer = None

        self.total_samples_count = 0
        self.train_samples_count = 0
        self.val_samples_count = 0
        self.test_samples_count = 0

    # 静态工厂方法
    @staticmethod
    def make_model(model_name):
        if model_name == "AvgSeqDenseModel":
            return AvgSeqDenseModel()
        else:
            pass

    def setup(self, raw_fname, train_fname, val_fname, test_fname,
              pretrained_word_vecs_fname):
        self.pretrained_word_vecs_fname = pretrained_word_vecs_fname
        reader.split_train_val_test(raw_fname, train_fname, val_fname, test_fname)
        self.raw_fname = raw_fname
        self.train_fname = train_fname
        self.val_fname = val_fname
        self.test_fname = test_fname

        self.batch_samples_number = net_conf.BATCH_SAMPLES_NUMBER
        self.tokenizer = reader.fit_tokenizer(self.raw_fname)
        self.vocab_size = len(self.tokenizer.word_index)

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

        print('================ In setup ================')
        print('Vocab size: %d' % self.vocab_size)
        print('Max sentence length: {}'.format(self.max_seq_len))
        print('Total samples count: %d' % self.total_samples_count)
        print('Train samples count: %d' % self.train_samples_count)
        print('Val samples count: %d' % self.val_samples_count)
        print('Test samples count: %d' % self.test_samples_count)

    def _do_build(self, word_vec_seq1, word_vec_seq2):
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
                              trainable=False)
        word_vec_seq1 = embedding(source1)
        word_vec_seq2 = embedding(source2)
        # print(embedding.get_input_shape_at(0))
        # print(embedding.get_output_shape_at(1))

        preds = self._do_build(word_vec_seq1, word_vec_seq2)
        self.model = Model(inputs=[source1, source2], outputs=preds)

        print('================ In build ================')
        print('Found %d word vectors.' % len(word2vec))
        print('\n############### Model summary ##################')
        print(self.model.summary())
        return self.model

    def compile(self):
        self.model.compile(loss='binary_crossentropy',
                           optimizer='rmsprop',
                           metrics=['accuracy'])

    def fit_generator(self):
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, min_delta=0.0001,
                                       verbose=1, mode='min')
        history = self.model.fit_generator(reader.generate_batch_data_file(self.train_fname,
                                                                           self.tokenizer,
                                                                           self.max_seq_len),
                                           validation_data=reader.generate_batch_data_file(self.val_fname,
                                                                                           self.tokenizer,
                                                                                           self.max_seq_len),
                                           validation_steps=self.val_samples_count / self.batch_samples_number,
                                           steps_per_epoch=self.train_samples_count / self.batch_samples_number,
                                           epochs=net_conf.TRAIN_EPOCH_TIMES, verbose=1,
                                           callbacks=[early_stopping])
        print('\n========================== history ===========================')
        acc = history.history.get('acc')
        loss = history.history['loss']
        val_acc = history.history['val_acc']
        val_loss = history.history['val_loss']
        print('train acc:', acc)
        print('train loss', loss)
        print('val acc', val_acc)
        print('val loss', val_loss)
        print('\n======================= acc & loss & val_acc & val_loss ============================')
        for i in range(len(acc)):
            print('epoch {0:<4} | acc: {1:6.3f}% | loss: {2:<10.5f} |'
                  ' val_acc: {3:6.3f}% | val_loss: {4:<10.5f}'.format(i + 1,
                                                                      acc[i] * 100, loss[i],
                                                                      val_acc[i] * 100, val_loss[i]))
        # 训练完毕后，将每轮迭代的acc、loss、val_acc、val_loss以画图的形式进行展示 => done
        plt_x = [x+1 for x in range(len(acc))]
        plt_acc = plt_x, acc
        plt_loss = plt_x, loss
        plt_val_acc = plt_x, val_acc
        plt_val_loss = plt_x, val_loss
        tools.plot_figure('acc & loss & val_acc & val_loss',
                          plt_acc, plt_loss, plt_val_acc, plt_val_loss)

    def evaluate_generator(self):
        scores = self.model.evaluate_generator(generator=reader.generate_batch_data_file(self.test_fname,
                                                                                         self.tokenizer,
                                                                                         self.max_seq_len),
                                               steps=self.test_samples_count / self.batch_samples_number)
        print("\n==================================\n性能评估：")
        print("%s: %.4f" % (self.model.metrics_names[0], scores[0]))
        print("%s: %.2f%%" % (self.model.metrics_names[1], scores[1] * 100))

    def save(self, model_url):
        self.model.save(model_url)

    def load(self, model_url):
        self.model = load_model(model_url)

    def __call__(self, X):
        return self.model(X)


class AvgSeqDenseModel(BasicModel):
    def __init__(self):
        super(AvgSeqDenseModel, self).__init__()

    def _do_build(self, word_vec_seq1, word_vec_seq2):

        def avg_embedding(X):
            X = K.mean(X, axis=1)
            return Reshape([self.word_vec_dim])(X)

        def avg_embedding_output_shape(input_shape):
            ret_shape = self.batch_samples_number, input_shape[2]
            return tuple(ret_shape)

        avg_seq = Lambda(function=avg_embedding,
                         output_shape=avg_embedding_output_shape)

        seq1_encoding = avg_seq(word_vec_seq1)
        seq2_encoding = avg_seq(word_vec_seq2)

        merged_vec = concatenate([seq1_encoding, seq2_encoding], axis=-1)
        middle_vec = Dropout(net_conf.DROPOUT_RATE)(merged_vec)
        middle_vec = Dense(64, activation='relu')(middle_vec)
        middle_vec = Dropout(net_conf.DROPOUT_RATE)(middle_vec)
        middle_vec = Dense(64, activation='relu')(middle_vec)
        middle_vec = Dropout(net_conf.DROPOUT_RATE)(middle_vec)
        middle_vec = Dense(64, activation='relu')(middle_vec)
        middle_vec = Dropout(net_conf.DROPOUT_RATE)(middle_vec)
        preds = Dense(1, activation='sigmoid')(middle_vec)
        return preds
