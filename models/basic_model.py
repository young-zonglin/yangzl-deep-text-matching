import os
import time

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Input, Embedding
from keras.models import Model
from keras.models import load_model

from configs import params, net_conf
from configs.net_conf import model_name_full_abbr
from configs.params import dataset_name_full_abbr
from utils import tools, reader

# TensorFlow显存管理
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))


# batch size和seq len随意，word vec dim训练和应用时应一致
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

    def setup(self, hyperparams, dataset_params):
        self.pretrained_word_vecs_fname = dataset_params.pretrained_word_vecs_url
        self.raw_fname = dataset_params.raw_url
        self.train_fname = dataset_params.train_url
        self.val_fname = dataset_params.val_url
        self.test_fname = dataset_params.test_url
        reader.split_train_val_test(self.raw_fname,
                                    self.train_fname, self.val_fname, self.test_fname)

        run_which_model = model_name_full_abbr[self.__class__.__name__]
        dataset_name = dataset_name_full_abbr[dataset_params.__class__.__name__]
        setup_time = tools.get_current_time()
        self.this_model_save_dir = \
            params.RESULT_SAVE_DIR + os.path.sep + \
            run_which_model + '_' + dataset_name + '_' + setup_time
        if not os.path.exists(self.this_model_save_dir):
            os.makedirs(self.this_model_save_dir)

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
        record_url = self.this_model_save_dir + os.path.sep + params.TRAIN_RECORD_FNAME
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
        record_info.append('Found %d word vectors.\n' % len(word2vec))
        record_info.append(str(self.hyperparams))
        record_str = ''.join(record_info)
        record_url = self.this_model_save_dir + os.path.sep + params.TRAIN_RECORD_FNAME
        tools.print_save_str(record_str, record_url)
        print('\n############### Model summary ##################')
        self.model.summary()

        return self.model

    # TODO 优化算法
    # 动态学习率 => done，在回调中更改学习率
    def compile(self):
        self.model.compile(loss='binary_crossentropy',
                           optimizer=self.hyperparams.optimizer,
                           metrics=['accuracy'])

        # Transformer-based model的图太复杂太乱，没有看的必要
        # 不要在IDE中打开，否则会直接OOM
        # model_vis_url = self.this_model_save_dir + os.path.sep + params.MODEL_VIS_FNAME
        # plot_model(self.model, to_file=model_vis_url, show_shapes=True, show_layer_names=True)

    def fit_generator(self):
        train_start = float(time.time())
        early_stopping = EarlyStopping(monitor=self.hyperparams.early_stop_monitor,
                                       patience=self.hyperparams.early_stop_patience,
                                       min_delta=self.hyperparams.early_stop_min_delta,
                                       mode=self.hyperparams.early_stop_mode,
                                       verbose=1)
        # callback_instance.set_model(self.model) => set_model方法由Keras调用
        lr_scheduler = self.hyperparams.lr_scheduler
        save_url = \
            self.this_model_save_dir + os.path.sep + \
            'epoch_{epoch:03d}-{'+self.hyperparams.early_stop_monitor+':.4f}' + '.h5'
        model_saver = ModelCheckpoint(save_url,
                                      monitor=self.hyperparams.early_stop_monitor,
                                      mode=self.hyperparams.early_stop_mode,
                                      save_best_only=True, save_weights_only=False, verbose=1)
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
                                           callbacks=[model_saver, lr_scheduler, early_stopping])
        tools.show_save_record(self.this_model_save_dir, history, train_start)

    # TODO 评价指标
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
        record_url = self.this_model_save_dir + os.path.sep + params.TRAIN_RECORD_FNAME
        tools.print_save_str(record_str, record_url)

    def save(self, model_url):
        self.model.save(model_url)
        print("\n================== 保存模型 ==================")
        print(self.__class__.__name__, 'has been saved in', model_url)

    def load(self, model_url):
        self.model = load_model(model_url)
        print("\n================== 加载模型 ==================")
        print('Model has been loaded from', model_url)

    def __call__(self, x):
        return self.model(x)


# 学习Keras的激活层
# middle_vec = Dense(self.hyperparams.linear_unit_num, activation='relu')(middle_vec)
# middle_vec = Activation('relu')(middle_vec)
# middle_vec = Activation(keras.activations.relu)(middle_vec)
# middle_vec = Activation(K.relu)(middle_vec)
# middle_vec = keras.layers.ReLU()(middle_vec)
