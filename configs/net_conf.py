from keras.callbacks import Callback
from keras.optimizers import Adam, RMSprop

from layers import transformer

model_name_abbr_full = {'ASDModel': 'AvgSeqDenseModel',
                        'SBLDModel': 'StackedBiLSTMDenseModel',
                        'TEBLDModel': 'TransformerEncoderBiLSTMDenseModel',
                        'REBLDModel': "RNMTPlusEncoderBiLSTMDenseModel"}
model_name_full_abbr = {v: k for k, v in model_name_abbr_full.items()}
available_models = ['ASDModel', 'SBLDModel', 'TEBLDModel', 'REBLDModel']


def get_hyperparams(model_name):
    if not model_name:
        return BasicHParams()

    if model_name == available_models[0]:
        return AvgSeqDenseHParams()
    elif model_name == available_models[1]:
        return StackedBiLSTMDenseHParams()
    elif model_name == available_models[2]:
        return TransformerEncoderBiLSTMDenseHParams()
    elif model_name == available_models[3]:
        return RNMTPlusEncoderBiLSTMDenseHParams()
    else:
        return BasicHParams()


# hyper params
fastText_EN_WORD_VEC_DIM = 300
fastText_ES_WORD_VEC_DIM = 300


# Avoid crossing import between modules.
# Definition need before calling it.
# Calling in a function/method does not require prior definition.
# Class properties will be initialized without instantiation.
class LRSchedulerDoNothing(Callback):
    def __init__(self):
        super(LRSchedulerDoNothing, self).__init__()


# return hyper params string => done
class BasicHParams:
    def __init__(self):
        self.current_classname = self.__class__.__name__

        self.pad = 'pre'
        self.cut = 'pre'

        self.batch_size = 32  # 32 64 128 256

    def __str__(self):
        ret_info = list()
        ret_info.append("pad: " + self.pad + '\n')
        ret_info.append("cut: " + self.cut + '\n\n')

        ret_info.append('batch size: '+str(self.batch_size)+'\n')
        return ''.join(ret_info)


class TrainHParams(BasicHParams):
    def __init__(self):
        BasicHParams.__init__(self)

        self.optimizer = Adam()
        self.lr_scheduler = LRSchedulerDoNothing()

        self.kernel_l2_lambda = 0
        self.recurrent_l2_lambda = 0
        self.bias_l2_lambda = 0
        self.activity_l2_lambda = 0

        self.early_stop_monitor = 'val_loss'
        self.early_stop_mode = 'auto'
        # 10 times waiting is not enough.
        # Maybe 20 is a good value.
        self.early_stop_patience = 20
        self.early_stop_min_delta = 1e-4

        self.train_epoch_times = 1000

    def __str__(self):
        ret_info = list()
        ret_info.append('optimizer: ' + str(self.optimizer) + '\n')
        ret_info.append('lr scheduler: ' + str(self.lr_scheduler) + '\n\n')

        ret_info.append('kernel l2 lambda: ' + str(self.kernel_l2_lambda) + '\n')
        ret_info.append('recurrent l2 lambda: ' + str(self.recurrent_l2_lambda) + '\n')
        ret_info.append('bias l2 lambda: ' + str(self.bias_l2_lambda) + '\n')
        ret_info.append('activity l2 lambda: ' + str(self.activity_l2_lambda) + '\n\n')

        ret_info.append('early stop monitor: ' + str(self.early_stop_monitor) + '\n')
        ret_info.append('early stop mode: ' + str(self.early_stop_mode) + '\n')
        ret_info.append('early stop patience: ' + str(self.early_stop_patience) + '\n')
        ret_info.append('early stop min delta: ' + str(self.early_stop_min_delta) + '\n\n')

        ret_info.append('train epoch times: ' + str(self.train_epoch_times) + '\n\n')

        super_str = super(TrainHParams, self).__str__()
        return ''.join(ret_info)+super_str


class AvgSeqDenseHParams(TrainHParams):
    def __init__(self):
        super(AvgSeqDenseHParams, self).__init__()
        self.dense_layer_num = 3
        self.linear_unit_num = 64

        self.p_dropout = 0.5

        self.optimizer = RMSprop()
        self.lr_scheduler = LRSchedulerDoNothing()

    def __str__(self):
        ret_info = list()
        ret_info.append('\n================== '+self.current_classname+' ==================\n')
        ret_info.append('dense layer num: '+str(self.dense_layer_num)+'\n')
        ret_info.append('linear unit num: ' + str(self.linear_unit_num) + '\n\n')

        ret_info.append('dropout probability: ' + str(self.p_dropout) + '\n\n')

        super_str = super(AvgSeqDenseHParams, self).__str__()
        return ''.join(ret_info) + super_str


class StackedBiLSTMDenseHParams(TrainHParams):
    def __init__(self):
        super(StackedBiLSTMDenseHParams, self).__init__()
        self.bilstm_retseq_layer_num = 2  # best value according to experiment
        self.state_dim = 100
        # dropout rate up, overfitting down
        # 0.4 or 0.5 is a good value
        # Information will be lost as the rate continue to increase.
        self.lstm_p_dropout = 0.5

        # Excessive regularization means high bias.
        # Insufficient regularization means no improvement in overfitting.
        # The following configuration of hyper params are good values.
        self.kernel_l2_lambda = 1e-5
        self.recurrent_l2_lambda = 1e-5
        self.bias_l2_lambda = 1e-5
        self.activity_l2_lambda = 0

        self.unit_reduce = False
        self.dense_layer_num = 2
        self.linear_unit_num = 128
        self.dense_p_dropout = 0.4

        self.optimizer = RMSprop()
        self.lr_scheduler = LRSchedulerDoNothing()

        self.early_stop_monitor = 'val_acc'
        self.early_stop_patience = 30
        self.early_stop_min_delta = 0

        # Set the value of hyper params batch_size => done
        # See my evernote for more info.
        # There is no need to adjust batch_size dynamically.
        self.batch_size = 128  # 32 64 128 256

    def __str__(self):
        ret_info = list()
        ret_info.append('\n================== '+self.current_classname+' ==================\n')
        ret_info.append('bi-lstm retseq layer num: ' + str(self.bilstm_retseq_layer_num) + '\n')
        ret_info.append('state dim: ' + str(self.state_dim) + '\n')
        ret_info.append('lstm dropout proba: ' + str(self.lstm_p_dropout) + '\n\n')

        ret_info.append('unit reduce: ' + str(self.unit_reduce) + '\n')
        ret_info.append('dense layer num: ' + str(self.dense_layer_num) + '\n')
        ret_info.append('linear unit num: ' + str(self.linear_unit_num) + '\n')
        ret_info.append('dense dropout proba: ' + str(self.dense_p_dropout) + '\n\n')

        super_str = super(StackedBiLSTMDenseHParams, self).__str__()
        return ''.join(ret_info) + super_str


# The scale of the model and state vec dim should be proportional to the scale of the data.
class RNMTPlusEncoderBiLSTMDenseHParams(TrainHParams):
    def __init__(self):
        super(RNMTPlusEncoderBiLSTMDenseHParams, self).__init__()
        self.retseq_layer_num = 2
        self.state_dim = 100
        # Since layer norm also has a regularization effect
        self.lstm_p_dropout = 0.1

        self.unit_reduce = False
        self.dense_layer_num = 2
        self.initial_unit_num = 128
        self.dense_p_dropout = 0.5

        # follow origin paper
        self.kernel_l2_lambda = 1e-5
        self.recurrent_l2_lambda = 1e-5
        self.bias_l2_lambda = 1e-5
        self.activity_l2_lambda = 0

        self.lr = 0.001
        self.beta_1 = 0.9
        self.beta_2 = 0.999
        self.eps = 1e-6
        self.optimizer = Adam(self.lr, self.beta_1, self.beta_2, epsilon=self.eps)  # follow origin paper
        # should use RNMT+ lr scheduler here
        self.lr_scheduler = LRSchedulerDoNothing()

        self.early_stop_monitor = 'val_acc'

        self.batch_size = 128  # Recommended by "Exploring the Limits of Language Modeling".

    def __str__(self):
        ret_info = list()
        ret_info.append('\n================== '+self.current_classname+' ==================\n')
        ret_info.append('ret seq layer num: ' + str(self.retseq_layer_num) + '\n')
        ret_info.append('state dim: ' + str(self.state_dim) + '\n')
        ret_info.append('lstm dropout proba: ' + str(self.lstm_p_dropout) + '\n\n')

        ret_info.append('unit reduce: ' + str(self.unit_reduce) + '\n')
        ret_info.append('dense layer num: ' + str(self.dense_layer_num) + '\n')
        ret_info.append('initial unit num: ' + str(self.initial_unit_num) + '\n')
        ret_info.append('dense dropout proba: ' + str(self.dense_p_dropout) + '\n\n')

        ret_info.append('lr: ' + str(self.lr) + '\n')
        ret_info.append('beta_1: ' + str(self.beta_1) + '\n')
        ret_info.append('beta_2: ' + str(self.beta_2) + '\n')
        ret_info.append('epsilon: ' + str(self.eps) + '\n')

        super_str = super(RNMTPlusEncoderBiLSTMDenseHParams, self).__str__()
        return ''.join(ret_info) + super_str


class TransformerEncoderBiLSTMDenseHParams(TrainHParams):
    def __init__(self):
        super(TransformerEncoderBiLSTMDenseHParams, self).__init__()
        self.transformer_mode = 0
        self.word_vec_dim = 300  # fastText pretrained word vec
        self.layers_num = 3
        self.d_model = self.word_vec_dim
        self.d_inner_hid = 512  # d_ff, follow "attention-is-all-you-need-keras"
        self.n_head = 5  # h head
        self.d_k = self.d_v = int(self.d_model/self.n_head)
        self.d_pos_enc = self.d_model
        self.p_dropout = 0.1  # follow origin paper

        self.state_dim = 100
        self.lstm_p_dropout = 0.5

        # Follow the practice of CNNs
        self.unit_reduce = False
        self.dense_layer_num = 2
        self.initial_unit_num = 128
        self.dense_p_dropout = 0.5

        self.lr = 0.001
        self.beta_1 = 0.9
        self.beta_2 = 0.98
        self.eps = 1e-9
        self.optimizer = Adam(self.lr, self.beta_1, self.beta_2, epsilon=self.eps)  # follow origin paper
        self.warmup_step = 4000  # in origin paper, this value is set to 4000
        self.lr_scheduler = transformer.LRSchedulerPerStep(self.d_model, self.warmup_step)

        self.pad = 'post'
        self.cut = 'post'

        self.early_stop_monitor = 'val_acc'

        self.batch_size = 64  # follow "attention-is-all-you-need-keras"

    def __str__(self):
        ret_info = list()
        ret_info.append('\n================== '+self.current_classname+' ==================\n')
        ret_info.append('transformer mode: ' + str(self.transformer_mode) + '\n')
        ret_info.append('word vec dim: ' + str(self.word_vec_dim) + '\n')
        ret_info.append('encoder layer num: ' + str(self.layers_num) + '\n')
        ret_info.append('d_model: ' + str(self.d_model) + '\n')
        ret_info.append('dim of inner hid: ' + str(self.d_inner_hid) + '\n')
        ret_info.append('n head: ' + str(self.n_head) + '\n')
        ret_info.append('dim of k: ' + str(self.d_k) + '\n')
        ret_info.append('dim of v: ' + str(self.d_v) + '\n')
        ret_info.append('pos enc dim: ' + str(self.d_pos_enc) + '\n')
        ret_info.append('dropout proba: ' + str(self.p_dropout) + '\n\n')

        ret_info.append('state dim: ' + str(self.state_dim) + '\n')
        ret_info.append('lstm dropout proba: ' + str(self.lstm_p_dropout) + '\n\n')

        ret_info.append('unit reduce: ' + str(self.unit_reduce) + '\n')
        ret_info.append('dense layer num: ' + str(self.dense_layer_num) + '\n')
        ret_info.append('initial unit num: ' + str(self.initial_unit_num) + '\n')
        ret_info.append('dense dropout proba: ' + str(self.dense_p_dropout) + '\n\n')

        ret_info.append('lr: ' + str(self.lr) + '\n')
        ret_info.append('beta_1: ' + str(self.beta_1) + '\n')
        ret_info.append('beta_2: ' + str(self.beta_2) + '\n')
        ret_info.append('epsilon: ' + str(self.eps) + '\n')
        ret_info.append('warm up step: ' + str(self.warmup_step) + '\n')

        super_str = super(TransformerEncoderBiLSTMDenseHParams, self).__str__()
        return ''.join(ret_info) + super_str


if __name__ == '__main__':
    print(BasicHParams())
    print(TrainHParams())
    print(AvgSeqDenseHParams())
    print(StackedBiLSTMDenseHParams())
    print(RNMTPlusEncoderBiLSTMDenseHParams())
    print(TransformerEncoderBiLSTMDenseHParams())
