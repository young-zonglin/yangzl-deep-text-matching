from keras.optimizers import Adam, RMSprop

from tools import LRSchedulerDoNothing
from transformer import LRSchedulerPerStep

# TODO 网格搜索
GRID_SEARCH = True

# AvgSeqDenseModel | StackedBiLSTMDenseModel |
# TransformerEncoderBiLSTMDenseModel | RNMTPlusEncoderBiLSTMDenseModel |
RUN_WHICH_MODEL = 'RNMTPlusEncoderBiLSTMDenseModel'

# en es
WHICH_LANGUAGE = 'en'

# hyper params
fastText_EN_WORD_VEC_DIM = 300
fastText_ES_WORD_VEC_DIM = 300


def get_hyperparams(model_name):
    if not model_name:
        return BasicParams()

    if model_name == "AvgSeqDenseModel":
        return AvgSeqDenseParams()
    elif model_name == 'StackedBiLSTMDenseModel':
        return StackedBiLSTMDenseParams()
    elif model_name == 'TransformerEncoderBiLSTMDenseModel':
        return TransformerEncoderBiLSTMDenseParams()
    elif model_name == 'RNMTPlusEncoderBiLSTMDenseModel':
        return RNMTPlusEncoderBiLSTMDenseParams()
    else:
        return BasicParams()


# TODO auto save hyperparams to file
class BasicParams:
    pad = 'pre'
    cut = 'pre'
    batch_size = 32  # 32 64 128 256


class AvgSeqDenseParams:
    dense_layer_num = 3
    linear_unit_num = 64

    optimizer = RMSprop()
    lr_scheduler = LRSchedulerDoNothing()

    pad = 'pre'
    cut = 'pre'

    p_dropout = 0.5
    early_stop_patience = 30
    early_stop_min_delta = 1e-4
    train_epoch_times = 1000
    batch_size = 32  # 32 64 128 256


class StackedBiLSTMDenseParams:
    bilstm_retseq_layer_num = 2
    state_dim = 100
    lstm_p_dropout = 0.5

    dense_layer_num = 2
    linear_unit_num = 128
    dense_p_dropout = 0.5

    optimizer = RMSprop()
    lr_scheduler = LRSchedulerDoNothing()

    pad = 'pre'
    cut = 'pre'

    early_stop_patience = 30
    early_stop_min_delta = 1e-4
    train_epoch_times = 1000
    # TODO 超参batch_size的设置
    # TODO 动态batch_size
    batch_size = 128  # 32 64 128 256


# The scale of the model and state vec dim should be proportional to the scale of the data
class RNMTPlusEncoderBiLSTMDenseParams:
    retseq_layer_num = 2
    state_dim = 100
    lstm_p_dropout = 0.5

    dense_layer_num = 2
    initial_unit_num = 128
    dense_p_dropout = 0.5

    optimizer = RMSprop()
    lr_scheduler = LRSchedulerDoNothing()

    pad = 'pre'
    cut = 'pre'

    early_stop_patience = 30
    early_stop_min_delta = 1e-4
    train_epoch_times = 1000
    batch_size = 128  # recommended by "Exploring the Limits of Language Modeling"


class TransformerEncoderBiLSTMDenseParams:
    transformer_mode = 0
    word_vec_dim = 300  # fastText pretrained word vec
    layers_num = 3
    d_model = word_vec_dim
    d_inner_hid = 512  # d_ff, follow "attention-is-all-you-need-keras"
    n_head = 5  # h head
    d_k = d_v = int(d_model/n_head)
    d_pos_enc = d_model
    p_dropout = 0.1  # follow origin paper

    state_dim = 100
    lstm_p_dropout = 0.5

    # Follow the practice of CNNs
    dense_layer_num = 2
    initial_unit_num = 128
    dense_p_dropout = 0.5

    optimizer = Adam(0.001, 0.9, 0.98, epsilon=1e-9)  # follow origin paper
    warmup_step = 6000  # in origin paper, this value is set to 4000
    lr_scheduler = LRSchedulerPerStep(d_model, warmup_step)

    pad = 'post'
    cut = 'post'

    early_stop_patience = 30
    early_stop_min_delta = 1e-4
    train_epoch_times = 1000
    batch_size = 64  # follow "attention-is-all-you-need-keras"
