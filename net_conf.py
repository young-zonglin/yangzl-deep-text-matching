from keras.optimizers import Adam, RMSprop

from tools import LRSchedulerDoNothing
from transformer import LRSchedulerPerStep

# TODO 网格搜索
GRID_SEARCH = True

# AvgSeqDenseModel | StackedBiLSTMDenseModel |
# TransformerEncoderDenseModel | TransformerDenseModelTest |
RUN_WHICH_MODEL = 'TransformerEncoderDenseModel'

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
    elif model_name == 'TransformerEncoderDenseModel':
        return TransformerEncoderDenseParams()
    elif model_name == 'TransformerDenseModelTest':
        return TransformerDenseParamsTest()
    else:
        return BasicParams()


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
    state_dim = 50
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
    # TODO 超参batch_size的设置
    # TODO 动态batch_size
    batch_size = 512  # 32 64 128 256


class TransformerDenseParamsTest:
    transformer_mode = 0
    word_vec_dim = 300
    layers_num = 1
    d_model = word_vec_dim
    d_inner_hid = 32  # d_ff
    n_head = 5  # h head
    d_k = d_v = int(d_model/n_head)
    d_pos_enc = d_model
    p_dropout = 0.1

    dense_layer_num = 1
    linear_unit_num = 32
    dense_p_dropout = 0.5

    optimizer = Adam(0.001, 0.9, 0.98, epsilon=1e-9)
    lr_scheduler = LRSchedulerPerStep(d_model, 4000)

    pad = 'post'
    cut = 'post'

    early_stop_patience = 1
    early_stop_min_delta = 1e-4
    train_epoch_times = 5
    batch_size = 64  # 32 64 128 256


class TransformerEncoderDenseParams:
    transformer_mode = 0
    word_vec_dim = 300
    layers_num = 6
    d_model = word_vec_dim
    d_inner_hid = 512  # d_ff
    n_head = 5  # h head
    d_k = d_v = int(d_model/n_head)
    d_pos_enc = d_model
    p_dropout = 0.1

    bilstm_retseq_layer_num = 1
    state_dim = 50
    lstm_p_dropout = 0.5

    dense_layer_num = 1
    linear_unit_num = 64
    dense_p_dropout = 0.5

    optimizer = Adam(0.001, 0.9, 0.98, epsilon=1e-9)
    lr_scheduler = LRSchedulerPerStep(d_model, 4000)

    pad = 'post'
    cut = 'post'

    early_stop_patience = 30
    early_stop_min_delta = 1e-4
    train_epoch_times = 1000
    batch_size = 64  # 32 64 128 256
