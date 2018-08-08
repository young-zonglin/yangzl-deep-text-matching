"""
Find best hyper params of model by grid search.
Then, save them to hyper params configuration class.
"""
import net_conf
import tools
from models import BasicModel
from net_conf import available_models
from net_conf import model_name_addr_full


def tune_dropout_rate_SBLDModel(which_language):
    run_which_model = available_models[1]
    net_conf.RUN_WHICH_MODEL = run_which_model
    model_full_name = model_name_addr_full[run_which_model]
    print('============ ' + model_full_name + ' tune dropout rate ============')
    # RNMTPlusEncoderBiLSTMDenseModel | StackedBiLSTMDenseModel
    # Don't set dropout rate too large, because it will cause information loss.
    # According to previous experiment: lstm rate >= 0.5, 0 <= dense rate <= 0.2
    lstm_p_dropout = [0.5, 0.6, 0.7]
    dense_p_dropout = [0, 0.1, 0.2]
    for lstm_rate in lstm_p_dropout:
        for dense_rate in dense_p_dropout:
            text_match_model = BasicModel.make_model(run_which_model)
            hyperparams = net_conf.get_hyperparams(run_which_model)
            hyperparams.lstm_p_dropout = lstm_rate
            hyperparams.dense_p_dropout = dense_rate
            tools.train_model(text_match_model, hyperparams, which_language)


def tune_layer_num_SBLDModel(which_language):
    run_which_model = available_models[1]
    net_conf.RUN_WHICH_MODEL = run_which_model
    model_full_name = model_name_addr_full[run_which_model]
    print('============ ' + model_full_name + ' tune layer num ============')
    # RNMTPlusEncoderBiLSTMDenseModel | StackedBiLSTMDenseModel
    layer_num = [0, 1, 2]
    for num in layer_num:
        text_match_model = BasicModel.make_model(run_which_model)
        hyperparams = net_conf.get_hyperparams(run_which_model)
        hyperparams.bilstm_retseq_layer_num = num
        tools.train_model(text_match_model, hyperparams, which_language)


def tune_l2_lambda_SBLDModel(which_language):
    run_which_model = available_models[1]
    net_conf.RUN_WHICH_MODEL = run_which_model
    model_full_name = model_name_addr_full[run_which_model]
    print('============ ' + model_full_name + ' tune l2 lambda ============')
    l2_lambda = [1, 0.1, 0.01, 0.001]
    for num in l2_lambda:
        text_match_model = BasicModel.make_model(run_which_model)
        hyperparams = net_conf.get_hyperparams(run_which_model)
        hyperparams.l2_lambda = num
        tools.train_model(text_match_model, hyperparams, which_language)


if __name__ == '__main__':
    which_language = net_conf.WHICH_LANGUAGE
    # tune_dropout_rate_SBLDModel(which_language)
    # tune_layer_num_SBLDModel(which_language)
    tune_l2_lambda_SBLDModel(which_language)
