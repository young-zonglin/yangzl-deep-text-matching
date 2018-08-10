"""
Find best hyper params of model by grid search.
Then, save them to hyper params configuration class.
"""
from configs import net_conf
from configs.net_conf import available_models
from configs.net_conf import model_name_addr_full
from models.model_factory import ModelFactory
from utils import tools


def tune_dropout_rate_SBLDModel(which_language):
    run_which_model = available_models[1]
    net_conf.RUN_WHICH_MODEL = run_which_model
    model_full_name = model_name_addr_full[run_which_model]
    print('============ ' + model_full_name + ' tune dropout rate ============')
    # RNMTPlusEncoderBiLSTMDenseModel | StackedBiLSTMDenseModel
    # Don't set dropout rate too large, because it will cause information loss.
    # According to previous experiment: lstm rate >= 0.5, 0 <= dense rate <= 0.2
    lstm_p_dropouts = [0.5, 0.6, 0.7]
    dense_p_dropouts = [0, 0.1, 0.2]
    for lstm_rate in lstm_p_dropouts:
        for dense_rate in dense_p_dropouts:
            text_match_model = ModelFactory.make_model(run_which_model)
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
    layer_nums = [0, 1, 2]
    for num in layer_nums:
        text_match_model = ModelFactory.make_model(run_which_model)
        hyperparams = net_conf.get_hyperparams(run_which_model)
        hyperparams.bilstm_retseq_layer_num = num
        tools.train_model(text_match_model, hyperparams, which_language)


def tune_l2_lambda_SBLDModel(which_language):
    run_which_model = available_models[1]
    net_conf.RUN_WHICH_MODEL = run_which_model
    model_full_name = model_name_addr_full[run_which_model]
    print('============ ' + model_full_name + ' tune l2 lambda ============')
    kernel_l2_lambdas = [1e-5, 1e-4]
    recurrent_l2_lambdas = [1e-5, 1e-4]
    bias_l2_lambdas = [1e-5, 1e-4]
    activity_l2_lambdas = [0, 1e-5, 1e-4]
    for kernel_l2_lambda in kernel_l2_lambdas:
        for recurrent_l2_lambda in recurrent_l2_lambdas:
            for bias_l2_lambda in bias_l2_lambdas:
                for activity_l2_lambda in activity_l2_lambdas:
                    text_match_model = ModelFactory.make_model(run_which_model)
                    hyperparams = net_conf.get_hyperparams(run_which_model)
                    hyperparams.kernel_l2_lambda = kernel_l2_lambda
                    hyperparams.recurrent_l2_lambda = recurrent_l2_lambda
                    hyperparams.bias_l2_lambda = bias_l2_lambda
                    hyperparams.activity_l2_lambda = activity_l2_lambda
                    tools.train_model(text_match_model, hyperparams, which_language)


if __name__ == '__main__':
    which_language = net_conf.WHICH_LANGUAGE
    # tune_dropout_rate_SBLDModel(which_language)
    # tune_layer_num_SBLDModel(which_language)
    tune_l2_lambda_SBLDModel(which_language)
