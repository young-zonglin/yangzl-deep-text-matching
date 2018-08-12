"""
Find best hyper params of model by grid search.
Then, save them to hyper params configuration class.
"""
from configs import net_conf, params
from configs.net_conf import available_models, model_name_abbr_full
from configs.params import available_datasets
from models.model_factory import ModelFactory
from utils import tools


def tune_dropout_rate_SBLDModel():
    model_name = available_models[1]
    model_full_name = model_name_abbr_full[model_name]
    print('============ ' + model_full_name + ' tune dropout rate ============')
    # RNMTPlusEncoderBiLSTMDenseModel | StackedBiLSTMDenseModel
    # Don't set dropout rate too large, because it will cause information loss.
    # According to previous experiment: lstm rate >= 0.5, 0 <= dense rate <= 0.2
    lstm_p_dropouts = [0.5, 0.6, 0.7]
    dense_p_dropouts = [0, 0.1, 0.2]
    for lstm_rate in lstm_p_dropouts:
        for dense_rate in dense_p_dropouts:
            text_match_model = ModelFactory.make_model(model_name)
            hyperparams = net_conf.get_hyperparams(model_name)
            hyperparams.lstm_p_dropout = lstm_rate
            hyperparams.dense_p_dropout = dense_rate
            dataset_name = available_datasets[0]
            dataset_params = params.get_dataset_params(dataset_name)
            tools.train_model(text_match_model, hyperparams, dataset_params)


def tune_layer_num_SBLDModel():
    run_this_model = available_models[1]
    model_full_name = model_name_abbr_full[run_this_model]
    print('============ ' + model_full_name + ' tune layer num ============')
    # RNMTPlusEncoderBiLSTMDenseModel | StackedBiLSTMDenseModel
    layer_nums = [0, 1, 2]
    for num in layer_nums:
        text_match_model = ModelFactory.make_model(run_this_model)
        hyperparams = net_conf.get_hyperparams(run_this_model)
        hyperparams.bilstm_retseq_layer_num = num
        dataset_name = available_datasets[0]
        dataset_params = params.get_dataset_params(dataset_name)
        tools.train_model(text_match_model, hyperparams, dataset_params)


def tune_l2_lambda_SBLDModel():
    run_this_model = available_models[1]
    model_full_name = model_name_abbr_full[run_this_model]
    print('============ ' + model_full_name + ' tune l2 lambda ============')
    kernel_l2_lambdas = [1e-5, 1e-4]
    recurrent_l2_lambdas = [1e-5, 1e-4]
    bias_l2_lambdas = [1e-5, 1e-4]
    activity_l2_lambdas = [0, 1e-5, 1e-4]
    for kernel_l2_lambda in kernel_l2_lambdas:
        for recurrent_l2_lambda in recurrent_l2_lambdas:
            for bias_l2_lambda in bias_l2_lambdas:
                for activity_l2_lambda in activity_l2_lambdas:
                    text_match_model = ModelFactory.make_model(run_this_model)
                    hyperparams = net_conf.get_hyperparams(run_this_model)
                    hyperparams.kernel_l2_lambda = kernel_l2_lambda
                    hyperparams.recurrent_l2_lambda = recurrent_l2_lambda
                    hyperparams.bias_l2_lambda = bias_l2_lambda
                    hyperparams.activity_l2_lambda = activity_l2_lambda
                    dataset_name = available_datasets[0]
                    dataset_params = params.get_dataset_params(dataset_name)
                    tools.train_model(text_match_model, hyperparams, dataset_params)


def tune_state_dim_SBLDModel():
    run_this_model = available_models[1]
    model_full_name = model_name_abbr_full[run_this_model]
    print('============ ' + model_full_name + ' tune hidden state dim num ============')
    # RNMTPlusEncoderBiLSTMDenseModel | StackedBiLSTMDenseModel
    state_dims = [150, 200, 250, 300]
    for state_dim in state_dims:
        text_match_model = ModelFactory.make_model(run_this_model)
        hyperparams = net_conf.get_hyperparams(run_this_model)
        hyperparams.state_dim = state_dim
        dataset_name = available_datasets[0]
        dataset_params = params.get_dataset_params(dataset_name)
        tools.train_model(text_match_model, hyperparams, dataset_params)


def tune_dropout_rate_REBLDModel():
    model_name = available_models[3]
    model_full_name = model_name_abbr_full[model_name]
    print('============ ' + model_full_name + ' tune dropout rate ============')
    lstm_p_dropouts = [0.2, 0.3, 0.4, 0.5]
    dense_p_dropouts = [0.2, 0.3, 0.4, 0.5]
    for lstm_rate in lstm_p_dropouts:
        for dense_rate in dense_p_dropouts:
            text_match_model = ModelFactory.make_model(model_name)
            hyperparams = net_conf.get_hyperparams(model_name)
            hyperparams.lstm_p_dropout = lstm_rate
            hyperparams.dense_p_dropout = dense_rate
            dataset_name = available_datasets[0]
            dataset_params = params.get_dataset_params(dataset_name)
            tools.train_model(text_match_model, hyperparams, dataset_params)


if __name__ == '__main__':
    # tune_dropout_rate_SBLDModel()
    # tune_layer_num_SBLDModel()
    # tune_l2_lambda_SBLDModel()
    # tune_state_dim_SBLDModel()
    tune_dropout_rate_REBLDModel()
