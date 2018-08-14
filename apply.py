from configs import params, net_conf
from configs.net_conf import available_models
from configs.params import available_datasets
from models.model_factory import ModelFactory


def apply():
    model_name = available_models[1]
    text_match_model = ModelFactory.make_model(model_name)
    model_url = ''
    text_match_model.load(model_url)
    hyperparams = net_conf.get_hyperparams(model_name)
    dataset_name = available_datasets[0]
    dataset_params = params.get_dataset_params(dataset_name)

    text_match_model.setup(hyperparams, dataset_params)
    text_match_model.evaluate_generator()


if __name__ == '__main__':
    apply()
