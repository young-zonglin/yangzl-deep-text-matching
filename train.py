from configs import net_conf, params
from configs.net_conf import available_models
from configs.params import available_datasets
from models.model_factory import ModelFactory
from utils import tools


def train():
    run_this_model = available_models[4]
    text_match_model = ModelFactory.make_model(run_this_model)
    hyperparams = net_conf.get_hyperparams(run_this_model)
    dataset_name = available_datasets[0]
    dataset_params = params.get_dataset_params(dataset_name)
    tools.train_model(text_match_model, hyperparams, dataset_params)


if __name__ == '__main__':
    train()
