from configs import params, net_conf
from configs.net_conf import available_models
from configs.params import available_datasets
from models.model_factory import ModelFactory
from utils import tools


def main():
    run_this_model = available_models[4]
    text_match_model = ModelFactory.make_model(run_this_model)
    hyperparams = net_conf.get_hyperparams(run_this_model)
    hyperparams.batch_size = 1
    dataset_name = available_datasets[2]
    dataset_params = params.get_dataset_params(dataset_name)
    tools.train_model(text_match_model, hyperparams, dataset_params)

if __name__ == '__main__':
    main()
