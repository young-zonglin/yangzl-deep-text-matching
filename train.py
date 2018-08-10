from configs import net_conf
from models.model_factory import ModelFactory
from utils import tools


def main():
    run_which_model = net_conf.RUN_WHICH_MODEL
    which_language = net_conf.WHICH_LANGUAGE
    text_match_model = ModelFactory.make_model(run_which_model)
    hyperparams = net_conf.get_hyperparams(run_which_model)
    tools.train_model(text_match_model, hyperparams, which_language)


if __name__ == '__main__':
    main()
