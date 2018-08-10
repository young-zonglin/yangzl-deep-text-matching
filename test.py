import net_conf
import params
from models import BasicModel
from net_conf import available_models


def test():
    run_this_model = available_models[1]
    text_match_model = BasicModel.make_model(run_this_model)
    hyperparams = net_conf.get_hyperparams(run_this_model)
    hyperparams.batch_size = 1
    text_match_model.setup(raw_fname=params.JUST_FOR_TEST,
                           train_fname=params.JUST_FOR_TEST,
                           val_fname=params.JUST_FOR_TEST,
                           test_fname=params.JUST_FOR_TEST,
                           pretrained_word_vecs_fname=params.PROCESSED_EN_WORD_VEC_URL,
                           hyperparams=hyperparams)
    text_match_model.build()
    text_match_model.compile()
    text_match_model.fit_generator()

    text_match_model.evaluate_generator()

if __name__ == '__main__':
    test()
