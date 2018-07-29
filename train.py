import os

import net_conf
import params
import tools
from models import BasicModel


def main():
    run_which_model = net_conf.RUN_WHICH_MODEL
    text_match_model = BasicModel.make_model(run_which_model)
    which_language = net_conf.WHICH_LANGUAGE
    hyperparams = net_conf.get_hyperparams(run_which_model)

    if which_language == 'en':
        text_match_model.setup(raw_fname=params.PROCESSED_EN_TRAIN_URL,
                               train_fname=params.EN_TRAIN_URL,
                               val_fname=params.EN_VAL_URL,
                               test_fname=params.EN_TEST_URL,
                               pretrained_word_vecs_fname=params.PROCESSED_EN_WORD_VEC_URL,
                               hyperparams=hyperparams)
    else:
        text_match_model.setup(raw_fname=params.PROCESSED_ES_TRAIN_URL,
                               train_fname=params.ES_TRAIN_URL,
                               val_fname=params.ES_VAL_URL,
                               test_fname=params.ES_TEST_URL,
                               pretrained_word_vecs_fname=params.PROCESSED_ES_WORD_VEC_URL,
                               hyperparams=hyperparams)
    text_match_model.build()
    text_match_model.compile()
    text_match_model.fit_generator()

    text_match_model.evaluate_generator()

    # current_time = tools.get_current_time()
    # save_url = \
    #     params.MODEL_SAVE_DIR + os.path.sep + \
    #     run_which_model + '_' + which_language + '_' + current_time + '.h5'
    # text_match_model.save(save_url)

if __name__ == '__main__':
    main()
