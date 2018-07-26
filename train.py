import os
import time

import net_conf
import params
import tools
from models import BasicModel


def main():
    train_start = float(time.time())
    run_which_model = net_conf.RUN_WHICH_MODEL
    text_match_model = BasicModel.make_model(run_which_model)
    which_language = net_conf.WHICH_LANGUAGE

    if which_language == 'en':
        text_match_model.setup(raw_fname=params.PROCESSED_EN_TRAIN_URL,
                               train_fname=params.EN_TRAIN_URL,
                               val_fname=params.EN_VAL_URL,
                               test_fname=params.EN_TEST_URL,
                               pretrained_word_vecs_fname=params.PROCESSED_EN_WORD_VEC_URL)
    else:
        text_match_model.setup(raw_fname=params.PROCESSED_ES_TRAIN_URL,
                               train_fname=params.ES_TRAIN_URL,
                               val_fname=params.ES_VAL_URL,
                               test_fname=params.ES_TEST_URL,
                               pretrained_word_vecs_fname=params.PROCESSED_ES_WORD_VEC_URL)
    text_match_model.build()
    text_match_model.compile()
    text_match_model.fit_generator()

    train_end = float(time.time())
    train_time = train_end - train_start
    print('================ Train end ================')
    print('Train time: {.2f}'.format(train_time))

    text_match_model.evaluate_generator()

    current_time = tools.get_current_time()
    save_url = params.MODEL_SAVE_DIR + os.path.sep + run_which_model + '_' + which_language + '_' + current_time
    text_match_model.save(save_url)

if __name__ == '__main__':
    main()
