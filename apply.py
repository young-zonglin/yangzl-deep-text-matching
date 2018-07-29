import net_conf
import params
from models import BasicModel


def main():
    text_match_model = BasicModel()
    model_url = ''
    text_match_model.load(model_url)
    which_language = net_conf.WHICH_LANGUAGE
    hyperparams = net_conf.get_hyperparams('BasicModel')

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
    text_match_model.evaluate_generator()

if __name__ == '__main__':
    main()
