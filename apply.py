import net_conf
import params
from models import BasicModel


def main():
    text_match_model = BasicModel()
    model_url = 'E:\PyCharmProjects\CIKM-AnalytiCup-2018\model_trained\StackedBiLSTMDenseModel_en_2018-07-26 17_06_52.h5'
    text_match_model.load(model_url)
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
    text_match_model.evaluate_generator()

if __name__ == '__main__':
    main()
