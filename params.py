import os

PROJECT_ROOT = os.path.dirname(os.path.realpath(__file__))
RESULT_SAVE_DIR = os.path.join(PROJECT_ROOT, 'result')
MODEL_SAVE_DIR = ''

TRAIN_RECORD_FNAME = 'a.train.info.record'
# 注意：后缀应为png，jpeg之类
MODEL_VIS_FNAME = 'a.model.visual.png'

# raw data
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw_data')
CIKM_ENGLISH_TRAIN_URL = os.path.join(RAW_DATA_DIR, 'cikm_english_train_20180516.txt')
CIKM_SPANISH_TRAIN_URL = os.path.join(RAW_DATA_DIR, 'cikm_spanish_train_20180516.txt')
CIKM_TEST_URL = os.path.join(RAW_DATA_DIR, 'cikm_test_a_20180516.txt')
CIMK_UNLABELED_SPANISH_URL = os.path.join(RAW_DATA_DIR, 'cikm_unlabel_spanish_train_20180516.txt')
fastText_EN_PRE_TRAINED_WIKI_WORD_VEC_URL = os.path.join(RAW_DATA_DIR,
                                                         'fast_text_vectors_wiki.en.vec',
                                                         'wiki.en.vec')
fastText_ES_PRE_TRAINED_WIKI_WORD_VEC_URL = os.path.join(RAW_DATA_DIR,
                                                         'fast_text_vectors_wiki.es.vec',
                                                         'wiki.es.vec')

# processed data
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed_data')
PROCESSED_EN_TRAIN_URL = os.path.join(PROCESSED_DATA_DIR, 'processed_en_train')
PROCESSED_ES_TRAIN_URL = os.path.join(PROCESSED_DATA_DIR, 'processed_es_train')
PROCESSED_EN_WORD_VEC_URL = os.path.join(PROCESSED_DATA_DIR, 'processed.wiki.en.vec')
PROCESSED_ES_WORD_VEC_URL = os.path.join(PROCESSED_DATA_DIR, 'processed.wiki.es.vec')

# train, val, test
EN_TRAIN_URL = os.path.join(PROCESSED_DATA_DIR, 'en_train')
EN_VAL_URL = os.path.join(PROCESSED_DATA_DIR, 'en_val')
EN_TEST_URL = os.path.join(PROCESSED_DATA_DIR, 'en_test')

ES_TRAIN_URL = os.path.join(PROCESSED_DATA_DIR, 'es_train')
ES_VAL_URL = os.path.join(PROCESSED_DATA_DIR, 'es_val')
ES_TEST_URL = os.path.join(PROCESSED_DATA_DIR, 'es_test')

OPEN_FILE_ENCODING = 'utf-8'
SAVE_FILE_ENCODING = 'utf-8'

MATCH_PUNCTUATION_STR = r'[ ,.;"?():!/]+'
MATCH_SINGLE_QUOTE_STR = r'[^a-zA-Z]*(\')[a-zA-Z ]*(\')[^a-zA-Z]+'
MATCH_NUMBER_STR = r'^[0-9]+\.*[0-9]*$'
MATCH_ILLEGAL_CHAR_STR = r'^[\']+$'

fastText_EN_WORD_VEC_TOTAL_COUNT = 2519370
fastText_ES_WORD_VEC_TOTAL_COUNT = 985667
