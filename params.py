import os

PROJECT_ROOT = os.path.dirname(os.path.realpath(__file__))

# raw data
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'raw_data')
CIKM_ENGLISH_TRAIN_DATA = os.path.join(RAW_DATA_PATH, 'cikm_english_train_20180516.txt')
CIKM_SPANISH_TRAIN_DATA = os.path.join(RAW_DATA_PATH, 'cikm_spanish_train_20180516.txt')
CIKM_TEST_DATA = os.path.join(RAW_DATA_PATH, 'cikm_test_a_20180516.txt')
CIMK_UNLABELED_SPANISH_DATA = os.path.join(RAW_DATA_PATH, 'cikm_unlabel_spanish_train_20180516.txt')
fastText_EN_PRE_TRAINED_WIKI_WORD_VEC = os.path.join(RAW_DATA_PATH,
                                                     'fast_text_vectors_wiki.en.vec',
                                                     'wiki.en.vec')
fastText_ES_PRE_TRAINED_WIKI_WORD_VEC = os.path.join(RAW_DATA_PATH,
                                                     'fast_text_vectors_wiki.es.vec',
                                                     'wiki.es.vec')

# processed data
PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed_data')
PROCESSED_EN_TRAIN_DATA = os.path.join(PROCESSED_DATA_PATH, 'processed_en_train')
PROCESSED_ES_TRAIN_DATA = os.path.join(PROCESSED_DATA_PATH, 'processed_es_train')
PROCESSED_EN_WORD_VEC = os.path.join(PROCESSED_DATA_PATH, 'processed.wiki.en.vec')
PROCESSED_ES_WORD_VEC = os.path.join(PROCESSED_DATA_PATH, 'processed.wiki.es.vec')

OPEN_FILE_ENCODING = 'utf-8'
SAVE_FILE_ENCODING = 'utf-8'

BATCH_SAMPLES_NUMBER = 32  # 32 64 128 256

SPLIT_PATTERN_STR = r'[ ,.;"?():!/]+'
MATCH_SINGLE_QUOTE_STR = r'[^a-zA-Z]*(\')[a-zA-Z ]+(\')[^a-zA-Z]+'
