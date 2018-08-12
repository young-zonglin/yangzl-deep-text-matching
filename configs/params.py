import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
RESULT_SAVE_DIR = os.path.join(PROJECT_ROOT, 'result')

TRAIN_RECORD_FNAME = 'a.train.info.record'
# 注意：后缀应为png，jpeg之类
MODEL_VIS_FNAME = 'a.model.visual.png'

MATCH_PUNCTUATION_STR = r'[ ,.;"?():!/]+'
MATCH_SINGLE_QUOTE_STR = r'[^a-zA-Z]*(\')[a-zA-Z ]*(\')[^a-zA-Z]+'
MATCH_NUMBER_STR = r'^[0-9]+\.*[0-9]*$'
MATCH_ILLEGAL_CHAR_STR = r'^[\']+$'

fastText_EN_WORD_VEC_TOTAL_COUNT = 2519370
fastText_ES_WORD_VEC_TOTAL_COUNT = 985667


class DataSetParams:
    def __init__(self):
        self.current_classname = self.__class__.__name__

        self.open_file_encoding = 'utf-8'
        self.save_file_encoding = 'utf-8'

        self.raw_url = None
        self.train_url = None
        self.val_url = None
        self.test_url = None

        self.pretrained_word_vecs_url = None

    def __str__(self):
        ret_info = list()
        ret_info.append("open file encoding: " + self.open_file_encoding + '\n')
        ret_info.append("save file encoding: " + self.save_file_encoding + '\n\n')

        ret_info.append("raw url: " + str(self.raw_url) + '\n')
        ret_info.append("train url: " + str(self.train_url) + '\n')
        ret_info.append("val url: " + str(self.val_url) + '\n')
        ret_info.append("test url: " + str(self.test_url) + '\n\n')

        ret_info.append("pretrained word vectors url: " + str(self.pretrained_word_vecs_url) + '\n\n')
        return ''.join(ret_info)


class JustForTest(DataSetParams):
    def __init__(self):
        super(JustForTest, self).__init__()
        # just for test
        just_for_test = os.path.join(PROJECT_ROOT, 'data', 'just_for_test')
        self.raw_url = just_for_test
        self.train_url = just_for_test
        self.val_url = just_for_test
        self.test_url = just_for_test

        self.pretrained_word_vecs_url = EN_CIKM_AnalytiCup_2018().pretrained_word_vecs_url

    def __str__(self):
        ret_info = list()
        ret_info.append('================== ' + self.current_classname + ' ==================\n')

        super_str = super(JustForTest, self).__str__()
        return ''.join(ret_info) + super_str


class EN_CIKM_AnalytiCup_2018(DataSetParams):
    def __init__(self):
        DataSetParams.__init__(self)
        # raw data
        self.raw_data_dir = os.path.join(PROJECT_ROOT, 'data', 'cikm_2018', 'raw_data')
        self.cikm_en_train_url = os.path.join(self.raw_data_dir, 'cikm_english_train_20180516.txt')
        self.fastText_en_pretrained_wiki_word_vecs_url = os.path.join(self.raw_data_dir,
                                                                      'fast_text_vectors_wiki.en.vec',
                                                                      'wiki.en.vec')

        # processed data
        self.processed_data_dir = os.path.join(PROJECT_ROOT, 'data', 'cikm_2018', 'processed_data')
        self.processed_en_train_url = os.path.join(self.processed_data_dir, 'processed_en_train')
        self.processed_en_word_vecs_url = os.path.join(self.processed_data_dir, 'processed.wiki.en.vec')

        # raw, train, val, test
        self.raw_url = self.processed_en_train_url
        self.train_url = os.path.join(self.processed_data_dir, 'en_train')
        self.val_url = os.path.join(self.processed_data_dir, 'en_val')
        self.test_url = os.path.join(self.processed_data_dir, 'en_test')

        self.pretrained_word_vecs_url = self.processed_en_word_vecs_url

    def __str__(self):
        ret_info = list()
        ret_info.append('================== '+self.current_classname+' ==================\n')
        ret_info.append("raw data dir: " + self.raw_data_dir + '\n')
        ret_info.append("processed data dir: " + self.processed_data_dir + '\n\n')

        super_str = super(EN_CIKM_AnalytiCup_2018, self).__str__()
        return ''.join(ret_info) + super_str


class ES_CIKM_AnalytiCup_2018(DataSetParams):
    def __init__(self):
        DataSetParams.__init__(self)
        # raw data
        self.raw_data_dir = os.path.join(PROJECT_ROOT, 'data', 'cikm_2018', 'raw_data')
        self.cikm_es_train_url = os.path.join(self.raw_data_dir, 'cikm_spanish_train_20180516.txt')
        self.fastText_es_pretrained_wiki_word_vecs_url = os.path.join(self.raw_data_dir,
                                                                      'fast_text_vectors_wiki.es.vec',
                                                                      'wiki.es.vec')

        # processed data
        self.processed_data_dir = os.path.join(PROJECT_ROOT, 'data', 'cikm_2018', 'processed_data')
        self.processed_es_train_url = os.path.join(self.processed_data_dir, 'processed_es_train')
        self.processed_es_word_vecs_url = os.path.join(self.processed_data_dir, 'processed.wiki.es.vec')

        # raw, train, val, test
        self.raw_url = self.processed_es_train_url
        self.train_url = os.path.join(self.processed_data_dir, 'es_train')
        self.val_url = os.path.join(self.processed_data_dir, 'es_val')
        self.test_url = os.path.join(self.processed_data_dir, 'es_test')

        self.pretrained_word_vecs_url = self.processed_es_word_vecs_url

        # currently not used data
        self.cikm_test_url = os.path.join(self.raw_data_dir, 'cikm_test_a_20180516.txt')
        self.cikm_unlabeled_es_url = os.path.join(self.raw_data_dir, 'cikm_unlabel_spanish_train_20180516.txt')

    def __str__(self):
        ret_info = list()
        ret_info.append('================== '+self.current_classname+' ==================\n')
        ret_info.append("raw data dir: " + self.raw_data_dir + '\n')
        ret_info.append("processed data dir: " + self.processed_data_dir + '\n\n')

        super_str = super(ES_CIKM_AnalytiCup_2018, self).__str__()
        return ''.join(ret_info) + super_str


dataset_name_abbr_full = {'cikm_en': EN_CIKM_AnalytiCup_2018().__class__.__name__,
                          'cikm_es': ES_CIKM_AnalytiCup_2018().__class__.__name__,
                          'just_for_test': JustForTest().__class__.__name__}
dataset_name_full_abbr = {v: k for k, v in dataset_name_abbr_full.items()}
available_datasets = ['cikm_en', 'cikm_es', 'just_for_test']


def get_dataset_params(dataset_name):
    if dataset_name == available_datasets[0]:
        return EN_CIKM_AnalytiCup_2018()
    elif dataset_name == available_datasets[1]:
        return ES_CIKM_AnalytiCup_2018()
    elif dataset_name == available_datasets[2]:
        return JustForTest()
    else:
        return DataSetParams()


if __name__ == '__main__':
    print(DataSetParams())
    print(EN_CIKM_AnalytiCup_2018())
    print(ES_CIKM_AnalytiCup_2018())
    print(JustForTest())
