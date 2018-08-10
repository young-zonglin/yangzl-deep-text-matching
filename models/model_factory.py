from configs.net_conf import available_models
from models.avg_seq_dense import AvgSeqDenseModel
from models.basic_model import BasicModel
from models.rnmt_encoder_bilstm_dense import RNMTPlusEncoderBiLSTMDenseModel
from models.stacked_bilstm_dense import StackedBiLSTMDenseModel
from models.transformer_encoder_bilstm_dense import TransformerEncoderBiLSTMDenseModel


class ModelFactory:
    # 静态工厂方法
    @staticmethod
    def make_model(model_name):
        if model_name == available_models[0]:
            return AvgSeqDenseModel()
        elif model_name == available_models[1]:
            return StackedBiLSTMDenseModel()
        elif model_name == available_models[2]:
            return TransformerEncoderBiLSTMDenseModel()
        elif model_name == available_models[3]:
            return RNMTPlusEncoderBiLSTMDenseModel()
        else:
            return BasicModel()
