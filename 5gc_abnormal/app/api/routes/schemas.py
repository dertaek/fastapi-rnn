from typing import Optional, List, Dict
from pydantic import BaseModel
from enum import Enum


# 预测算法类型
class PredictMethod(str, Enum):
    RNN = "RNN"


class Activation(str, Enum):
    tanh = 'tanh'
    relu = 'relu'


class NetElementType(str, Enum):
    AMF = 'AMF'
    SMF = 'SMF'
    UPF = 'UPF'
    UDM = 'UDM'
    SWITCH = 'SWITCH'
    INTERFACE = 'INTERFACE'


# 预测任务schema
class JobConfig(str, Enum):
    resource_type = "resource_type"
    predict_method = "predict_method"
    predict_history = "predict_history"
    multiprocessing = "multiprocessing"


class ChangePredictJobConfig(BaseModel):
    config_names: List[JobConfig]
    resource_type: Optional[List[NetElementType]] = ['AMF', 'SMF', 'UPF', 'UDM', 'SWITCH']
    predict_method: Optional[PredictMethod] = PredictMethod.RNN
    predict_history: Optional[int] = 7
    multiprocessing: Optional[int] = 7


class ResponsePredictJobConfig(BaseModel):
    resource_type: List[NetElementType]
    predict_method: PredictMethod
    predict_history: int
    multiprocessing: int


class RNNConfig(str, Enum):
    hidden_size = "hidden_size"
    num_layers = "num_layers"
    nonlinearity = "nonlinearity"
    bias = "bias"
    batch_first = "batch_first"
    dropout = "dropout"
    bidirectional = "bidirectional"
    learning_rate = "learning_rate"


class RNNConfigModify(BaseModel):
    type_: NetElementType
    config_names: List[RNNConfig]
    hidden_size: Optional[int] = 32
    num_layers: Optional[int] = 2
    nonlinearity: Optional[Activation] = Activation.relu
    bias: Optional[bool] = True
    batch_first: Optional[bool] = False
    dropout: Optional[float] = 0
    bidirectional: Optional[bool] = False
    learning_rate: Optional[float] = 0.01


class ResponseRNNConfig(BaseModel):
    AMF: Dict
    SMF: Dict
    UPF: Dict
    UDM: Dict
    SWITCH: Dict
    INTERFACE: Dict


class PredictDataConfig(BaseModel):
    type_: NetElementType
    res_id: str
    input_idx_name: str
    method: PredictMethod


class AbnormalConfig(BaseModel):
    type_: NetElementType
    input_idx_name: str




