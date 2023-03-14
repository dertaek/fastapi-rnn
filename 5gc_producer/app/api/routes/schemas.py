"""
定义类型对象文件
"""
from typing import Optional, List
from pydantic import BaseModel
from enum import Enum


# 预测算法类型
class PredictMethod(str, Enum):
    RNN = "RNN"


# 预测网元类型
class NetElementType(str, Enum):
    AMF = 'AMF'
    SMF = 'SMF'
    UPF = 'UPF'
    UDM = 'UDM'
    SWITCH = 'SWITCH'
    INTERFACE = 'INTERFACE'


# 预测任务schema
class PredictJobConfig(str, Enum):
    resource_type = "resource_type"
    predict_method = "predict_method"
    predict_history = "predict_history"
    predict_sleep_time = "predict_sleep_time"


# 异常检测任务schema
class AbnormalJobConfig(str, Enum):
    abnormal_sleep_time = "abnormal_sleep_time"
    resource_type = "resource_type"


# 修改预测参数schema
class ChangePredictJobConfig(BaseModel):
    config_names: List[PredictJobConfig]
    resource_type: Optional[List[NetElementType]] = ['AMF', 'SMF', 'UPF', 'UDM', 'SWITCH']
    predict_method: Optional[PredictMethod] = PredictMethod.RNN
    predict_history: Optional[int] = 5
    predict_sleep_time: Optional[int] = 5


# 修改异常检测参数schema
class ChangeAbnormalJobConfig(BaseModel):
    config_names: List[AbnormalJobConfig]
    resource_type: Optional[List[NetElementType]] = ['AMF', 'SMF', 'UPF', 'UDM', 'SWITCH']
    abnormal_sleep_time: Optional[float] = 0.05


# 修改预测参数后返回的响应
class ResponsePredictJobConfig(BaseModel):
    resource_type: List[NetElementType]
    predict_method: PredictMethod
    predict_history: int
    predict_sleep_time: int


# 修改异常检测参数后返回的响应
class ResponseAbnormalJobConfig(BaseModel):
    resource_type: List[NetElementType]
    abnormal_sleep_time: float





